#include "categories.h"

#include <argparse/argparse.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rbln/rbln.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <tuple>
#include <vector>

constexpr int kThread = 2;
cv::Mat image;

cv::Mat GetSquareImage(const cv::Mat &img, int target_width) {
  int width = img.cols, height = img.rows;

  cv::Mat square =
      cv::Mat(target_width, target_width, img.type(), cv::Vec3b(117, 117, 117));

  int max_dim = (width >= height) ? width : height;
  float scale = (static_cast<float>(target_width)) / max_dim;
  cv::Rect roi;
  // only support in case of width is bigger than height
  roi.width = target_width;
  roi.x = 0;
  roi.height = height * scale;
  roi.y = (target_width - roi.height) / 2;

  cv::resize(img, square(roi), roi.size());

  return square;
}

cv::Rect ScaleBox(cv::Rect pred, cv::Size img_size, int pred_size) {
  int img_long_axis = std::max(img_size.height, img_size.width);
  float scale = img_long_axis / (static_cast<float>(pred_size));
  pred.x *= scale;
  pred.y *= scale;
  pred.width *= scale;
  pred.height *= scale;
  auto padding_x = (img_long_axis - img_size.width) / 2;
  auto padding_y = (img_long_axis - img_size.height) / 2;
  pred.x -= padding_x;
  pred.y -= padding_y;
  return pred;
}

void PostProcess(const std::vector<cv::Mat>& outputs, int n_proposal, int n_size) {
  const auto& logits = outputs[0];
  // Run NMS
  std::vector<cv::Rect> nms_boxes;
  std::vector<float> nms_confidences;
  std::vector<size_t> nms_class_ids;
  for (size_t i = 0; i < n_proposal; i++) {
    auto cx = logits.at<float>(0, 0, i);
    auto cy = logits.at<float>(0, 1, i);
    auto w = logits.at<float>(0, 2, i);
    auto h = logits.at<float>(0, 3, i);
    auto x = cx - w / 2;
    auto y = cy - h / 2;
    cv::Rect rect{static_cast<int>(x), static_cast<int>(y), static_cast<int>(w),
                  static_cast<int>(h)};
    float confidence = std::numeric_limits<float>::min();
    int cls_id;
    for (size_t j = 4; j < n_size; j++) {
      if (confidence < logits.at<float>(0, j, i)) {
        confidence = logits.at<float>(0, j, i);
        cls_id = j - 4;
      }
    }
    nms_boxes.push_back(rect);
    nms_confidences.push_back(confidence);
    nms_class_ids.push_back(cls_id);
  }
  std::vector<int> nms_indices;
  cv::dnn::NMSBoxes(nms_boxes, nms_confidences, 0.25f, 0.45f, nms_indices);

  // Draw output image
  cv::Mat output_img = image.clone();
  for (size_t i = 0; i < nms_indices.size(); i++) {
    auto idx = nms_indices[i];
    auto class_id = nms_class_ids[idx];
    auto scaled_box = ScaleBox(nms_boxes[idx], output_img.size(), 640);
    cv::rectangle(output_img, scaled_box, cv::Scalar(255, 0, 0));
    std::stringstream ss;
    ss << COCO_CATEGORIES[class_id] << ": " << nms_confidences[idx];
    cv::putText(output_img, ss.str(), scaled_box.tl() - cv::Point(0, 1),
                cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
  }
  std::string file_name = "result.jpg";
  cv::imwrite(file_name, output_img);
}

int main(int argc, char **argv) {
  // Parse arguments
  argparse::ArgumentParser program("object_detection");
  program.add_argument("-i", "--input")
      .default_value(std::string("people4.jpg"))
      .help("specify the input image file.");
  program.add_argument("-m", "--model")
      .default_value(std::string("yolov8n.rbln"))
      .help("specify the model file. (.rbln)");
  program.add_argument("-o", "--output")
      .default_value("output.bin")
      .help("specify the output tensor file.");
  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }
  auto input_path = program.get<std::string>("--input");
  auto model_path = program.get<std::string>("--model");
  auto output_path = program.get<std::string>("--output");

  // Read images
  try {
    image = cv::imread(input_path);
  } catch (const cv::Exception &err) {
    std::cerr << err.what() << std::endl;
    std::exit(1);
  }

  // Convert image to tensor
  cv::Mat blob =
      cv::dnn::blobFromImage(GetSquareImage(image, 640), 1. / 255., cv::Size(),
                             cv::Scalar(), true, false, CV_32F);

  std::ofstream file("c_preprocessed_input.bin", std::ios::binary);
  file.write(reinterpret_cast<const char *>(blob.data),
             blob.total() * blob.elemSize());
  file.close();

  // Create model and runtime
  RBLNModel *mod = rbln_create_model(model_path.c_str());
  RBLNRuntime *rt = rbln_create_async_runtime(mod, "default", 0, 0);

  auto n_out = rbln_get_num_outputs(rt);

  // Allocate buffer for Async
  std::vector<void*> inputs = {blob.data};
  std::vector<cv::Mat> outputs(n_out);
  std::vector<char *> output_ptrs(n_out);
  for (auto idx = 0; idx < n_out; idx++) {
    const RBLNTensorLayout *layout = rbln_get_output_layout(rt, idx);
    outputs[idx] = cv::Mat{layout->ndim, layout->shape, CV_32F};
    output_ptrs[idx] = reinterpret_cast<char *>(outputs[idx].data);
  }

  // Run async execution
  int rid = rbln_async_run(rt, inputs.data(), output_ptrs.data());

  rbln_async_wait(rt, rid, 1000);

  const RBLNTensorLayout *layout = rbln_get_output_layout(rt, 0);
  int n_proposal = layout->shape[2];
  int n_size = layout->shape[1]; 
  PostProcess(outputs, n_proposal, n_size);

  rbln_destroy_runtime(rt);
  rbln_destroy_model(mod);

  return 0;
}
