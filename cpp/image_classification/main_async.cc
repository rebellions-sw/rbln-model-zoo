#include "categories.h"

#include <argparse/argparse.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rbln/rbln.h>

#include <fstream>
#include <iostream>
#include <numeric>

constexpr int kThread = 2;

void PostProcess(const std::vector<float> &logits, int idx) {
  size_t max_idx = 0;
  float max_val = std::numeric_limits<float>::min();
  for (size_t i = 0; i < 1000; i++) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      max_idx = i;
    }
  }
  std::cout << "Predicted category_" << std::to_string(idx) << ": "
            << IMAGENET_CATEGORIES[max_idx] << std::endl;
}

int main(int argc, char **argv) {
  // Parse arguments
  argparse::ArgumentParser program("image_classification");
  program.add_argument("-i", "--input")
      .required()
      .help("specify the input image file.");
  program.add_argument("-m", "--model")
      .required()
      .help("specify the model file. (.rbln)");
  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }
  auto input_path = program.get<std::string>("--input");
  auto model_path = program.get<std::string>("--model");

  // Preprocess images
  cv::Mat input_image;
  try {
    input_image = cv::imread(input_path);
  } catch (const cv::Exception &err) {
    std::cerr << err.what() << std::endl;
    std::exit(1);
  }
  cv::Mat image;
  cv::cvtColor(input_image, image, cv::COLOR_BGR2RGB);
  // Resize with aspect ratio
  float scale = image.rows < image.cols ? 256. / image.rows : 256. / image.cols;
  cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_LINEAR);
  // Center crop
  image =
      image(cv::Rect((image.cols - 224) / 2, (image.rows - 224) / 2, 224, 224));
  // Normalize image
  image.convertTo(image, CV_32F);
  cv::Vec3f mean(123.68, 116.28, 103.53);
  cv::Vec3f std(58.395, 57.120, 57.385);
  for (unsigned i = 0; i < image.rows; i++) {
    for (unsigned j = 0; j < image.cols; j++) {
      cv::subtract(image.at<cv::Vec3f>(i, j), mean, image.at<cv::Vec3f>(i, j));
      cv::divide(image.at<cv::Vec3f>(i, j), std, image.at<cv::Vec3f>(i, j));
    }
  }
  // Convert image to tensor
  cv::Mat blob = cv::dnn::blobFromImage(image);

  // Create model and runtime
  RBLNModel *mod = rbln_create_model(model_path.c_str());
  RBLNRuntime *rt = rbln_create_async_runtime(mod, "default", 0);

  // Alloc output buffer
  auto buf_size = rbln_get_layout_nbytes(rbln_get_output_layout(rt, 0));
  std::vector<std::vector<float>> outputs(
      kThread, std::vector<float>(buf_size / sizeof(float)));

  // Run async execution
  std::vector<int> rid(kThread, -1);
  for (int idx = 0; idx < kThread; idx++) {
    rid[idx] = rbln_async_run(rt, blob.data, outputs[idx].data());
  }

  // Wait 1st inference done
  rbln_async_wait(rt, rid[0], 1000);

  // Wait 2nd inference done
  rbln_async_wait(rt, rid[1], 1000);

  for (int idx = 0; idx < kThread; idx++) {
    PostProcess(outputs[idx], idx);
  }

  rbln_destroy_runtime(rt);
  rbln_destroy_model(mod);

  return 0;
}
