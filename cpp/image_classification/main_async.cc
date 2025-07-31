#include "categories.h"

#include <argparse/argparse.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rbln/rbln.h>

#include <fstream>
#include <iostream>
#include <numeric>

// Post-process function to find and print the predicted category
void post_process(const std::vector<float>& logits) {
    size_t max_idx = 0;
    float max_val = std::numeric_limits<float>::min();
    for (size_t i = 0; i < 1000; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    std::cout << "Predicted category: " << IMAGENET_CATEGORIES[max_idx]
              << std::endl;
}

int main(int argc, char **argv) {
  // Parse arguments
  argparse::ArgumentParser program("image_classification");
  program.add_argument("-i", "--input")
      .default_value(std::string("tabby.jpg"))
      .help("specify the input image file.");
  program.add_argument("-m", "--model")
      .default_value(std::string("resnet18.rbln"))
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

  std::ofstream file("c_preprocessed_input.bin", std::ios::binary);
  file.write(reinterpret_cast<const char *>(blob.data),
             blob.total() * blob.elemSize());
  file.close();

  // Create model and runtime
  RBLNModel *mod = rbln_create_model(model_path.c_str());
  RBLNRuntime *rt = rbln_create_async_runtime(mod, "default", 0, 0);
  rbln_model_description(rt);

  // Alloc output buffer
  std::vector<float> logits(rbln_get_layout_nbytes(rbln_get_output_layout(rt, 0))/sizeof(float));

  std::vector<void*> inputs = {blob.data};
  std::vector<void*> outputs = {logits.data()};

  // Run async execution
  int rid = rbln_async_run(rt, inputs.data(), outputs.data());

  // Wait inference done
  rbln_async_wait(rt, rid, 1000); // 1000 ms

  // Postprocess
  post_process(logits);

  rbln_destroy_runtime(rt);
  rbln_destroy_model(mod);

  return 0;
}
