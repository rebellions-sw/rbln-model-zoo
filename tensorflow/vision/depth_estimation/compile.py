import rebel
from huggingface_hub import from_pretrained_keras

import tensorflow as tf


def main():
    input_h = 480
    input_w = 640

    model = from_pretrained_keras("keras-io/monocular-depth-estimation")

    func = tf.function(lambda input_img: model(input_img))

    # for rebel compile
    input_info = [("input_img", [1, input_h, input_w, 3], tf.float32)]
    compiled_model = rebel.compile_from_tf_function(func, input_info)

    compiled_model.save("monocular_depth_estimation.rbln")


if __name__ == "__main__":
    main()
