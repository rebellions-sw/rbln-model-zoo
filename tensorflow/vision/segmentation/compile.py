import rebel
from huggingface_hub import from_pretrained_keras

import tensorflow as tf


def main():
    input_size = 512

    model = from_pretrained_keras("keras-io/deeplabv3p-resnet50")

    func = tf.function(lambda input_img: model(input_img))

    input_info = [("input_img", [1, input_size, input_size, 3], tf.float32)]
    compiled_model = rebel.compile_from_tf_function(func, input_info)

    compiled_model.save("deeplabv3.rbln")


if __name__ == "__main__":
    main()
