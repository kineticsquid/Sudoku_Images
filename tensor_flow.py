import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

# :todo https://www.tensorflow.org/datasets/overview

def main():
    mnist_train = tfds.load(name="mnist", split="train")
    assert isinstance(mnist_train, tf.data.Dataset)
    print(mnist_train)

    mnist = tfds.load("mnist:1.*.*")
    print(mnist)

    for mnist_example in mnist_train.take(1):  # Only take a single example
        image, label = mnist_example["image"], mnist_example["label"]

        plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
        print("Label: %d" % label.numpy())


if __name__ == '__main__':
    main()
