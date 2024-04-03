import struct
from array import array
import numpy as np


class MNISTDataloader(object):
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def convert_labels_to_one_hot(self, labels):
        """
        Convert labels to one hot encoding
        :param labels: list of labels
        :return: one hot encoded labels"""

        one_hot_labels = np.zeros((len(labels), 10))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1
        return one_hot_labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    mnist_dataloader = MNISTDataloader(
        "data/train-images.idx3-ubyte",
        "data/train-labels.idx1-ubyte",
        "data/t10k-images.idx3-ubyte",
        "data/t10k-labels.idx1-ubyte",
    )

    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    y_train = mnist_dataloader.convert_labels_to_one_hot(y_train)
    y_test = mnist_dataloader.convert_labels_to_one_hot(y_test)
    print(np.asarray(x_train[0]).shape)
    print(np.asarray(y_train[0]))
