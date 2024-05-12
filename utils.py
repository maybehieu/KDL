import struct
from array import array
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OneHotEncoder,
    LabelBinarizer,
)
from sklearn.model_selection import train_test_split
import pandas as pd
from ucimlrepo import fetch_ucirepo


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


class Dataloader:
    def __init__(self, X, y, name):
        self.name = name
        if name == "writing":
            # Filter only classes 0 and 1
            mask = (y == "A") | (y == "B")
            X, y = X[mask], y[mask]

            # Scale the features to have zero mean and unit variance
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Split the data into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Convert the labels to 0 and 1
            self.y_train = np.where(self.y_train == "A", 0, 1)
            self.y_test = np.where(self.y_test == "A", 0, 1)

        elif name == "mnist":
            mnist_dataloader = MNISTDataloader(
                "data/train-images.idx3-ubyte",
                "data/train-labels.idx1-ubyte",
                "data/t10k-images.idx3-ubyte",
                "data/t10k-labels.idx1-ubyte",
            )

            (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
            X_train = np.asarray(x_train)
            X_test = np.asarray(x_test)
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)

            # Filter only classes 0 and 1
            l1 = 0
            l2 = 2
            mask = (y_train == l1) | (y_train == l2)
            X_train, y_train = X_train[mask], y_train[mask]
            mask = (y_test == l1) | (y_test == l2)
            X_test, y_test = X_test[mask], y_test[mask]

            nsamples, nx, ny = X_train.shape
            X_train = X_train.reshape((nsamples, nx * ny))
            nsamples, nx, ny = X_test.shape
            X_test = X_test.reshape((nsamples, nx * ny))

            # Normalize the features
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(X_train)
            self.X_test = scaler.transform(X_test)

            # One-hot encode the labels
            le = OneHotEncoder()
            self.y_train = le.fit_transform(y_train.reshape(-1, 1)).toarray()
            self.y_test = le.transform(y_test.reshape(-1, 1)).toarray()

        elif name == "iris":
            print("fetching iris dataset")
            # fetch dataset
            iris = fetch_ucirepo(id=53)
            # labels: 'Iris-setosa' 'Iris-versicolor' 'Iris-virginica'

            # data (as pandas dataframes)
            X = iris.data.features
            y = iris.data.targets
            X = np.asarray(X)
            y = np.asarray(y).flatten()

            print(X.shape, y.shape)

            l1 = "Iris-setosa"
            l2 = "Iris-versicolor"

            # Filter only classes l1 and l2
            mask = (y == l1) | (y == l2)
            X, y = X[mask], y[mask]

            # Scale the features to have zero mean and unit variance
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # One-hot encode the labels
            le = OneHotEncoder()
            self.y_train = le.fit_transform(self.y_train.reshape(-1, 1)).toarray()
            self.y_test = le.transform(self.y_test.reshape(-1, 1)).toarray()

        elif name == "mushroom":
            print("fetching mushroom dataset")
            # fetch dataset
            iris = fetch_ucirepo(id=73)
            # labels: 'e' 'p'

            # data (as pandas dataframes)
            X = iris.data.features
            y = iris.data.targets

            le = LabelEncoder()
            # Iterate over all columns in the dataframe
            for col in X.columns:
                # Only convert columns with object type
                if X[col].dtype == "object":
                    # Use LabelEncoder to do the numeric transformation
                    X.loc[:, col] = le.fit_transform(X[col])

            X = np.asarray(X)
            y = np.asarray(y).flatten()

            # Scale the features to have zero mean and unit variance
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # One-hot encode the labels
            le = OneHotEncoder()
            self.y_train = le.fit_transform(self.y_train.reshape(-1, 1)).toarray()
            self.y_test = le.transform(self.y_test.reshape(-1, 1)).toarray()

        # pre-process to match SVM metrics
        self.y_train[self.y_train == 0] = -1
        self.y_test[self.y_test == 0] = -1

        print("Dataset: ", self.name)

        # print a preview of the data
        print("X_train: ", self.X_train[:2])
        print("y_train: ", self.y_train[:2])
        print("X_test: ", self.X_test[:2])
        print("y_test: ", self.y_test[:2])

        # flatten to export to cpp
        self.X_train_shape = self.X_train.shape
        self.X_test_shape = self.X_test.shape
        self.y_train_shape = self.y_train.shape
        self.y_test_shape = self.y_test.shape
        print(
            self.X_train_shape, self.y_train_shape, self.X_test_shape, self.y_test_shape
        )
        self.X_train = self.X_train.flatten()
        self.X_test = self.X_test.flatten()
        self.y_train = self.y_train.flatten()
        self.y_test = self.y_test.flatten()

    def print_array_as_string(self, shape, arr, fname=""):
        # print flatten arr as a string to a file
        print(f"Exported shape: {arr.shape}")
        if len(shape) == 1:
            formatted_strings = f"{shape[0]}, 1, "
        else:
            formatted_strings = f"{shape[0]}, {shape[1]}, "
        row_string = ", ".join(map(str, arr))
        formatted_strings += row_string

        print(formatted_strings, file=open(fname, "w"))

    def export_to_cpp(self):
        self.print_array_as_string(
            self.X_train_shape, self.X_train, f"{self.name}_xtrain.txt"
        )
        self.print_array_as_string(
            self.y_train_shape, self.y_train, f"{self.name}_ytrain.txt"
        )
        self.print_array_as_string(
            self.X_test_shape, self.X_test, f"{self.name}_xtest.txt"
        )
        self.print_array_as_string(
            self.y_test_shape, self.y_test, f"{self.name}_ytest.txt"
        )

    def export_to_python(self):
        return self.X_train, self.y_train, self.X_test, self.y_test


if __name__ == "__main__":
    datasets = ["mnist", "iris", "mushroom"]
    for dataset in datasets:
        loader = Dataloader(None, None, dataset)
        print(f"Exporting {dataset} to cpp... ")
        loader.export_to_cpp()
