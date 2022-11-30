#!usr/bin/env python

DATA_DIR = "dataset/"

TEST_DATA_FILENAME = f"{DATA_DIR}t10k-images-idx3-ubyte"
TRAIN_DATA_FILENAME = f"{DATA_DIR}train-images-idx3-ubyte"
TEST_LABELS_FILENAME = f"{DATA_DIR}t10k-labels-idx1-ubyte"
TRAIN_LABELS_FILENAME = f"{DATA_DIR}train-labels-idx1-ubyte"


def bytes_to_int(byte):
    return int.from_bytes(byte, "big")


def read_images(filename, number_max_img=None):
    images = [] 
    with open(filename, 'rb') as f:
        _ = f.read(4) # magic number
        number_images = bytes_to_int(f.read(4))
        number_rows = bytes_to_int(f.read(4))
        number_columns = bytes_to_int(f.read(4))
        if number_max_img:
            number_images = number_max_img
        for image_idx in range(number_images):
            image = []
            for row_idx in range(number_rows):
                row = []
                for col_inx in range(number_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images


def read_labels(filename, number_max_labels=None):
    labels = [] 
    with open(filename, 'rb') as f:
        _ = f.read(4) # magic number
        number_labels = bytes_to_int(f.read(4))
        if number_max_labels:
            number_labels = number_max_labels
        for lables_idx in range(number_labels):
            label = f.read(1)
            labels.append(label)
    return labels


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]


def extract_features_from_sample(sample):
    return flatten_list(sample)


def extract_features(X):
    return [flatten_list(sample) for sample in X]


def main():
    X_train = read_images(TRAIN_DATA_FILENAME, 100)
    y_train = read_labels(TRAIN_LABELS_FILENAME)
    X_test = read_images(TEST_DATA_FILENAME, 100)
    y_test = read_labels(TEST_LABELS_FILENAME)
    
    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    pass


if __name__ == "__main__":
    main()