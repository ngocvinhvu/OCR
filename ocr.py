#!usr/bin/env python

DEBUG = True

if DEBUG:
    from PIL import Image
    import numpy as np

    def read_image(path):
        return np.asarray(Image.open(path).convert("L"))
    
    def write_image(image, path):
        img = Image.fromarray(np.array(image), "L")
        img.save(path)

DATA_DIR = "dataset/"
TEST_DIR = "test/"
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


def get_training_distance_for_test_sample(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]


def dist(x, y):
    return sum(
        [(bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2 for x_i, y_i in zip(x, y)]
    ) ** 0.5


def get_most_frequent_element(l):
    return max(l, key=l.count)

def knn(X_train, y_train, X_test, k=3): # need to return y_test
    y_pred = []
    for test_sample_idx, test_sample in enumerate(X_test):
        training_distances = get_training_distance_for_test_sample(X_train, test_sample)
        sorted_distance_indices = [
            pair[0] for pair in sorted(enumerate(training_distances), key=lambda x: x[1])
        ]
        candidates = [
            bytes_to_int(y_train[idx]) for idx in sorted_distance_indices[:k]
        ]
        top_candidate = get_most_frequent_element(candidates)
        y_pred.append(top_candidate)
    return y_pred


def main():
    X_train = read_images(TRAIN_DATA_FILENAME, 1000)
    y_train = read_labels(TRAIN_LABELS_FILENAME, 1000)
    X_test = read_images(TEST_DATA_FILENAME, 5)
    y_test = read_labels(TEST_LABELS_FILENAME, 5)

    for idx, test_sample in enumerate(X_test):
        write_image(test_sample, f"{TEST_DIR}{idx}.png")
    
    X_train = extract_features(X_train)
    X_test = extract_features(X_test)
    y_pred = knn(X_train, y_train, X_test, 3)

    accuracy = sum([int(y_pred_i == bytes_to_int(y_test_i)) for y_pred_i, y_test_i in zip(y_pred, y_test)]) / len(y_test)

    print(y_pred)
    print(accuracy)



if __name__ == "__main__":
    main()