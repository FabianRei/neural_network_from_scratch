import pickle


def loadMnist(path):
    mnist = pickle.load(open(path, 'rb'))
    x_train = mnist["training_images"]
    y_train = mnist["training_labels"]
    x_test = mnist["test_images"]
    y_test = mnist["test_labels"]
    print(f'loaded Mnist from {path}.')
    return x_train, y_train, x_test, y_test
