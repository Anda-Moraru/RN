import numpy as np
from torchvision.datasets import MNIST
import time


def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

train_X = np.array(train_X) / 255.0
test_X = np.array(test_X) / 255.0

def one_hot_encode(y, nr_classes=10):
    nr_examples = len(y)
    one_hot_labels = np.zeros((nr_examples, nr_classes))

    for i in range(nr_examples):
        one_hot_labels[i, y[i]] = 1

    return one_hot_labels

train_Y = one_hot_encode(train_Y)
test_Y = one_hot_encode(test_Y)

def create_batches(X, y, batch_size=100):
    num_batches = X.shape[0] // batch_size
    for i in range(num_batches):
        X_batch = X[i * batch_size:(i + 1) * batch_size]
        y_batch = y[i * batch_size:(i + 1) * batch_size]
        yield X_batch, y_batch


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(X, W, b):
    z = np.dot(X, W) + b
    return softmax(z)

def cross_entropy_loss(predictions, targets):
    return -np.mean(np.sum(targets * np.log(predictions), axis=1))


def backward(X_batch, y_batch, predictions, W, b, learning_rate=0.01):
    m = X_batch.shape[0]

    error = y_batch - predictions

    W_update = np.dot(X_batch.T, error)
    W = W + learning_rate * W_update / m

    b_update = np.sum(error, axis=0)
    b = b + learning_rate * b_update / m

    return W, b


def train_perceptron(X, y, epochs=50, batch_size=100, learning_rate=0.01):
    start_time = time.time()
    nr_inputs = X.shape[1]
    nr_classes = y.shape[1]

    W = np.random.randn(nr_inputs, nr_classes) * 0.01
    b = np.zeros((nr_classes, ))

    for epoch in range(epochs):
        epoch_loss = 0

        for X_batch, y_batch in create_batches(X, y, batch_size):
            predictions = forward(X_batch, W, b)

            loss = cross_entropy_loss(predictions, y_batch)
            epoch_loss += loss

            W, b = backward(X_batch, y_batch, predictions, W, b, learning_rate)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / (X.shape[0] // batch_size)}')

        end_time = time.time()
        print('Training duration: %.2f seconds' % (end_time - start_time))

    return W, b

def evaluate(X, y, W, b):
    predictions = forward(X, W, b)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return accuracy


W, b = train_perceptron(train_X, train_Y, epochs=200, batch_size=100, learning_rate=0.01)

accuracy = evaluate(test_X, test_Y, W, b)

