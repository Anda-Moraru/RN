import numpy as np
from torchvision.datasets import MNIST

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data', transform=lambda x: np.array(x).flatten(), download=True, train=is_train)
    mnist_data, mnist_labels = [], []
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

train_Y, test_Y = one_hot_encode(train_Y), one_hot_encode(test_Y)

def create_batches(X, y, batch_size=100):
    num_batches = X.shape[0] // batch_size
    for i in range(num_batches):
        X_batch = X[i * batch_size:(i + 1) * batch_size]
        y_batch = y[i * batch_size:(i + 1) * batch_size]
        yield X_batch, y_batch

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    hidden_activations = sigmoid(Z1)
    Z2 = np.dot(hidden_activations, W2) + b2
    output_activations = softmax(Z2)
    return Z1, hidden_activations, Z2, output_activations

def cross_entropy_loss(predictions, targets, W1, W2, lambda_reg):
    n = targets.shape[0]
    base_loss = -np.sum(targets * np.log(predictions)) / n
    reg_loss = (lambda_reg / (2 * n)) * (np.sum(W1**2) + np.sum(W2**2))
    return base_loss + reg_loss

def backward(X, Y, Z1, hidden_activations, Z2, output_activations, W1, W2, b1, b2, learning_rate, lambda_reg):
    m = X.shape[0]

    delta_L = output_activations - Y

    dW2 = (1 / m) * np.dot(hidden_activations.T, delta_L) + (lambda_reg / m) * W2
    db2 = (1 / m) * np.sum(delta_L, axis=0, keepdims=True)

    delta_l = np.dot(delta_L, W2.T) * (hidden_activations * (1 - hidden_activations))

    dW1 = (1 / m) * np.dot(X.T, delta_l) + (lambda_reg / m) * W1
    db1 = (1 / m) * np.sum(delta_l, axis=0)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2

def train_mlp(X, Y, epochs=50, batch_size=100, learning_rate=0.01, lambda_reg=0.01):
    input_size, hidden_size, output_size = 784, 100, 10
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / (input_size + hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / (hidden_size + output_size))
    b2 = np.zeros((1, output_size))

    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, Y_batch in create_batches(X, Y, batch_size):
            Z1, hidden_activations, Z2, output_activations = forward(X_batch, W1, b1, W2, b2)
            loss = cross_entropy_loss(output_activations, Y_batch, W1, W2, lambda_reg)
            epoch_loss += loss
            W1, b1, W2, b2 = backward(X_batch, Y_batch, Z1, hidden_activations, Z2, output_activations, W1, W2, b1, b2, learning_rate, lambda_reg)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / (X.shape[0] // batch_size):.4f}')

    return W1, b1, W2, b2

def evaluate(X, Y, W1, b1, W2, b2):
    _, _, _, output_activations = forward(X, W1, b1, W2, b2)
    predictions = np.argmax(output_activations, axis=1)
    labels = np.argmax(Y, axis=1)
    accuracy = np.mean(predictions == labels)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')
    return accuracy

W1, b1, W2, b2 = train_mlp(train_X, train_Y, epochs=200, batch_size=100, learning_rate=0.01, lambda_reg=0.005)
accuracy = evaluate(test_X, test_Y, W1, b1, W2, b2)
