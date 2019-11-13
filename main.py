import numpy as np
from network import Network
from mlxtend.data import loadlocal_mnist
from sklearn import preprocessing


# load data
train_images, train_labels = loadlocal_mnist(
        images_path='C:/Users/Novozhilova Nastya/DeepLearningBP/Data/train-images.idx3-ubyte', 
        labels_path='C:/Users/Novozhilova Nastya/DeepLearningBP/Data/train-labels.idx1-ubyte')
test_images, test_labels = loadlocal_mnist(
        images_path='C:/Users/Novozhilova Nastya/DeepLearningBP/Data/t10k-images.idx3-ubyte', 
        labels_path='C:/Users/Novozhilova Nastya/DeepLearningBP/Data/t10k-labels.idx1-ubyte')

num_classes = 10

# data processing
x_train = preprocessing.normalize(train_images).astype('float32')
y_train = np.eye(num_classes)[train_labels]
x_test = preprocessing.normalize(test_images).astype('float32')
y_test = np.eye(num_classes)[test_labels]

print('Data loaded...')


net = Network(num_nodes_in_layers = [784, 300, 10], 
                 batch_size = 32,
                 num_epochs = 20,
                 learning_rate = 0.1)
net.train(x_train, y_train, x_test, y_test)