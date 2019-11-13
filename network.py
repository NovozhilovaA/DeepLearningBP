import numpy as np 
import pickle
import functions
import matplotlib.pyplot as plt
from datetime import datetime

class Network:

    def __init__(self, 
                 num_nodes_in_layers, 
                 batch_size,
                 num_epochs,
                 learning_rate
                 ):

        self.num_nodes_in_layers = num_nodes_in_layers #[input, hidden, output]
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.weight1 = np.array([])
        self.bias1 = np.array([])
        self.weight2 = np.array([])
        self.bias2 = np.array([])

    def initial_weights_biases(self):

        self.weight1 = np.random.normal(0., np.sqrt(2. / self.num_nodes_in_layers[0]), [self.num_nodes_in_layers[1], self.num_nodes_in_layers[0]])
        self.weight2 = np.random.normal(0., np.sqrt(2. / self.num_nodes_in_layers[1]), [self.num_nodes_in_layers[2], self.num_nodes_in_layers[1]])
        self.bias1 = np.zeros((self.num_nodes_in_layers[1], 1))
        self.bias2 = np.zeros((self.num_nodes_in_layers[2], 1))

    def predict(self, inputs, labels, weight_1, weight_2, bias_1, bias_2):
        W_layer_1 = weight_1.dot(inputs.T) + bias_1  
        layer_1 = functions.relu(W_layer_1)  

        W_layer_2 = weight_2.dot(layer_1) + bias_2
        Y_predict = functions.softmax(W_layer_2)
        Y_predict = Y_predict.T

        cross = -np.sum(labels * np.log(Y_predict)) / labels.shape[0]

        Y_predict = np.argmax(Y_predict, axis=1)
        labels = np.argmax(labels, axis=1)
        accuracy = (labels == Y_predict).mean()
        return accuracy, cross
  
    def train(self, inputs, labels, inputs_test, labels_test):
        time_start = datetime.now()
        print('\n Train:  \n\n')
        self.initial_weights_biases()
        random_state = np.random.get_state()
        np.random.shuffle(inputs)
        np.random.set_state(random_state)
        np.random.shuffle(labels)
        for epoch in range(self.num_epochs):

            for iteration in range(0, inputs.shape[0], self.batch_size):
               
                # batch input
                inputs_batch = inputs[iteration:iteration+self.batch_size]
                labels_batch = labels[iteration:iteration+self.batch_size]
              
                # forward pass
                inputs_batch = inputs_batch.T
                z1 = self.weight1.dot(inputs_batch) + self.bias1
                a1 = functions.relu(z1)
                z2 = self.weight2.dot(a1) + self.bias2
                y = functions.softmax(z2) 

                delta_2 = labels_batch.T - y
                weight2_gradient = delta_2.dot(a1.T) / self.batch_size
                bias2_gradient = np.sum(delta_2, axis=1, keepdims=True) / self.batch_size

                delta_1 = self.weight2.T.dot(delta_2) * functions.relu_deriv(z1)
                weight1_gradient = delta_1.dot(inputs_batch.T) / self.batch_size
                bias1_gradient = np.sum(delta_1, axis=1, keepdims=True) / self.batch_size

                self.weight1 = self.weight1 + self.learning_rate * weight1_gradient
                self.weight2 = self.weight2 + self.learning_rate * weight2_gradient
                self.bias1 = self.bias1 + self.learning_rate * bias1_gradient
                self.bias2 = self.bias2 + self.learning_rate * bias2_gradient
            print('Epoch---{}'.format(epoch))
            accuracy, crossentropy = self.predict(inputs, labels, self.weight1, self.weight2, self.bias1, self.bias2)
            print(" accuracy: ", str(accuracy), " loss: ", str(crossentropy))
        accuracy, crossentropy = self.predict(inputs, labels, self.weight1, self.weight2, self.bias1, self.bias2)
        print("Training: \n", " accuracy: ", str(accuracy), " loss: ", str(crossentropy))
        delta_time = datetime.now() - time_start
        print('Time:', delta_time)  
        test_accuracy, test_entropy = self.predict(inputs_test, labels_test, self.weight1, self.weight2, self.bias1, self.bias2)
        print("Test: \n", " accuracy: ", str(test_accuracy), " loss: ", str(test_entropy))