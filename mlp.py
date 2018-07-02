import numpy as np
from sklearn.neural_network._base import softmax, relu, tanh
from activation_functions import AF
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss


class MLP:
    def __init__(self,
                 input_dim=2,
                 first_hid_dim=20,
                 second_hid_dim=15,
                 output_dim=3,
                 learning_rate=1e-3,
                 batch_size=630):

        # np.random.seed(42)
        self.w1 = np.random.randn(first_hid_dim, input_dim)
        self.w2 = np.random.randn(second_hid_dim, first_hid_dim)
        self.w3 = np.random.randn(output_dim, second_hid_dim)
        self.ws = np.eye(second_hid_dim, input_dim)

        self.b1 = np.random.randn(first_hid_dim, 1)
        self.b2 = np.random.randn(second_hid_dim, 1)
        self.b3 = np.random.randn(output_dim, 1)

        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def forward(self, x):
        self.x = x
        self.x_1_hidden = self.w1.dot(self.x) + self.b1
        self.x_1_hidden_act = tanh(self.x_1_hidden)

        self.x_2_hidden = self.w2.dot(self.x_1_hidden_act) + self.b2
        self.x_2_hidden_act = relu(self.x_2_hidden)

        self.x_2_hidden_act_with_skip_connnection = self.x_2_hidden_act + self.ws.dot(self.x)
        self.x3_hidden = self.w3.dot(self.x_2_hidden_act_with_skip_connnection) + self.b3
        self.y_pred = softmax(self.x3_hidden)

        return self.y_pred

    def back_propagation(self, y):
        delta3 = self.y_pred.copy()
        delta3[y, range(len(y))] -= 1
        delta3 /= len(y)
        # delta3 = (self.y_pred - y) # / self.batch_size
        self.w3 -= self.learning_rate * delta3.dot(self.x_2_hidden_act_with_skip_connnection.T)

        deltab3 = np.sum(delta3, axis=1, keepdims=True)
        self.b3 -= self.learning_rate * deltab3

        drelu = AF.drelu(self.x_2_hidden)
        delta2 = drelu * self.w3.T.dot(delta3)
        self.w2 -= self.learning_rate * delta2.dot(self.x_1_hidden_act.T)

        deltab2 = np.sum(delta2, axis=1, keepdims=True)
        self.b2 -= self.learning_rate * deltab2

        dtanh = AF.dtanh(self.x_1_hidden)
        delta1 = dtanh * self.w2.T.dot(delta2)
        self.w1 -= self.learning_rate * delta1.dot(self.x.T)

        deltab1 = np.sum(delta1, axis=1, keepdims=True)
        self.b1 -= self.learning_rate * deltab1

    def predict(self, x):
        self.forward(x)
        return np.argmax(self.y_pred, axis=0)

    def train(self, x, y, epoch_num=500):
        losses = []
        for e in range(epoch_num):
            for i in range(0, len(x), self.batch_size):
                begin = i * self.batch_size
                end = begin + self.batch_size
                self.forward(x[:, begin:end])
                # loss = AF.cross_entropy(self.y_pred, Y)
                loss = log_loss(y[begin:end], self.y_pred.T)
                losses.append(loss)
                self.back_propagation(y[begin:end])

            print("Epoch: {}, Loss: {}".format(e, losses[-1]))

        # visualize loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(epoch_num), losses, 'b', label="loss")

        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()
