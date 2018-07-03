import numpy as np
from sklearn.neural_network._base import softmax, relu, tanh
from activation_functions import AF
from sklearn.metrics import log_loss


class MLP:
    """
    Feed-forward multilayer perceptron with two hidden layers and
    ResNet-like shortcut connection.

    y^ = softmax((ReLU(tanh(w1*x + b1)*w2 + b2) + ws*x)*w3 + b3)
    """
    def __init__(self,
                 input_dim,
                 first_hid_dim,
                 second_hid_dim,
                 output_dim,
                 learning_rate,
                 batch_size):

        self.xavier_init(input_dim, first_hid_dim, second_hid_dim, output_dim)

        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def xavier_init(self, input_dim, first_hid_dim, second_hid_dim, output_dim):
        stdv_input = 2. / (input_dim + first_hid_dim)
        stdv_first = 2. / (first_hid_dim + second_hid_dim)
        stdv_second = 2. / (second_hid_dim + output_dim)

        self.w1 = np.random.uniform(-stdv_input, stdv_input, (first_hid_dim, input_dim))
        self.w2 = np.random.uniform(-stdv_first, stdv_second, (second_hid_dim, first_hid_dim))
        self.w3 = np.random.uniform(-stdv_second, stdv_second, (output_dim, second_hid_dim))
        self.ws = np.eye(second_hid_dim, input_dim)

        self.b1 = np.random.uniform(-stdv_input, stdv_input, (first_hid_dim, 1))
        self.b2 = np.random.uniform(-stdv_first, stdv_first, (second_hid_dim, 1))
        self.b3 = np.random.uniform(-stdv_second, stdv_second, (output_dim, 1))

    def forward(self, x):
        """
        Forward pass. Tanh->ReLU->Softmax
        :param x: input data (n_features, n_samples)
        :return: predicted labels (n_classes, n_samples)
        """
        self.x = x
        # first layer. x_1_hidden_act.shape = (first_hid_dim, batch_size)
        self.x_1_hidden = self.w1.dot(self.x) + self.b1
        self.x_1_hidden_act = tanh(self.x_1_hidden)

        # second layer. x_2_hidden_act.shape = (second_hid_dim, batch_size)
        self.x_2_hidden = self.w2.dot(self.x_1_hidden_act) + self.b2
        self.x_2_hidden_act = relu(self.x_2_hidden)

        # third(output) layer. y_pred.shape = (output_dim, batch_size)
        self.x_2_hidden_act_with_skip_connnection = self.x_2_hidden_act + self.ws.dot(self.x)
        self.x3_hidden = self.w3.dot(self.x_2_hidden_act_with_skip_connnection) + self.b3
        self.y_pred = softmax(self.x3_hidden)

        return self.y_pred

    def back_propagation(self, y):
        """
        Backprop. Get deltas for each layer and update weights.
        :param y: ground truth labels
        """
        # delta3.shape = (output_dim, batch_size)
        # delta_b3.shape = (output_dim, 1)
        delta3 = self.y_pred.copy()
        delta3[y, range(len(y))] -= 1
        delta3 /= len(y)
        delta_b3 = np.sum(delta3, axis=1, keepdims=True)

        # delta2.shape = (second_hid_dim, batch_size)
        # delta_b2.shape = (second_hid_dim, 1)
        drelu = AF.drelu(self.x_2_hidden)
        delta2 = drelu * self.w3.T.dot(delta3)
        delta_b2 = np.sum(delta2, axis=1, keepdims=True)

        # delta1.shape = (first_hid_dim, batch_size)
        # delta1.shape = (first_hid_dim, 1)
        dtanh = AF.dtanh(self.x_1_hidden)
        delta1 = dtanh * self.w2.T.dot(delta2)
        delta_b1 = np.sum(delta1, axis=1, keepdims=True)

        # update weights and biases
        self.w3 -= self.learning_rate * delta3.dot(self.x_2_hidden_act_with_skip_connnection.T)
        self.w2 -= self.learning_rate * delta2.dot(self.x_1_hidden_act.T)
        self.w1 -= self.learning_rate * delta1.dot(self.x.T)

        self.b3 -= self.learning_rate * delta_b3
        self.b2 -= self.learning_rate * delta_b2
        self.b1 -= self.learning_rate * delta_b1

    def predict(self, x):
        """
        Predict labels for given x
        :param x: input data (n_features, n_samples)
        :return predicted labels (n_samples,)
        """
        self.forward(x)
        return np.argmax(self.y_pred, axis=0)

    def train(self, x_train, y_train, x_test, y_test, epoch_num=500):
        """
        Train network using x, y for epoch_num iterations
        :param x_train: train x data (n_features, n_samples)
        :param y_train: train y data (n_samples,)
        :param x_test: test x data (n_features, n_samples)
        :param y_test: test y data (n_samples,)
        :param epoch_num: number of epochs
        :return: dict of train statistics
        """
        train_stats = {'loss': [], 'test_accuracy': []}
        for e in range(epoch_num):
            # for each batch in train data
            for i in range(0, len(x_train), self.batch_size):
                begin = i * self.batch_size
                end = begin + self.batch_size

                self.forward(x_train[:, begin:end])
                # loss = AF.cross_entropy(self.y_pred, Y)
                loss = log_loss(y_train[begin:end], self.y_pred.T)
                # train_stats['loss'].append(loss)
                self.back_propagation(y_train[begin:end])

            train_stats['loss'].append(loss)
            train_stats['test_accuracy'].append(get_accuracy(self.predict(x_test), y_test))
            if not e % (epoch_num // 10):
                print("Epoch: {}, Loss: {}".format(e, train_stats['loss'][-1]))
        return train_stats

def get_accuracy(predicted, true_value):
    return sum(true_value == predicted) / len(true_value)
