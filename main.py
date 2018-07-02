import numpy as np
from matplotlib import pyplot as plt
from mlp import MLP


def plot_stats(epoch_count, stats):
    plt.subplot(211)
    plt.title('Train Loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.plot(range(epoch_count), stats['loss'], 'b', label="loss")

    plt.subplot(212)
    plt.title('Test Accuracy')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.plot(range(epoch_count), stats['test_accuracy'], 'b', label="accuracy")
    plt.show()


def main():
    data = np.loadtxt('data/data_class_train.txt')
    data_x = data[:, 0:2]
    data_y = data[:, 2]
    n_samples = data.shape[0]

    n_train = int(n_samples * 0.7)
    perm = np.random.permutation(n_samples)
    train_indx = perm[:n_train]
    test_indx = perm[n_train:]

    data_x_train, data_y_train = data_x[train_indx, :], data_y[train_indx]
    data_x_test, data_y_test = data_x[test_indx, :], data_y[test_indx]


    data_y_train = np.array(data_y_train, dtype='int8').T
    data_y_test = np.array(data_y_test, dtype='int8').T
    data_x_train = data_x_train.T
    data_x_test = data_x_test.T

    number_of_iterations = 500
    nn = MLP(input_dim=2,
             first_hid_dim=40,
             second_hid_dim=50,
             output_dim=3,
             batch_size=32)
    stats = nn.train(data_x_train, data_y_train, data_x_test, data_y_test, number_of_iterations)

    def get_accuracy(predicted, true_value):
        return sum(true_value == predicted) / len(true_value)

    data_y_train_pred = nn.predict(data_x_train)
    print("Train acc: {}".format(get_accuracy(data_y_train_pred, data_y_train)))

    data_y_test_pred = nn.predict(data_x_test)
    print("Test acc: {}".format(get_accuracy(data_y_test_pred, data_y_test)))

    plot_stats(number_of_iterations, stats)


if __name__ == '__main__':
    main()