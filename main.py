import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mlp import MLP, get_accuracy


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

    plt.subplots_adjust(hspace=0.8)
    plt.show()


def main(dataset_filename,
         input_dim,
         first_hid_dim,
         second_hid_dim,
         output_dim,
         learning_rate,
         batch_size,
         number_of_iterations):
    data_dir = 'data'
    df = pd.read_csv(os.path.join(data_dir, dataset_filename))
    x = df[['age', 'gender', 'education', 'country', 'ethnicity', 'nscore', 'escore', 'oscore',
            'ascore', 'cscore', 'impulsive', 'ss']]
    y = df['cannabis']
    x = np.array(x)
    y = np.array(y)
    n_samples = x.shape[0]
    n_train = int(n_samples * 0.7)

    perm = np.random.permutation(n_samples)
    train_indx = perm[:n_train]
    test_indx = perm[n_train:]

    data_x_train, data_y_train = x[train_indx, :], y[train_indx]
    data_x_test, data_y_test = x[test_indx, :], y[test_indx]

    data_x_train = data_x_train.T
    data_x_test = data_x_test.T

    nn = MLP(input_dim=input_dim,
             first_hid_dim=first_hid_dim,
             second_hid_dim=second_hid_dim,
             output_dim=output_dim,
             learning_rate=learning_rate,
             batch_size=batch_size)

    stats = nn.train(data_x_train, data_y_train, data_x_test, data_y_test, number_of_iterations)

    data_y_train_pred = nn.predict(data_x_train)
    print("Train acc: {}".format(get_accuracy(data_y_train_pred, data_y_train)))

    data_y_test_pred = nn.predict(data_x_test)
    print("Test acc: {}".format(get_accuracy(data_y_test_pred, data_y_test)))

    plot_stats(number_of_iterations, stats)


if __name__ == '__main__':
    main(dataset_filename='data_cannabis_2.csv',
         input_dim=12,
         first_hid_dim=25,
         second_hid_dim=15,
         output_dim=2,
         learning_rate=1e-3,
         batch_size=20,
         number_of_iterations=600)
