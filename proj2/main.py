import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Uncomment below code if you need to create a preprocessing of all datasets
# import os
# os.system('python preprocessing.py')

# Variables - Hyper Parameters
lambda_value = 0.03
alpha = 0.01
M = 10
epochs = 600

# Ignoring numpy runtime warnings due to future imports in python
np.seterr(divide='ignore', invalid='ignore', under='ignore')


def human_feature_subtraction():
    """
    Implements the human feature set subtraction.
    :return: feature set and target set
    """
    # Importing the dataset
    data = pd.read_csv('./HumanObserved-Dataset/HumanObserved-Features-Data/subtracted_dataset.csv')
    data = data.sample(frac=1).reset_index(drop=True)
    # Reading the design matrix X
    x_data = data.iloc[:, 3:12]
    y_data = data.iloc[:, -1]
    return x_data, y_data


def human_feature_concat():
    """
    Implements the human feature set concatenation.
    :return: feature set and target set
    """
    data = pd.read_csv('./HumanObserved-Dataset/HumanObserved-Features-Data/concatenated_dataset.csv')
    data = data.sample(frac=1).reset_index(drop=True)
    # Reading the design matrix X
    x_data = data.iloc[:, 3:21]
    y_data = data.iloc[:, -1]
    return x_data, y_data


def gsc_feature_subtraction():
    """
    Implements the GSC feature set subtraction.
    :return: feature set and target set
    """
    data = pd.read_csv("./GSC-Dataset/GSC-Features-Data/subtracted_dataset.csv")
    data = data.sample(frac=1).reset_index(drop=True)
    # Reading the design matrix X
    x_data = data.iloc[:, 3:515]
    y_data = data.iloc[:, -1]
    return x_data, y_data


def gsc_feature_concat():
    """
    Implements the GSC feature set concatenation.
    :return: feature set and target set
    """
    data = pd.read_csv("./GSC-Dataset/GSC-Features-Data/concatenated_dataset.csv")
    data = data.sample(frac=1).reset_index(drop=True)
    # Reading the design matrix X
    x_data = data.iloc[:, 3:1027]
    y_data = data.iloc[:, -1]
    return x_data, y_data


def linear_regression(x_data, y, tag=""):
    # Splitting the dataset into the Training set and Test set
    # Here I am using the train_test_split function present in the SKlearn library.
    # This function splits the data into training testing and validation datasets as shown below.
    # we can specify the percentage of test size compared to the input matrix size.

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.2, random_state=0)

    # The rows are randomly sorted into the groups to form M initial clusters. An exchange algorithm is applied to
    # this initial configuration which searches for the rows of data that would produce a maximum decrease in a
    # least-squares penalty function.
    kmeans = KMeans(n_clusters=M, random_state=0).fit(x_train)
    Mu = kmeans.cluster_centers_

    def big_sigma_matrix(Data):
        """
        Summation of all the variance values along the diagonal. This is also called the co-variance matter.
        :param Data: This will be the training data that we are using to generate the covariance matrix
        :return: Covariance matrix
        """
        BigSigma = np.zeros((Data.shape[1], Data.shape[1]))
        DataT = np.transpose(Data)
        for i in range(0, len(DataT)):
            BigSigma[i][i] = np.var(DataT[i])
        BigSigma = np.dot(200, BigSigma)
        return BigSigma

    def phi_matrix(Data, MuMatrix, BigSigma):
        """
        This will generate the design matrix phi
        :param Data: Training/Testing/Validation sets
        :param MuMatrix: Matrix with centroid values for the clusters
        :param BigSigma: Covariance matrix
        :return: Design matrix Phi
        """
        phi = np.zeros((int(len(Data)), len(MuMatrix)))
        BigSigInv = np.linalg.inv(BigSigma)
        for C in range(0, len(MuMatrix)):
            for R in range(0, int(len(Data))):
                sub = np.subtract(Data.iloc[R], MuMatrix[C])
                sub_1 = np.dot(BigSigInv, np.transpose(sub))
                phi[R][C] = np.math.exp(-0.5 * np.dot(sub, sub_1))
        return phi

    def closed_form_weights(phi_matrix, lambda_value, targets):
        """
        Calculate the weights (W/theta Matrix)
        :param phi_matrix: Design matrix Phi
        :param lambda_value: term that governs the relative importance of the regularization term
        :param targets: Target vector
        :return: Weight matrix W/theta
        """
        lamba_identity = np.identity(M)
        for i in range(M):
            lamba_identity[i][i] = lambda_value
        weight = np.linalg.inv(np.add(lamba_identity, np.dot(phi_matrix.transpose(), phi_matrix))).dot(
            phi_matrix.transpose().dot(targets))
        return weight

    def root_mean_squared_error_SGD(data, output):
        """
        Method to find the RMS value for SGD method
        :param data: Data matrix
        :param output: Actual Output matrix retrieved through regression
        :return: RMS value
        """
        sum = 0.0
        for i in range(0, len(output)):
            sum = sum + np.math.pow((data[i] - output[i]), 2)
        return str(np.math.sqrt(sum / len(output)))

    # Generate the big sigma, training/testing/validation design matrices and the weights using the training design
    # matrix
    big_sigma = big_sigma_matrix(x_data)
    training_phi = phi_matrix(x_train, Mu, big_sigma)
    testing_phi = phi_matrix(x_test, Mu, big_sigma)
    weights_cf = closed_form_weights(training_phi, lambda_value, y_train)

    # Choose a random initial weight
    initial_weight = np.dot(220, weights_cf)
    initial_weight = initial_weight.transpose()

    # Iterate over 500 times to find the minima using the gradient descent approach
    # To find a local minimum of a function using gradient descent, we take steps proportional to the negative of the
    # gradient of the function at the current point.
    testing_error = []
    for i in range(0, epochs):
        delta_error = np.add(-np.dot(training_phi[i] - np.dot(initial_weight, training_phi[i]), (training_phi[i])),
                             np.dot(2, initial_weight))
        new_weight = -np.dot(alpha, delta_error) + initial_weight
        initial_weight = new_weight

        # Find the RMS value at each iteration to check the learning of the model. We try to approach the minima which is
        # ideally 0. In our case with 500 iterations I was able to reach 0.05-0.06. With higher number of iterations,
        # we can decrease the RMS value further.
        training_sgd_output_rms = root_mean_squared_error_SGD(np.dot(new_weight, np.transpose(training_phi)),
                                                              training_phi.transpose()[0])
        testing_sgd_output_rms = root_mean_squared_error_SGD(np.dot(new_weight, np.transpose(testing_phi)),
                                                             testing_phi.transpose()[0])
        testing_error.append(float(testing_sgd_output_rms))
    print(testing_error[len(testing_error) - 1])
    figure = plt.figure()
    plt.plot(np.arange(epochs), testing_error)
    plt.suptitle('Error RMS')
    plt.xlabel('Number of iterations')
    plt.ylabel('Testing RMS error rate')
    figure.savefig(tag)


def logistic_regression(x, y, epochs, learning_rate, tag=""):
    """
    This method implements the logic for logistic regression
    :param x: Feature matrix
    :param y: Target matrix
    :param epochs: number of times to run GD
    :param learning_rate: alpha value for learning
    :param tag: string for image naming
    """

    def loss(h, y):
        """
        Implements the log likelihood function
        :param h: predicted hypothesis
        :param y: actual target values
        :return: log likelihood
        """
        temp = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        return temp

    theta = np.zeros(x.shape[1])
    L_Erms_Train = []
    from sklearn.metrics import mean_squared_error
    import math
    tlen = int(math.ceil(len(y) * 0.8))
    x_train = x.iloc[0:tlen, :]
    y_train = y.iloc[0:tlen]
    x_test = x.iloc[tlen:, :]
    y_test = y.iloc[tlen:]
    for i in range(0, epochs):
        z = np.dot(x_train, theta)
        h = 1 / (1 + np.exp(-z))
        temp = h - list(y_train)
        gradient = np.dot(x_train.T, temp) / len(y_train)
        theta = theta - learning_rate * gradient
        L_Erms_Train.append(loss(h, np.asarray(y_train)))
    pred = np.around(1 / (1 + np.exp(np.dot(x_test, theta))))
    accuracy = (pred == y_test).mean()
    print("accuracy = {}".format(accuracy))
    mse = mean_squared_error(y_test, pred)
    print("RMSE:{}".format(mse))
    print("Log likelihood function = " + str(np.around(min(L_Erms_Train), 5)))
    figure = plt.figure()
    plt.plot(np.arange(epochs), L_Erms_Train)
    plt.suptitle('Error RMS')
    plt.xlabel('Number of iterations')
    plt.ylabel('Testing RMS error rate')
    figure.savefig(tag)


def neural_net_implementation(x, y, tag=""):
    """
    Neural Network implementation for the GSC and HOF features
    :param x: Input Feature matrix
    :param y: Target matrix
    :param tag: tag for saving file
    """
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import EarlyStopping
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = Sequential()
    model.add(Dense(50, input_dim=x.shape[1], activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    history = model.fit(x, y, validation_data=(x_test, y_test), callbacks=[monitor], verbose=0, epochs=5000)

    score, acc = model.evaluate(x_test, y_test)
    print("Score (RMSE) : {}".format(score))
    print("accuracy is :{}".format(acc))

    df = pd.DataFrame(history.history)
    df.plot(subplots=True, grid=True, figsize=(10, 15))
    plt.savefig(tag)


if __name__ == '__main__':
    """
    Driver Code for the program.
    """
    print("Human Feature Subtraction\n")
    x, y = human_feature_subtraction()
    linear_regression(x, y, "lin_reg_HOF_sub.png")
    logistic_regression(x, y, 10000, 0.15, "log_reg_HOF_sub.png")
    neural_net_implementation(x, y, "NN_HOF_sub.png")

    print("Human Feature Concatenation\n")
    x, y = human_feature_concat()
    linear_regression(x, y, "lin_reg_HOF_concat.png")
    logistic_regression(x, y, 10000, 0.15, "log_reg_HOF_concat.png")
    neural_net_implementation(x, y, "NN_HOF_concat.png")

    print("GSC Feature Subtraction\n")
    x, y = gsc_feature_subtraction()
    linear_regression(x, y, "lin_reg_GSC_sub.png")
    logistic_regression(x, y, 10000, 0.15, "log_reg_GSC_sub.png")
    neural_net_implementation(x, y, "NN_GSC_sub.png")

    print("GSC Feature Concatenation\n")
    x, y = gsc_feature_concat()
    linear_regression(x, y, "lin_reg_GSC_concat.png")
    logistic_regression(x, y, 10000, 0.15, "log_reg_GSC_concat.png")
    neural_net_implementation(x, y, "NN_GSC_concat.png")
