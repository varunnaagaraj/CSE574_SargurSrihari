# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Variables - Hyper Parameters
lambda_value = 0.03
alpha = 0.01
M = 10
epochs = 600

print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')

# Importing the dataset
# Reading the target dataset T
y_data = pd.read_csv('Querylevelnorm_t.csv')
# Reading the design matrix X
x_data = pd.read_csv('Querylevelnorm_X.csv')

# Get rid of rows which are entirely zero on the X-axis
x_data = x_data.loc[:, (x_data != 0).any(axis=0)]
x_data = x_data.iloc[:, :].values

# y holds the values of the target from the target CSV
y = y_data.iloc[:, :].values

# Splitting the dataset into the Training set and Test set
# Here I am using the train_test_split function present in the SKlearn library.
# This function splits the data into training testing and validation datasets as shown below.
# we can specify the percentage of test size compared to the input matrix size.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.2, random_state=0)
x_train, val_train, y_train, val_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

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
    BigSigma = np.zeros((len(Data[0]), len(Data[0])))
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
            sub = np.subtract(Data[R], MuMatrix[C])
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


def root_mean_squared_error(data, output):
    """
    Calculates the RMS error value for closed form method
    :param data: Data matrix
    :param output: Actual Output matrix retrieved through regression
    :return: Accuracy and RMS value
    """
    sum = 0.0
    counter = 0
    for i in range(0, len(output)):
        sum = sum + np.math.pow((data[0][i] - output[i]), 2)
        if int(np.around(output[i], 0)) == data[0][i]:
            counter += 1
    accuracy = (float((counter * 100)) / float(len(output)))
    return str(accuracy) + ',' + str(np.math.sqrt(sum / len(output)))


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


# Generate the big sigma, training/testing/validation design matrices and the weights using the training design matrix
big_sigma = big_sigma_matrix(x_data)
training_phi = phi_matrix(x_train, Mu, big_sigma)
validation_phi = phi_matrix(val_train, Mu, big_sigma)
testing_phi = phi_matrix(x_test, Mu, big_sigma)
weights_cf = closed_form_weights(training_phi, lambda_value, y_train)

# Find the output got through w(transpose)* X
training_output = np.dot(weights_cf.transpose(), np.transpose(training_phi))
validation_output = np.dot(weights_cf.transpose(), np.transpose(validation_phi))
testing_output = np.dot(weights_cf.transpose(), np.transpose(testing_phi))

# Find the accuracy and the RMS error values
train_accuracy = root_mean_squared_error(y_train.transpose(), training_output.transpose())
val_accuracy = root_mean_squared_error(val_test.transpose(), validation_output.transpose())
test_accuracy = root_mean_squared_error(y_test.transpose(), testing_output.transpose())

print(train_accuracy, val_accuracy, test_accuracy)

# Stochastic Gradient Descent

print ('----------------------------------------------------')
print ("-------------Stochastic Gradient Descent------------")
print ('----------------------------------------------------')
# Choose a random initial weight
initial_weight = np.dot(220, weights_cf)
initial_weight = initial_weight.transpose()[0]

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
    validation_sgd_output_rms = root_mean_squared_error_SGD(np.dot(new_weight, np.transpose(validation_phi)),
                                                            validation_phi.transpose()[0])
    testing_error.append(float(testing_sgd_output_rms))

# Create a graph to output the error RMS values against the number of epochs
figure = plt.figure()
plt.plot(np.arange(epochs), testing_error)
plt.suptitle('Error RMS')
plt.xlabel('Number of iterations')
plt.ylabel('Testing RMS error rate')
plt.show()
