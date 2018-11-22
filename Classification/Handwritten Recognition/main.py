# Created by Varun at 18/11/18

import logistic_softmax
import svm
import random_forest
import neural_network_classifier
import preprocess
import numpy as np
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, accuracy_score

print("--------------------------")
print("\tUBID:50290761\n\tName: Varun Nagaraj")
print("--------------------------")

print("Starting Preprocess")
mnist_training, mnist_validation, mnist_testing, USPS_mat, USPS_target = preprocess.preprocess()
print("Done with Preprocess")

print("Starting Logistic")
y_train_soft, y_valid_soft, y_test_soft, usps_y_soft = logistic_softmax.logistic_softmax(mnist_training,
                                                                                         mnist_validation,
                                                                                         mnist_testing, USPS_mat,
                                                                                         USPS_target)
softmax_score_mnist = accuracy_score(mnist_testing[1], y_test_soft)
cm = confusion_matrix(mnist_testing[1], y_test_soft)
print("Accuracy for MNIST Softmax: {}".format(softmax_score_mnist))
print("Confusion Matrix Softmax: \n{}".format(cm))

softmax_score_usps = accuracy_score(USPS_target, usps_y_soft)
cm = confusion_matrix(USPS_target, usps_y_soft)
print("Accuracy for MNIST Softmax: {}".format(softmax_score_usps))
print("Confusion Matrix Softmax: \n{}".format(cm))

print("Done with Logistic")

print("Starting SVM")
y_mnist_svm, y_usps_svm = svm.support_vector_machine(mnist_training, mnist_testing, USPS_mat, USPS_target)

# ##Uncomment to run SVM with Gamma = 1
# print("\n\tSVM with Gamma value set to 1")
# y_mnist_svm_1, y_usps_svm_1 = svm.support_vector_machine(mnist_training, mnist_testing, USPS_mat, USPS_target, gamma=1)
print("Done with SVM")

print("Starting Random Forest")
y_mnist_rf, y_usps_rf = random_forest.random_forest_implementation(mnist_training, mnist_testing, USPS_mat, USPS_target)
print("Done with Random Forest")

print("Starting Neural Network Classifier")
y_pred_mnist_nn, y_pred_usps_nn = neural_network_classifier.neural_net_implementation(mnist_training, mnist_testing,
                                                                                      USPS_mat, USPS_target)
print("Done with Neural Network Classifier")

final_pred_mnist = np.array([])
for i in range(0, len(mnist_testing[1])):
    final_pred_mnist = np.append(final_pred_mnist, mode(
        [y_test_soft[i], y_mnist_svm[i], y_mnist_rf[i], y_pred_mnist_nn[i]])[0][0])


cm = confusion_matrix(mnist_testing[1], final_pred_mnist)
score = accuracy_score(mnist_testing[1], final_pred_mnist)
print("CM for MNIST Final majority vote\n{}".format(cm))
print("Score for MNIST Final majority vote:{}".format(score))

final_pred_mnist = np.array([])
for i in range(0, len(USPS_target)):
    final_pred_mnist = np.append(final_pred_mnist, mode(
        [usps_y_soft[i], y_usps_svm[i], y_usps_rf[i], y_pred_usps_nn[i]])[0][0])

cm = confusion_matrix(USPS_target, final_pred_mnist)
score = accuracy_score(USPS_target, final_pred_mnist)
print("CM for USPS Final majority vote\n{}".format(cm))
print("Score for USPS Final majority vote:{}".format(score))