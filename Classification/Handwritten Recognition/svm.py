import numpy as np
import time

start =time.time()


def generate_confusion_matrix(y_actual, y_pred):
    """
    Generates the confusion matrix
    :param y_actual: True values of the target
    :param y_pred: predicted values of the target
    """
    print("Python code to generate Confusion Matrix")
    cm =np.zeros((10,10), dtype=int)
    for i in range(y_pred.shape[0]):
        cm[y_actual[i]][y_pred[i]] += 1
    print(np.asmatrix(cm))


def support_vector_machine(dataset, mnist_test, usps_data, usps_test, gamma="auto_deprecated"):
    """
    Implements the Support Vector Machine Classification algorithm
    :param dataset: Dataset
    :param mnist_test: MNIST Test dataset
    :param usps_data: USPS Feature Set
    :param usps_test: USPS Target
    :param gamma: Gamma value
    """
    # Fitting classifier to the Training set
    from sklearn.svm import SVC
    if gamma == 1.0:
        classifier = SVC(kernel='rbf', gamma=gamma, random_state=0, max_iter=2000, C=1.75)
    else:
        classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(dataset[0], dataset[1])
    
    from sklearn.metrics import confusion_matrix, accuracy_score
    # Testing with MNIST test dataset 
    print("MNIST dataset Test")
    mnist_pred = classifier.predict(mnist_test[0])
    cm = confusion_matrix(mnist_test[1], mnist_pred)
    generate_confusion_matrix(mnist_test[1], mnist_pred)
    score = accuracy_score(mnist_test[1], mnist_pred)
    print("SKlearn method to generate Confusion Matrix")
    print(cm)
    print("MNIST Accuracy is: {}".format(score))
    
    # Testing with USPS test dataset
    print("USPS dataset Test")
    usps_pred = classifier.predict(usps_data)
    cm = confusion_matrix(usps_test, usps_pred)
    generate_confusion_matrix(usps_test, usps_pred)
    score = accuracy_score(usps_test, usps_pred)
    print("SKlearn method to generate Confusion Matrix")
    print(cm)
    print("USPS Accuracy is: {}".format(score))
    print("Running time = "+str(time.time()-start))
    return mnist_pred, usps_pred
