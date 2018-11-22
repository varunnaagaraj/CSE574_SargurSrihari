# Random Forest Classifier

# Importing the libraries
import numpy as np


def generate_confusion_matrix(y_actual, y_pred):
    """
    Confusion Matrix generator
    :param y_actual: true values of target
    :param y_pred: predicted values of target
    """
    print("Python code to generate Confusion Matrix")
    cm =np.zeros((10,10), dtype=int)
    for i in range(y_pred.shape[0]):
        cm[y_actual[i]][y_pred[i]] += 1
    print(np.asmatrix(cm))

def random_forest_implementation(minst_training, minst_testing,USPS_mat, USPS_target):
    """
    Random Forest Classifier implementation
    :param minst_training: MNIST Feature set
    :param minst_testing: MNIST Testing set
    :param USPS_mat: USPS Feature Set
    :param USPS_target: USPS Target set
    """
    # Fitting classifier to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=150, criterion="entropy", random_state=0)
    classifier.fit(minst_training[0], minst_training[1])
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    
    #Test on MNIST dataset
    mnist_pred = classifier.predict(minst_testing[0])
    cm = confusion_matrix(minst_testing[1], mnist_pred)
    generate_confusion_matrix(minst_testing[1], mnist_pred)
    score = accuracy_score(minst_testing[1], mnist_pred)
    print("SKlearn method to generate Confusion Matrix")
    print(cm)
    print("MNIST Accuracy is: {}".format(score))

    
    # Testing with USPS test dataset
    print("USPS dataset Test")
    usps_pred = classifier.predict(USPS_mat)
    cm = confusion_matrix(USPS_target, usps_pred)
    generate_confusion_matrix(USPS_target, usps_pred)
    score = accuracy_score(USPS_target, usps_pred)
    print("SKlearn method to generate Confusion Matrix")
    print(cm)
    print("USPS Accuracy is: {}".format(score))
    return mnist_pred, usps_pred
