import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def neural_net_implementation(dataset, minst_testing, USPS_mat, USPS_target):
    """
    Neural Network Implementation
    :param dataset: Dataset
    :param minst_testing: MNIST Test data
    :param USPS_mat: USPS Feature Matrix
    :param USPS_target: USPS Test Matrix
    """
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import EarlyStopping
    from keras.utils import to_categorical

    image_vector_size=28*28
    x_train = dataset[0].reshape(dataset[0].shape[0], image_vector_size)
    y_train = to_categorical(dataset[1], 10)
    model = Sequential()
    model.add(Dense(50, input_shape=(784,), activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    _ = model.fit(x_train, y_train, validation_split=0.2, callbacks=[monitor], verbose=0, epochs=50)

    print("MNIST")
    mnist_x_train = minst_testing[0].reshape(minst_testing[0].shape[0], image_vector_size)
    mnist_y_train = to_categorical(minst_testing[1], 10)
    y_pred_mnist = model.predict_classes(mnist_x_train)
    score, acc = model.evaluate(mnist_x_train, mnist_y_train)
    print("Score (RMSE) : {}".format(score))
    print("accuracy is :{}".format(acc))
    print("{}".format(confusion_matrix(minst_testing[1], y_pred_mnist)))
    
    print("USPS")
    USPS_mat1 = np.array(USPS_mat)
    usps_x = USPS_mat1.reshape(USPS_mat1.shape[0], image_vector_size)
    usps_y = to_categorical(USPS_target, 10)
    y_pred_usps = model.predict_classes(usps_x)
    score, acc = model.evaluate(np.array(usps_x), usps_y)
    print("Score (RMSE) : {}".format(score))
    print("accuracy is :{}".format(acc))
    print("{}".format(confusion_matrix(USPS_target, y_pred_usps)))

    mnist_history = model.fit(mnist_x_train, mnist_y_train, validation_split=0.2, callbacks=[monitor], verbose=0, epochs=5000)
    df = pd.DataFrame(mnist_history.history)
    df.plot(subplots=True, grid=True, figsize=(10, 15))
    plt.savefig("MNIST_NNClassifier")
    
    usps_history = model.fit(usps_x, usps_y, validation_split=0.2, callbacks=[monitor], verbose=0, epochs=5000)
    df = pd.DataFrame(usps_history.history)
    df.plot(subplots=True, grid=True, figsize=(10, 15))
    plt.savefig("USPS_NNClassifier")
    return y_pred_mnist, y_pred_usps


