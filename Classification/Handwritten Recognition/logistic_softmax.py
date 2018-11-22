import numpy as np
import time


def logistic_softmax(minst_training, minst_validation, minst_testing, USPS_mat, USPS_target):
    """
    Implements the Softmax Logistic Regression
    :param minst_training: training dataset of MNIST
    :param minst_validation: validation dataset of MNIST
    :param minst_testing: testing dataset of MNIST
    :param USPS_mat: Feature dataset of USPS
    :param USPS_target: training dataset of MNIST
    """
    start =time.time()
    def design_T(train_y,validation_y,test_y):
        T_train=np.zeros((len(train_y),10))
        T_validation=np.zeros((len(validation_y),10))
        T_test=np.zeros((len(test_y),10))

        for i in range(len(train_y)):
            T_train[i][train_y[i]]=1
        for i in range(len(validation_y)):
            T_validation[i][validation_y[i]]=1
        for i in range(len(test_y)):
            T_test[i][test_y[i]]=1

        return T_train,T_validation,T_test


    train_x=minst_training[0]
    train_y=minst_training[1]

    validation_x=minst_validation[0]
    validation_y=minst_validation[1]

    test_x=minst_testing[0]
    test_y=minst_testing[1]

    W=np.ones((10,len(train_x[0])))

    Y=np.zeros(10)

    T_train, T_validation, T_test=design_T(train_y,validation_y,test_y)

    print ("Started the logistic regression")

    def fit():
        """
        Performs the model fitting for the classification
        """
        cnt=0
        for z in range(25):
            for i in range(len(train_x)):
                A=np.dot(W,train_x[i])+1
                denom = np.sum(np.exp(A))
                E=0
                for k in range(10):
                    Y[k]=np.exp(A[k])/denom
                    E-=np.dot(T_train[i][k],np.log(Y[k]))
                cnt+=1
                for j in range(10):
                    W[j]=W[j]-0.005*(Y[j]-T_train[i][j])*train_x[i]
            print ("Iteration {} Error value is: {} ".format(str(z+1), E))


    def train_and_find_accuracy(weight, train_x_local, design_T):
        """
        Find the weights and predict and calculate the accuracy of the predictions
        :param weight: Weight matrix
        :param train_x_local: dataset being used
        :param design_T: One hot vector representation of the matrix
        """
        count=0.0
        final_y = []
        for i in range(len(train_x_local)):
            A=np.dot(weight,train_x_local[i])
            denom = np.sum(np.exp(A))
            for k in range(10):
                Y[k]=np.exp(A[k])/denom
            final_y.append(np.argmax(Y))
            if np.where(Y==max(Y))[0][0]==np.where(design_T[i]==max(design_T[i]))[0][0]:
                count+=1
        print ("Accuracy of the fit is: {}".format(count/len(train_x_local)))
        return final_y

    fit()
    y_train = train_and_find_accuracy(W, train_x, T_train)
    y_valid = train_and_find_accuracy(W, validation_x, T_validation)
    y_test = train_and_find_accuracy(W, test_x, T_test)

    print ("Running time = "+str(time.time()-start))

    count = 0.0
    USPS_mat = np.array(USPS_mat)
    usps_y = np.zeros(10)
    final_y = []
    for i in range(len(USPS_mat)):
        A=np.dot(W,USPS_mat[i])
        denom = np.sum(np.exp(A))
        for k in range(10):
            usps_y[k]=np.exp(A[k])/denom
        final_y.append(np.argmax(usps_y))
        if np.where(usps_y==max(usps_y))[0][0]==USPS_target[i]:
            count+=1
    print ("USPS Accuracy")
    print (count/len(USPS_target))
    return y_train, y_valid, y_test, final_y
