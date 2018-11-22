import pickle
import gzip
from PIL import Image
import os
import numpy as np


def preprocess():
    filename = 'mnist.pkl.gz'
    f = gzip.open(filename, 'rb')
    minst_training_data, minst_validation_data, minst_test_data = pickle.load(f)#, encoding='latin1')
    f.close()
    
    USPSMat  = []
    USPSTar  = []
    curPath  = 'USPSdata/Numerals'
    
    for j in range(0,10):
        curFolderPath = curPath + '/' + str(j)
        imgs =  os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg,'r')
                img = img.resize((28, 28))
                imgdata = (255-np.array(img.getdata()))/255
                USPSMat.append(imgdata)
                USPSTar.append(j)
    
    return minst_training_data, minst_validation_data, minst_test_data, USPSMat, USPSTar
