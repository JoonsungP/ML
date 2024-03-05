import numpy as np


def normalize(data,mu,std) :
    data_shape = data.shape
    for i in range(data_shape[1]):
        data[:,i] = (data[:,i]-mu[i])/std[i]

def denormalize(data,mu,std) :
    data = data*std + mu
