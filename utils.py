import numpy as np


def normalize(data,mu,std) :
    N, Nvar, Nlat, Nlon = data.shape
    for i in range(Nvar):
        data[:,i,:,:] = (data[:,i,:,:]-mu[i])/std[i]

def denormalize(data,mu,std) :
    data = data*std + mu
