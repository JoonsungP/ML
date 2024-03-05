import numpy as np


def normalize(data,mu,std) :
    data = (data-mu)/std

def denormalize(data,mu,std) :
    data = data*std + mu
