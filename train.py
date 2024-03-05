# %%
import numpy as np
import netCDF4 as nc
import os
import xarray as xr
import pandas as pd
# %%
N_EPOCHS = 100 #total number of epochs
LR_REDUCE = 20 #after how many epochs to reduce the learning rate by 1/10
EPOCH_SIZE = 20 #days (x24 samples)  
BATCH_SIZE = 12 #samples per mini-batch
VALID_FREQ = 10 #use only every Nth day from the validation set (for speed)
CHIP_SHAPE = (35,35) #size of the samples used for training (hi-res size)
Ny, Nx = CHIP_SHAPE
CHEM = ["O3_SRF","NO2_SRF"]
nvar = len(CHEM)
n_stn = 439

# %%
root_path = '/home/pjs/_scratch/script/ML/down_pjs/'
train_dir = 'train'
valid_dir = 'valid/preproc'
out_dir = 'output'
train_path = root_path + train_dir
obs_path = root_path + valid_dir
true_path = obs_path+"/2016"
valid_path = obs_path+"/2018"

train_list = os.listdir(train_path)
# Manually indicating the year of observation
# Please modify this line when user change the path of the directories.
true_list = os.listdir(true_path)
valid_list = os.listdir(valid_path)

ntrain_files = len(train_list)


# Calculate mean and standard deviation
train_file_list = [train_path+"/"+d for d in train_list]
total_train_nc = nc.MFDataset(train_file_list,aggdim="Time")
MU = []
SIGMA = []
for i in range(nvar):
    chem_val_tmp = total_train_nc[CHEM[i]][:]
    MU.append(chem_val_tmp.mean())
    SIGMA.append(chem_val_tmp.std())

del(chem_val_tmp,total_train_nc)
# %%

# Define function that reads train and true data from wrfout and airkorea
def get_epoch():
    N = EPOCH_SIZE*24 # Size of time dimension for a sample of each epoch
    date_index = np.random.randint(0,ntrain_files-1,(EPOCH_SIZE,))
    train_files_tmp = [train_path+"/"+train_list[d] for d in date_index]
    true_files_tmp = [true_path+"/"+true_list[d] for d in date_index]

    train_val = np.zeros(shape=(N,nvar,Ny,Nx),dtype='float32')
    for i in range(EPOCH_SIZE):
        fopen = nc.Dataset(train_files_tmp[i])
        for j in range(nvar):
            train_val[i*24:(i+1)*24,j] = fopen[CHEM[j]][:]

    true_val = np.zeros(shape=(N,n_stn))
    for i in range(EPOCH_SIZE):
        df = pd.read_csv(true_files_tmp[i])
        true_val[i*24:(i+1)*24,:] = df.sort_values(['stn','date'],ascending=True)['O3'].to_numpy().reshape(439,24).T

    return train_val, true_val
            
#%%

# Traing Part
import torch
import torch.nn as nn
import concurrent.futures # Module for multi processing, but not used in this script
import gc
from neural_net import edrn_core
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from utils import normalize, denormalize

DEVICE = torch.device('cuda')
edrn = edrn_core(nvar,n_stn=n_stn).to(DEVICE)
LEARNING_RATE = 0.001
optimizer = optim.RMSprop(edrn.parameters(), lr=LEARNING_RATE, weight_decay=1e-8, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
loss_function = nn.CrossEntropyLoss()
print("define train iteration")

def train(model):
    save_dir = root_path + '/' + out_dir
    #threader = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    TL, VL, MAE = [],[],[]
    
    model.train()
    for e in range(1,N_EPOCHS+1):
        x_train, x_true = get_epoch()
        x_train = normalize(x_train)

        # Make dataset and dataloader in tensor format
        # This process enables NN to use mini-batch
        x_train_tensor = torch.from_numpy(x_train)   #.to(device=DEVICE,dtype=torch.float32)
        x_true_tensor = torch.from_numpy(x_true)     # .to(device=DEVICE,dtype=torch.float32)
        ds = TensorDataset(x_train_tensor,x_true_tensor)
        train_loader = DataLoader(ds,shuffle=True, batch_size=BATCH_SIZE)
        gc.collect()   # garbage memory collector
        epoch_loss = 0
        with tqdm(total=BATCH_SIZE,desc=f'Epoch {e}/{N_EPOCHS}') as pbar :
            for iter, batch in enumerate(train_loader):   # train mini-batches
                wrf_in = batch[0].to(device=DEVICE,dtype=torch.float32)
                obs_o3 = batch[1].to(device=DEVICE,dtype=torch.float32)
                pred = model(wrf_in)
                pred = denormalize(pred)
                loss = loss_function(pred,obs_o3)
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update() # tqdm update
                
                epoch_loss+=loss.item()
                pbar.set_postfix(**{'loss(batch)':loss.item()})
        TL.append(epoch_loss/BATCH_SIZE)

#%%

train(edrn)