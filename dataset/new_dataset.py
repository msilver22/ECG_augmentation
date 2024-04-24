import matplotlib.pyplot as plt
import numpy as np
from record import create_record 
import pandas as pd
import hyperparameters as hp
import torch
from torch import Tensor
from tqdm import tqdm
import os

tensorECG_path = '/tensorECG.pt'
labels_path = '/labels.pt'
trainECG_path = '/trainECG.pt'
trainlabels_path = '/trainlabels.pt'
testECG_path = '/testECG.pt'
testlabels_path = '/testlabels.pt'

def create_ECG_tensor(n_record):
    #ECG pre-process
    record_name = f'record_{n_record:03}'
    target = f'/Users/silver22/Documents/AI trends/data/iridia-af-records/record_{n_record:03}/record_{n_record:03}_ecg_00.h5'
    metadata_df = pd.read_csv(hp.METADATA_PATH)
    record = create_record(record_name, metadata_df, hp.RECORDS_PATH)
    record.load_ecg()
    ecg = record.ecg[0][:,0]
    #Find index splitting
    index_path = '/Users/silver22/Documents/AI trends/data/datasets/dataset_detection_ecg_8192.csv'
    df = pd.read_csv(index_path)
    list_file = np.array(df['file'])
    list_indeces_start = np.array(df['start_index'])
    list_indeces_end = np.array(df['end_index'])
    list_labels = np.array(df['label'])
    start = np.where(list_file == target)
    start_ind = start[0][0]
    y = start[0][-1]
    shift = 4217
    list_ind_start = list_indeces_start[start_ind:start_ind+ shift]
    list_ind_end = list_indeces_end[start_ind:start_ind+ shift]
    #New dataset
    T = torch.zeros(len(list_ind_start),8192)
    labels = Tensor(list_labels[start_ind:start_ind+ shift])
    for i in range(len(list_ind_start)):
        #print(i)
        T[i,:] = Tensor(ecg[list_ind_start[i]:list_ind_end[i]])
    return T,labels

def create_ECG_tensor_database():

    #ECG pre-process
    record_names = [f'record_{n_record:03}' for n_record in range(0,10)]
    T = torch.zeros((10,4217,8192))
    labels = torch.zeros((10,4217))
    for j in tqdm(range(10)):
        target = f'/Users/silver22/Documents/AI trends/data/iridia-af-records/record_{j:03}/record_{j:03}_ecg_00.h5'
        metadata_df = pd.read_csv(hp.METADATA_PATH)
        record = create_record(record_names[j], metadata_df, hp.RECORDS_PATH)
        record.load_ecg()
        ecg = record.ecg[0][:,0]
        #Find index splitting
        index_path = '/Users/silver22/Documents/AI trends/data/datasets/dataset_detection_ecg_8192.csv'
        df = pd.read_csv(index_path)
        list_file = np.array(df['file'])
        list_indeces_start = np.array(df['start_index'])
        list_indeces_end = np.array(df['end_index'])
        list_labels = np.array(df['label'])
        start = np.where(list_file == target)
        start_ind = start[0][0]
        y = start[0][-1]
        shift = 4217
        list_ind_start = list_indeces_start[start_ind:start_ind+ shift]
        list_ind_end = list_indeces_end[start_ind:start_ind+ shift]
        labels[j,:] = Tensor(list_labels[start_ind:start_ind+ shift])
        for i in range(len(list_ind_start)):
            T[j,i,:] = Tensor(ecg[list_ind_start[i]:list_ind_end[i]])

    #Saving data
    torch.save(T, tensorECG_path)
    torch.save(labels, labels_path)

def train_test_dataset(value_split):

        T = torch.load(tensorECG_path)
        labels = torch.load(labels_path)
        keys = np.arange(0,10)
        length = len(keys)
        np.random.shuffle(keys)
        num_train = int(length * value_split)
        train_keys = keys[:num_train]
        test_keys = keys[num_train:]
        length_train = len(train_keys)
        length_test = len(test_keys)

        desc_train = "[LOG] Creating training set"
        trainECG = torch.zeros((num_train,4217,8192))
        trainlabels = torch.zeros((num_train,4217))
        for i in tqdm(range(length_train), desc=desc_train):
            trainECG[i,:,:] = T[train_keys[i],:,:]
            trainlabels[i,:] = labels[train_keys[i],:]

        desc_test = "[LOG] Creating test set"
        testECG = torch.zeros((length_test,4217,8192))
        testlabels = torch.zeros((length_test,4217))
        for j in tqdm(range(length_test), desc=desc_test):
            testECG[j,:,:] = T[test_keys[j],:,:]
            testlabels[j,:] = labels[test_keys[j],:]

        torch.save(trainECG,trainECG_path)
        torch.save(testECG,testECG_path)
        torch.save(trainlabels,trainlabels_path)
        torch.save(testlabels,testlabels_path)
        print("[LOG] Train and test set are completed.")

if __name__ == "__main__":
    create_ECG_tensor_database()
    train_test_dataset()
    
    
