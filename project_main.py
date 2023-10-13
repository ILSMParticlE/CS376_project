import sys
import os
import time
import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from Dataset import trainDataset, transform, split_dataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
from kaggle import get_score
from train import train_model
from test import predict_test
from model import TLmodel
from utils import *

TRAIN_PATH = './imgs/train_all'
TEST_PATH = './imgs/test'
# RESULT_PATH = './submission.csv'
RESULT_PATH = './result/'
WEIGHT_PATH = './weight/'
RESULT_FLINE = 'model, pretained, optimizer, max_epoch, batchsize, initial_lr, public score, private score, statement\n'
LOG_FLINE = 'train_acc, train_loss, val_acc, val_loss\n'

NUM_CLASS = 10
LEARNING_RATE=0.001
NUM_EPOCH = 20
BATCH_SIZE = 128
cuda = torch.device('cuda')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(model_name='resnet34', 
        pretrained=False, 
        weight_load_flag=False, 
        train_flag=True, 
        train_path=TRAIN_PATH, 
        test_path=TEST_PATH, 
        num_epoch=NUM_EPOCH, 
        batch_size=BATCH_SIZE, 
        weight_fname='', 
        learning_rate=LEARNING_RATE,
        optimizer='adam',
        workers=10,
        statement='test'):
    
    print('Selected parameter:')
    print(f'{model_name}, {pretrained}, {optimizer}, {num_epoch}, {batch_size}, {learning_rate}, {statement}')
    ''' MKDIR '''
    if not os.path.isdir(WEIGHT_PATH): # weight folder
        os.makedirs(WEIGHT_PATH)
    if not os.path.isdir(RESULT_PATH): # result folder
        os.makedirs(RESULT_PATH)

    ''' Load Train data '''
    print('Load the training data')
    curr_time = get_current_time() # string of current date time
    train_data = trainDataset(dir_data = train_path, mode_test = False, crop_mode=1, affine=True,  color=True)
    test_data = trainDataset(dir_data = test_path, mode_test = True)
    train_sampler, valid_sampler = split_dataset(train_data) # split train & validation data
    train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=workers, pin_memory=True)
    valid_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=workers, pin_memory=True)

    ''' Load the CNN model '''
    print(f'Load the CNN model - (Use previous weight = {weight_load_flag})')
    model = TLmodel(model_name=model_name, 
                    pretrained=pretrained, 
                    learning_rate=learning_rate,
                    optimizer=optimizer)
    
    ''' load trained weight '''
    if weight_load_flag:
        if os.path.isfile(f'{WEIGHT_PATH}{weight_fname}'):
            model = TLmodel.load_from_checkpoint(f'{WEIGHT_PATH}{weight_fname}',
                                                model_name=model_name, 
                                                learning_rate=learning_rate,
                                                optimizer=optimizer)
            print('Previous weight is successfully loaded')
        else:
            print('there is no existing weight file')
    
    ''' Define the pytorch lightning trainer '''
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                        dirpath=WEIGHT_PATH, 
                                        filename=f'lightweight_{curr_time}', 
                                        mode='min') ## Set checkpoint
    trainer = Trainer(gpus=1, 
                    max_epochs=num_epoch, 
                    precision=16, # 16bit precision
                    accumulate_grad_batches=1,
                    profiler="simple", 
                    callbacks=[checkpoint_callback])

    if train_flag:
        ''' Training the model '''
        print('Start the training ')
        time_anchor = time.time()
        trainer.fit(model, train_dataloader, valid_dataloader) # traininig process
        
        ''' Collect training log '''
        train_acc_history = list(map(lambda x: x.item(), model.train_acc))
        train_loss_history = list(map(lambda x: x.item(), model.train_loss))
        val_acc_history = list(map(lambda x: x.item(), model.val_acc))[1:]
        val_loss_history = list(map(lambda x: x.item(), model.val_loss))[1:]
        model = TLmodel.load_from_checkpoint(checkpoint_callback.best_model_path,
                                            model_name=model_name)
        print(f'Training finished - time: {time.time()-time_anchor:.2f} s')

        ''' Draw the learning curve '''
        print('Draw the learning curve')
        fontsize = 15
        epochs = list(range(num_epoch))
        plt.plot(epochs, train_acc_history, 'r', label='train_acc_history')
        plt.plot(epochs, train_loss_history, 'g', label='train_loss_history')
        plt.plot(epochs, val_acc_history, 'b', label='val_acc_history')
        plt.plot(epochs, val_loss_history, 'k', label='val_loss_history')

        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('Accuracy & Loss', fontsize=fontsize)
        plt.legend(fontsize=fontsize)

        if not os.path.isdir(f'{RESULT_PATH}{curr_time}'): # result folder
            os.makedirs(f'{RESULT_PATH}{curr_time}')
        plt.savefig(f'{RESULT_PATH}{curr_time}/learning_curve_{curr_time}.png') # save learning curve

        ''' Prediction '''
        print('Start prediction')
        time_anchor = time.time()
        predict_test(model, test_dataloader) # Prediction
        public_score, private_score = get_score() # Get public & private score through Kaggle API
        print(f'Prediction finished - public: {public_score} / private: {private_score} / time: {time.time()-time_anchor:.2f} s')
        
        ''' Save result file '''
        with open(f'{RESULT_PATH}{curr_time}/result_{curr_time}.csv', 'w') as f: # Result table
            f.write(RESULT_FLINE)
            f.write(f'{model_name}, {pretrained}, {optimizer}, {num_epoch}, {batch_size}, {learning_rate}, {public_score}, {private_score}, {statement}')
            f.close()
        with open(f'{RESULT_PATH}{curr_time}/log_{curr_time}.csv', 'w') as f: # Training log table
            f.write(LOG_FLINE)
            for i in range(num_epoch):
                f.write(f'{train_acc_history[i]}, {train_loss_history[i]}, {val_acc_history[i]}, {val_loss_history[i]}\n')
            f.close()

    else: 
        ''' Prediction '''
        print('Start prediction')
        time_anchor = time.time()
        predict_test(model, test_dataloader) # Prediction
        public_score, private_score = get_score() # Get public & private score through Kaggle API
        print(f'Prediction finished - public: {public_score} / private: {private_score} / time: {time.time()-time_anchor:.2f} s')


if __name__ == '__main__':
    statement = 'resnet 34 with lr=0.01 and high epoch' # Do not include ','
    # model_name = 'vgg16'
    # model_name = 'resnet18'
    model_name = 'resnet34'
    weight = 'lightweight_2021-05-31_17_14_06.ckpt'
    workers=10

    main(model_name=model_name, 
            pretrained=False, # is pretrained imagenet model?
            weight_load_flag=False, # load trained weight
            weight_fname=weight, # file name of trained weight
            train_flag=True, # Access training step
            num_epoch=50, 
            batch_size=256,
            optimizer='adam', # Types of optimier - 'adam': Adam, 'sgd_pure': SGD without scheduler, 'sgd_schedule': SGD with scheduler
            learning_rate=0.01, # Initial learning rate
            workers=10, # The number of CPU process to load training data
            statement=statement # The specific statement show what the current experiment focus on
        )