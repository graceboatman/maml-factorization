#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def reshape_im(image):
    image = np.reshape(image, (3,32,32))
    image = np.swapaxes(image, 0, 1)
    image = np.swapaxes(image, 1, 2)
    return image

def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
file1 = 'meta'
file2 = 'train'
file3 = 'test'

all_1 = unpickle(file2)
labels1 = all_1[b'fine_labels']
data1 = all_1[b'data']
data1 = [x for x in data1]
ims1 = [reshape_im(x) for x in data1]

all_2 = unpickle(file3)
labels2 = all_2[b'fine_labels']
data2 = all_2[b'data']
data2 = [x for x in data2]
ims2 = [reshape_im(x) for x in data2]

columns = ['Data', 'Labels']
df_train = pd.DataFrame(columns = columns)
df_train['Labels'] = labels1
df_train['Data'] = ims1

df_test = pd.DataFrame(columns = columns)
df_test['Labels'] = labels2
df_test['Data'] = ims2

for i in range(0,100):
    createFolder('./data/cifar100/train/' + str(i) + '/')
    createFolder('./data/cifar100/test/' + str(i) + '/')
    createFolder('./data/cifar100/val/' + str(i) + '/')

for i in range(0,100):
    foo = df_train.where(df_train['Labels'] == i).dropna()
    foo_im = foo['Data']
    for j, image in enumerate(foo_im):
        im = Image.fromarray(image)
        im.save('data/cifar100/train/'+str(i)+'/'+str(j)+'.png')
for i in range(0,100):
    foo = df_test.where(df_train['Labels'] == i).dropna()
    foo_im = foo['Data']
    foo_im_val = foo_im[:50]
    foo_im_test = foo_im[50:]
    for j, image in enumerate(foo_im_val):
        im = Image.fromarray(image)
        im.save('data/cifar100/val/'+str(i)+'/'+str(j)+'.png')
    for j, image in enumerate(foo_im_test):
        im = Image.fromarray(image)
        im.save('data/cifar100/test/'+str(i)+'/'+str(j)+'.png')        

