import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

file1 = 'cifar-10-batches-py/data_batch_1'
file2 = 'cifar-10-batches-py/data_batch_2'
file3 = 'cifar-10-batches-py/data_batch_3'
file4 = 'cifar-10-batches-py/data_batch_4'
file5 = 'cifar-10-batches-py/data_batch_5'

all_1 = unpickle(file1)
labels1 = all_1[b'labels']
data1 = all_1[b'data']
data1 = [x for x in data1]

all_2 = unpickle(file2)
labels2 = all_2[b'labels']
data2 = all_2[b'data']
data2 = [x for x in data2]

all_3 = unpickle(file3)
labels3 = all_3[b'labels']
data3 = all_3[b'data']
data3 = [x for x in data3]

all_4 = unpickle(file4)
labels4 = all_4[b'labels']
data4 = all_4[b'data']
data4 = [x for x in data4]

all_5 = unpickle(file5)
labels5 = all_5[b'labels']
data5 = all_5[b'data']
data5 = [x for x in data5]

data = data1 + data2 + data3 + data4 + data5
labels = labels1 + labels2 + labels3 + labels4 + labels5

columns = ['Data', 'Labels']
df = pd.DataFrame(columns = columns)
df['Labels'] = labels
df['Data'] = data

# separate the 10 classes (0-9):
class0 = df.where(df['Labels'] == 0).dropna()
class1 = df.where(df['Labels'] == 1).dropna()
class2 = df.where(df['Labels'] == 2).dropna()
class3 = df.where(df['Labels'] == 3).dropna()
class4 = df.where(df['Labels'] == 4).dropna()
class5 = df.where(df['Labels'] == 5).dropna()
class6 = df.where(df['Labels'] == 6).dropna()
class7 = df.where(df['Labels'] == 7).dropna()
class8 = df.where(df['Labels'] == 8).dropna()
class9 = df.where(df['Labels'] == 9).dropna()


class0_im = [reshape_im(x) for x in list(class0['Data'])]
class1_im = [reshape_im(x) for x in list(class1['Data'])]
class2_im = [reshape_im(x) for x in list(class2['Data'])]
class3_im = [reshape_im(x) for x in list(class3['Data'])]
class4_im = [reshape_im(x) for x in list(class4['Data'])]
class5_im = [reshape_im(x) for x in list(class5['Data'])]
class6_im = [reshape_im(x) for x in list(class6['Data'])]
class7_im = [reshape_im(x) for x in list(class7['Data'])]
class8_im = [reshape_im(x) for x in list(class8['Data'])]
class9_im = [reshape_im(x) for x in list(class9['Data'])]

