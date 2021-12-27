import cv2
import collections
from PIL import Image
import numpy as np
import pdb


anotaton21=np.array(Image.open('./result/annotations/training/21_manual1.png'))

stat=collections.Counter(anotaton21.flatten())

print(anotaton21.shape)

print(stat)

print('--------')

anotaton21_or=np.array(Image.open('./data/data_binarization/training/1st_manual/21_manual1.gif'))

#pdb.set_trace()

stat_or=collections.Counter(anotaton21_or.flatten())

print(anotaton21_or.shape)

print(stat_or)