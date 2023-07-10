'''
This code is for pixel-average slicing method mentioned in the paper

Author: Ninghe Liu (lnh20@mails.tsinghua.edu.cn)

'''
####################################################################
import numpy as np
from PIL import Image
import cv2
import time
import scipy.io as scio

def layer(input_matrix,maxlayer):
    num = np.size(input_matrix)
    num_per_layer = (num) / maxlayer
    f = input_matrix.flatten()
    c = np.argsort(f)
    for i in range(num):
        f[c[i]] = i
    sort = f.reshape(input_matrix.shape)
    output_matrix = np.floor(sort/num_per_layer)
    return output_matrix

max_layer = 50

pname = 'example/testset.jpg'
dname = 'example/testset_depth.npy'
depth = np.load(dname)


pic = cv2.imread(pname, 0)

time_start=time.time()
max_depth = np.max(depth)
min_depth = np.min(depth)
#print(max_depth)
#print(min_depth)

seg_layer = layer(depth, max_layer)

avr_depth_list = []
depth_list = []

for i in range(max_layer):
    mask = (seg_layer == i)
    segment = np.zeros_like(pic)
    segment[mask] = pic[mask]
    avr_depth = np.sum(depth[mask])/np.sum((mask==True))
    rel_depth = (avr_depth-min_depth)/(max_depth-min_depth)
    avr_depth_list.append(avr_depth)
    depth_list.append(rel_depth)
    arr = np.array(segment, dtype='uint8')
    arr = Image.fromarray(arr)
    #arr.save('segment/test/test_'+str(i+1)+'.bmp', 'bmp')
time_end=time.time()
time_sum=time_end-time_start
print(time_sum)
scio.savemat('segment/rel_depth.mat', mdict={'rel_depth':depth_list})







