'''
"This source code is for DGE-CNN inference to recover depth map from single RGB 2D input"
"The output depth map is boundary-enhanced and will be used for layer-based angular spectrum calculation of 3D hologram "

Reference:
N. Liu, Z. Huang, Z. He and L. Cao, "DGE-CNN: 2D-to-3D holographic display based on depth gradient
extracting module and CNN network"

load trained parameters before running this code or use our provided trained network

Author: Ninghe Liu (lnh20@mails.tsinghua.edu.cn)
'''

import os
import torch
import numpy as np
from PIL import Image
from fcrn import FCRN
from torch.autograd import Variable
import matplotlib.pyplot as plot
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

data_path = './example/testset.jpg'
dtype = torch.cuda.FloatTensor
data = np.array(Image.open(data_path))
row = data.shape[0]
col = data.shape[1]
img = Image.fromarray(data)

input_transform = transforms.Compose([transforms.Resize([228,304]),
                                              transforms.ToTensor()])
input = input_transform(img)
input = input.unsqueeze(0)

batch_size = 1
model = FCRN(batch_size)
model = model.cuda()
resume_from_file = True
resume_file = './model/model_0.2gd.pth'
if resume_from_file:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_file))

time_start=time.time()
model.eval()
with torch.no_grad():
    input_var = Variable(input.type(dtype))

    output = model(input_var)

    output = F.interpolate(output, size=[228,304], mode='bilinear')

    output = F.interpolate(output, size=[row, col], mode='bilinear')
    pred_depth_image = output.data.squeeze().cpu().numpy().astype(np.float32)

    print('predict complete.')
    plot.imsave('./example/test_depth.png', pred_depth_image, cmap="viridis")

    #np.save("./example/test_depth.npy",pred_depth_image)


