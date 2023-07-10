'''
"This source code is for training DGE-CNN network to recover depth map from single RGB 2D input"
"The output depth map is boundary-enhanced and will be used for layer-based angular spectrum calculation of 3D hologram "

Reference:
N. Liu, Z. Huang, Z. He and L. Cao, "DGE-CNN: 2D-to-3D holographic display based on depth gradient
extracting module and CNN network"

DGE module is packed in loss_gd, to generate boundary-enhanced depth maps, train the network with loss_gd

Author: Ninghe Liu (lnh20@mails.tsinghua.edu.cn)
'''
###################################################################
import matplotlib.pyplot as plt
import torch.nn
from loader import NyuDepthLoader
import os
from fcrn import FCRN
from torch.autograd import Variable
from weights import load_weights
from utils import load_split, loss_mse, loss_gd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot
import numpy as np

dtype = torch.cuda.FloatTensor

#pretrained weight file
weights_file = "./model/pretrained.npy"

def main():
    batch_size = 16
    data_path = './data/nyu_depth_v2_labeled.mat'
    learning_rate = 1.0e-5
    monentum = 0.9
    weight_decay = 0.0001
    num_epochs = 20

    epochlist = []
    losslist1 = []
    losslist2 = []

    # 1.Load data
    train_lists, val_lists, test_lists = load_split()
    print("Loading data...")
    train_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, train_lists),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, val_lists),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, test_lists),
                                             batch_size=batch_size, shuffle=True, drop_last=True)

    # 2.Load model
    print("Loading model...")
    model = FCRN(batch_size)
    model.load_state_dict(load_weights(model, weights_file, dtype)) #load pretrained model
    #loading pretrained file
    resume_from_file = False
    resume_file = './model/checkpoint.pth'
    start_epoch = 0
    if resume_from_file:
        if os.path.isfile(resume_file):
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))
        else:
            print("can not find!")

    model = model.cuda()
    print('Is model on gpu: ', next(model.parameters()).is_cuda)

    # 3.Loss

    # loss_fn1 = loss_mse().cuda()

    # using DGE in our loss funciotn
    loss_fn2 = loss_gd().cuda()

    print("loss_fn set...")

    # 4.Optim
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=monentum)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=monentum, weight_decay=weight_decay)
    print("optimizer set...")

    # 5.Train
    best_val_err = 1.0e-4


    for epoch in range(num_epochs):
        print('Starting train epoch %d / %d' % (start_epoch + epoch + 1, num_epochs + start_epoch))
        model.train()
        running_loss = 0
        count = 0
        epoch_loss = 0
        for input, depth in train_loader:

            #input_var = Variable(input.type(dtype))
            #depth_var = Variable(depth.type(dtype))
            input_var = input.cuda()
            depth_var = depth.cuda()

            output = model(input_var)
            loss,_ = loss_fn2(output, depth_var)

            print('loss: %f' % loss.data.cpu().item())
            count += 1
            running_loss += loss.data.cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if num_epochs == epoch + 1:
            # 关于保存的测试图片可以参考 loader 的写法
            # input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            input_rgb_image = input[0].data.permute(1, 2, 0).detach().numpy()
            input_gt_depth_image = depth_var[0][0].data.cpu().numpy().astype(np.float32)
            pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)

            input_gt_depth_image /= np.max(input_gt_depth_image)
            pred_depth_image /= np.max(pred_depth_image)

            plot.imsave('./result/train_input_rgb_epoch_{}.png'.format(start_epoch + epoch + 1), input_rgb_image)
            plot.imsave('./result/traingt_depth_epoch_{}.png'.format(start_epoch + epoch + 1), input_gt_depth_image,
                        cmap="viridis")
            plot.imsave('./result/train_pred_depth_epoch_{}.png'.format(start_epoch + epoch + 1), pred_depth_image,
                        cmap="viridis")

        epoch_loss = running_loss / count
        print('epoch loss:', epoch_loss)

        # validate
        model.eval()
        num_correct, num_samples = 0, 0
        loss_local = 0
        with torch.no_grad():
            for input, depth in val_loader:

                #input_var = Variable(input.type(dtype))
                #depth_var = Variable(depth.type(dtype))
                input_var = input.cuda()
                depth_var = depth.cuda()


                output = model(input_var)
                if num_epochs == epoch + 1:
                    # 关于保存的测试图片可以参考 loader 的写法
                    # input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    input_rgb_image = input[0].data.permute(1, 2, 0).detach().numpy()
                    input_gt_depth_image = depth_var[0][0].data.cpu().numpy().astype(np.float32)
                    pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)

                    input_gt_depth_image /= np.max(input_gt_depth_image)
                    pred_depth_image /= np.max(pred_depth_image)

                    plot.imsave('./result/input_rgb_epoch_{}.png'.format(start_epoch + epoch + 1), input_rgb_image)
                    plot.imsave('./result/gt_depth_epoch_{}.png'.format(start_epoch + epoch + 1), input_gt_depth_image, cmap="viridis")
                    plot.imsave('./result/pred_depth_epoch_{}.png'.format(start_epoch + epoch + 1), pred_depth_image, cmap="viridis")

                val_loss,_ = loss_fn2(output, depth_var)

                #val_loss = loss_fn(output, depth_var)
                loss_local += val_loss.data.cpu().numpy()
                num_samples += 1

        err = loss_local / num_samples
        print('val_error: %f' % err)

        epochlist.append(epoch)
        losslist1.append(epoch_loss)
        losslist2.append(err)

        fig1 = plot.figure()
        plot.plot(epochlist,losslist1,label='testloss',color='blue')
        plot.plot(epochlist,losslist2,label='varloss',color='red')
        plot.legend()
        plot.draw()
        plt.pause(2)
        plt.close(fig1)


        if err < best_val_err or epoch == num_epochs - 1:
            best_val_err = err
            torch.save({
                'epoch': start_epoch + epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, './model/Model_' + str(start_epoch + epoch + 1) + '.pth')

        if epoch % 5 == 0:
            learning_rate = learning_rate * 0.8

        if (start_epoch + epoch + 1) % 5 == 0:
            torch.save({
                'epoch': start_epoch+epoch+1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, './model/checkpoint.pth')
            print('checkpoint loaded')

    fig1 = plot.figure()
    plot.plot(epochlist,losslist1,label='testloss',color='blue')
    plot.plot(epochlist,losslist2,label='varloss',color='red')
    plot.legend()
    plot.show()



if __name__ == '__main__':
    main()
