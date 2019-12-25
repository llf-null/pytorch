import torch.nn as nn
import torch
import numpy as np
import pickle
from PIL import Image
#import cv2
import time
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




print(device)
batch_size=64
num_workers=4
batch_size = 16
validation_split = .3
shuffle_dataset = True
random_seed= 42
# Load training images
train_images = pickle.load(open("D:/pytorch_program/full_CNN_train.p", "rb" ))

# Load image labels
label = pickle.load(open("D:/pytorch_program/full_CNN_labels.p", "rb" ))
# Subset the dataset into train dataset and test dataset 
# Make into arrays as the neural network wants these
train_images = np.array(train_images)
#print(train_images.shape)
train_images.shape=(12764, 3,80, 160)
#print(train_images.shape)
labels = np.array(label)
#print(labels.shape)
labels.shape=(12764, 1,80, 160)
#print(labels[0].shape)
#print(len(train_images))
train_images = torch.from_numpy(train_images)
train_images=train_images.type(torch.FloatTensor)
#train_images = train_images.cuda()
labels = torch.from_numpy(labels)
labels=labels.type(torch.FloatTensor)
#print(labels[0])
#labels = labels.cuda()

# #change the size of data
# str1='Train_image/image'
# str2='.jpg'
# filename=str1+str(51)+str2
# image=Image.open(filename).convert('RGB') #读取图像，转换为三维矩阵
# image=image.resize((224,224),Image.ANTIALIAS) #将其转换为要求的输入大小224*224
# transform=transforms.Compose([transforms.ToTensor()])
# img = transform(image) #转为Tensor
# train_images_tensor=img.resize(1,3,224,224)
# for i in range(50):
#     filename=str1+str(i)+str2
#     image=Image.open(filename).convert('RGB') #读取图像，转换为三维矩阵
#     image=image.resize((224,224),Image.ANTIALIAS) #将其转换为要求的输入大小224*224
#     transform=transforms.Compose([transforms.ToTensor()])
#     img = transform(image) #转为Tensor
#     img=img.resize(1,3,224,224)
#     train_images_tensor=torch.cat((train_images_tensor,img),0)
# print(train_images_tensor.size())

# Creating data indices for training and validation splits:
dataset_size = len(train_images)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

train_datas = torch.utils.data.DataLoader(train_images, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
train_labels = torch.utils.data.DataLoader(labels, batch_size=batch_size,sampler=train_sampler, num_workers=num_workers)
test_datas = torch.utils.data.DataLoader(train_images, batch_size=batch_size,sampler=test_sampler, num_workers=num_workers)
test_labels = torch.utils.data.DataLoader(labels, batch_size=batch_size,sampler=test_sampler, num_workers=num_workers)




class conv2DBatchNormRelu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias=True,dilation=1,is_batchnorm=True):
        super(conv2DBatchNormRelu,self).__init__()
        if is_batchnorm:
            self.cbr_unit=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,
                          bias=bias,dilation=dilation),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.cbr_unit=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias, dilation=dilation),
                nn.ReLU(inplace=True)
            )

    def forward(self,inputs):
        outputs=self.cbr_unit(inputs)
        return outputs

class segnetDown2(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(segnetDown2,self).__init__()
        self.conv1=conv2DBatchNormRelu(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv2=conv2DBatchNormRelu(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.maxpool_with_argmax=nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)

    def forward(self,inputs):
        outputs=self.conv1(inputs)
        outputs=self.conv2(outputs)
        unpooled_shape=outputs.size()
        outputs,indices=self.maxpool_with_argmax(outputs)
        return outputs,indices,unpooled_shape

class segnetDown3(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(segnetDown3,self).__init__()
        self.conv1=conv2DBatchNormRelu(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv2=conv2DBatchNormRelu(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv3=conv2DBatchNormRelu(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.maxpool_with_argmax=nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)

    def forward(self,inputs):
        outputs=self.conv1(inputs)
        outputs=self.conv2(outputs)
        outputs=self.conv3(outputs)
        unpooled_shape=outputs.size()
        outputs,indices=self.maxpool_with_argmax(outputs)
        return outputs,indices,unpooled_shape


class segnetUp2(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(segnetUp2,self).__init__()
        self.unpool=nn.MaxUnpool2d(2,2)
        self.conv1=conv2DBatchNormRelu(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv2=conv2DBatchNormRelu(out_channels,out_channels,kernel_size=3,stride=1,padding=1)

    def forward(self,inputs,indices,output_shape):
        outputs=self.unpool(inputs,indices=indices,output_size=output_shape)
        outputs=self.conv1(outputs)
        outputs=self.conv2(outputs)
        return outputs

class segnetUp3(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(segnetUp3,self).__init__()
        self.unpool=nn.MaxUnpool2d(2,2)
        self.conv1=conv2DBatchNormRelu(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv2=conv2DBatchNormRelu(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv3=conv2DBatchNormRelu(out_channels,out_channels,kernel_size=3,stride=1,padding=1)

    def forward(self,inputs,indices,output_shape):
        outputs=self.unpool(inputs,indices=indices,output_size=output_shape)
        outputs=self.conv1(outputs)
        outputs=self.conv2(outputs)
        outputs=self.conv3(outputs)
        return outputs

class segnet(nn.Module):
    def __init__(self,in_channels=3,num_classes=1):
        super(segnet,self).__init__()
        self.down1=segnetDown2(in_channels=in_channels,out_channels=64)
        self.down2=segnetDown2(64,128)
        self.down3=segnetDown3(128,256)
        self.down4=segnetDown3(256,512)
        self.down5=segnetDown3(512,512)

        self.up5=segnetUp3(512,512)
        self.up4=segnetUp3(512,256)
        self.up3=segnetUp3(256,128)
        self.up2=segnetUp2(128,64)
        self.up1=segnetUp2(64,64)
        self.finconv=conv2DBatchNormRelu(64,num_classes,3,1,1)

    def forward(self,inputs):
        down1,indices_1,unpool_shape1=self.down1(inputs)
        down2,indices_2,unpool_shape2=self.down2(down1)
        down3,indices_3,unpool_shape3=self.down3(down2)
        down4,indices_4,unpool_shape4=self.down4(down3)
        down5,indices_5,unpool_shape5=self.down5(down4)

        up5=self.up5(down5,indices=indices_5,output_shape=unpool_shape5)
        up4=self.up4(up5,indices=indices_4,output_shape=unpool_shape4)
        up3=self.up3(up4,indices=indices_3,output_shape=unpool_shape3)
        up2=self.up2(up3,indices=indices_2,output_shape=unpool_shape2)
        up1=self.up1(up2,indices=indices_1,output_shape=unpool_shape1)
        outputs=self.finconv(up1)

        return outputs

if __name__=="__main__":
    inputs=torch.ones(2,3,80,160)
    model=segnet()
    shape=inputs.size()
    #print(shape)
    #print(model(inputs).size())
    #print(model(inputs))
    #print(model)





def evaluate_accuracy(test_data,test_label, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in zip(test_data,test_label):
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n

def train_ch5(net, train_data,train_label,test_data,test_label, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in zip(train_data,train_label):
            X = X.to(device)
            y = y.to(device)
            #pdb.set_trace()
            y_hat = net(X)
            l = loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_data,test_label, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    torch.save(net,'D:/pytorch_program/model/lanenet.pkl')
    print("finish----------------------------------------")





batch_size=16
lr,num_epochs=0.001,5
optimizer=torch.optim.Adagrad(model.parameters(),lr=lr)
train_ch5(model,train_datas,train_labels,test_datas,test_labels, batch_size, optimizer, device, num_epochs)