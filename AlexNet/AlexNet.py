import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import transforms as transforms
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
lr=0.001
epoch=200
BatchSize=100
device = 'cuda' if torch.cuda.is_available() else 'cpu'




train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=100, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=100, shuffle=False)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# print(trainset.data.shape)
NUM_CLASSES = 10
class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


net=AlexNet().to(device)


optimizer=optim.Adam(net.parameters(),lr=lr)
loss_func=torch.nn.CrossEntropyLoss().to(device)

for epoch in range(1,1+epoch):
    train_loss = 0
    train_correct = 0
    total = 0
    for batch_num,(data,target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out=net(data)
        loss=loss_func(out,target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        prediction = torch.max(out, 1)
        total += target.size(0)
        # train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
        train_correct=float((prediction[1].cpu().numpy() == target.cpu().numpy()).astype(int).sum()) / float(target.cpu().size(0))
        print('| train loss: %.4f' % loss.cpu().data.numpy(),
              '| test accuracy: %.2f' %  train_correct)








        #     out=alexNet(x)
        #     print(out.cpu().data[0])
        #     # out=out.view(1080)
        #     # print(out.shape)
        #     # print(y.shape)
        #     # print(y)
        #     # print(out)
        #
        #     loss=loss_func(out,y)
        #
        #     loss.backward()
        #     optimizer.step()
        #     pre_y = torch.max(out.cpu(), 1)[1].data.numpy()
        #     print('pre_y',pre_y)
        #     accuracy = float((pre_y == y.cpu().data.numpy()).astype(int).sum()) / float(y.size(0))
        #     print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.2f' % accuracy)











# train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()
# EPOCH = 150
# BATCH_SIZE = 50
# LR = 0.5
#
# # plt.imshow(train_set_x_orig[25])
# # plt.show()
#
#
# x=torch.from_numpy(train_set_x_orig)/255.0
# y=torch.from_numpy(train_set_y)
# x=x.float().permute(0,3,1,2).float().cuda()
# y=y[0].cuda()
#
# x_t=torch.from_numpy(test_set_x_orig)/255.0
# y_t=torch.from_numpy(test_set_y)
# x_t=x_t.float().permute(0,3,1,2).float().cuda()
# y_t=y_t[0].cuda()
#
#
#
# class AlexNet(nn.Module):
#     def __init__(self, num_classes=1):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             #第一个卷积层，3个输入通道，64个11*11大小的卷积核
#             nn.Conv2d(3, 64, kernel_size=12, stride=1, padding=1),
#             #卷积层后面跟一个relu激活函数
#             nn.ReLU(inplace=True),
#             #最大池化操作
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             #以上可以作为一个block,整个网络基本就是多个block的叠加
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, 6),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         x = self.classifier(x)
#         return x
#
# alexNet=AlexNet()
# loss_func=torch.nn.CrossEntropyLoss()
# optimizer=torch.optim.SGD(alexNet.parameters(),lr=LR,momentum=0.9)
# alexNet.cuda()
# for epoch in range(EPOCH):
#     optimizer.zero_grad()
#     print('epoch',epoch)
#
#     out=alexNet(x)
#     print(out.cpu().data[0])
#     # out=out.view(1080)
#     # print(out.shape)
#     # print(y.shape)
#     # print(y)
#     # print(out)
#
#     loss=loss_func(out,y)
#
#     loss.backward()
#     optimizer.step()
#     pre_y = torch.max(out.cpu(), 1)[1].data.numpy()
#     print('pre_y',pre_y)
#     accuracy = float((pre_y == y.cpu().data.numpy()).astype(int).sum()) / float(y.size(0))
#     print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.2f' % accuracy)
#
#
# test_output = alexNet(x_t)
# pred_y = torch.max(test_output, 1)[1].data.numpy()
# accurac = float((pred_y == y_t.data.numpy()).astype(int).sum()) / float(y_t.size(0))
# print( '| test accuracy: %.2f' % accurac)
# print(pred_y, 'prediction number')
# print(y_t.numpy(), 'real number')

