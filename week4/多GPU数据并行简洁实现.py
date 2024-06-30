import d2l.torch
import torch
from torch import nn
"""稍加修改的ResNet-18模型"""
def resnet18(num_classes,in_channels=1):
    def Residul_Block(in_channels,out_channels,num_residul,first_residul_block=False):
        blk = []
        for i in range(num_residul):
            if i==0 and not first_residul_block:
                blk.append(d2l.torch.Residual(in_channels,out_channels,use_1x1conv=True,strides=2))
            else:
                blk.append(d2l.torch.Residual(out_channels,out_channels,use_1x1conv=False,strides=1))
        return nn.Sequential(*blk)
    # 模型使用了更小的卷积核、步长和填充，而且删除了最大汇聚层
    net = nn.Sequential(nn.Conv2d(in_channels,64,kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
    net.add_module('resnet_block1',Residul_Block(in_channels=64,out_channels=64,num_residul=2,first_residul_block=True))
    net.add_module('resnet_block2',Residul_Block(in_channels=64,out_channels=128,num_residul=2,first_residul_block=False))
    net.add_module('resnet_block3',Residul_Block(in_channels=128,out_channels=256,num_residul=2,first_residul_block=False))
    net.add_module('resnet_block4',Residul_Block(in_channels=256,out_channels=512,num_residul=2,first_residul_block=False))
    net.add_module('adaptivepool2d',nn.AdaptiveAvgPool2d((1,1)))
    net.add_module('Flatten',nn.Flatten())
    net.add_module('Linear',nn.Linear(in_features=512,out_features=num_classes))
    return net
print(resnet18(10))

def train(net,num_gpu,lr,batch_size,epochs):
    train_iter,test_iter = d2l.torch.load_data_fashion_mnist(batch_size)
    #获取所有GPU
    devices = [d2l.torch.try_gpu(i) for i in range(num_gpu)]
    #初始化网络权重参数
    def init_weight(m):
        if type(m) in [nn.Linear,nn.Conv2d]:
            nn.init.normal_(m.weight,mean=0,std=0.01)
    net.apply(init_weight)
    # 在多个GPU对模型进行数据并行，将网络net使用多GPU数据并行进行训练
    net = nn.DataParallel(module=net,device_ids=devices)
    net = net.to(devices[0])
    optim = torch.optim.SGD(net.parameters(),lr=lr)
    loss = nn.CrossEntropyLoss()
    timer = d2l.torch.Timer()
    animator = d2l.torch.Animator('epoch','test acc',xlim=[1,epochs])
    for epoch in range(epochs):
        net.train()#网络用于训练，必须有，因为模型中有batch_normalization层，batch_norm层在训练和测试两种状态下的权重参数不一样
        timer.start()
        for X,y in train_iter:
            optim.zero_grad()
            X = X.to(devices[0])#将输入数据和label都复制到第一个GPU上面
            y = y.to(devices[0])
            # 模型通过上面nn.DataParallel()多GPU数据并行函数得知使用了多少个GPU,同时将小批量数据平均分配到所有GPU上，然后计算loss和梯度，最后再把每个GPU上面的梯度聚合在一起，然后广播到所有GPU上面，进行网络参数权重更新
            y_hat = net(X)
            ls = loss(y_hat,y)
            ls.backward()
            optim.step()
        timer.stop()
        net.eval()#模型用于测试，必须有，因为模型中有batch_normalization层
        animator.add(epoch+1,(d2l.torch.evaluate_accuracy_gpu(net,test_iter),))
    print(f'测试精度：{animator.Y[0][-1]:.2f},{timer.avg():.2f}秒/轮,在{str(devices)}')
#网络初始化
resnet = resnet18(num_classes=10,in_channels=1)
lr,epochs,batch_size=0.1,10,256
num_gpu= 2
train(resnet,num_gpu,lr,batch_size,epochs)
