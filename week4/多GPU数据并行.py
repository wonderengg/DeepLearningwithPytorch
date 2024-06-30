import d2l.torch
import torch
from torch import nn
from torch.nn import functional as F
scale = 0.01
#自定义网络每个层初始化参数，注意在神经网络中创建tensor张量时，不需要requires_grad=True的属性(这个属性表示需要进行梯度计算)，因为神经网络是自动求梯度，不能有requires_grad属性
#w1 = torch.randn(size=(20,1,3,3),requires_grad=True)*scale
w1 = torch.randn(size=(20,1,3,3))*scale
b1 = torch.zeros(20)
w2 = torch.randn(size=(50,20,5,5))*scale
b2 = torch.zeros(50)
w3 = torch.randn(size=(800,128))*scale
b3 = torch.zeros(128)*scale
w4 = torch.randn(size=(128,10))*scale
b4 = torch.zeros(10)*scale
params = [w1,b1,w2,b2,w3,b3,w4,b4]

#定义LeNet网络
def LeNet(X,params):
    l1_conv2d = F.conv2d(input=X,weight=params[0],bias=params[1])
    l1_relu = F.relu(input=l1_conv2d)
    l1_avgpool2d = F.avg_pool2d(input=l1_relu,kernel_size=2,stride=2)
    l2_conv2d = F.conv2d(input=l1_avgpool2d,weight=params[2],bias=params[3])
    l2_relu = F.relu(input=l2_conv2d)
    l2_avgpool2d = F.avg_pool2d(input=l2_relu,kernel_size=2,stride=2)
    l3_flatten = l2_avgpool2d.reshape(l2_avgpool2d.shape[0],-1)
    l4_linear = torch.mm(l3_flatten,params[4])+params[5]
    l4_relu = F.relu(input=l4_linear)
    l5_linear = torch.mm(l4_relu,params[6])+params[7]
    y_hat = l5_linear
    return y_hat
loss = nn.CrossEntropyLoss(reduction='none')
#将网络所有层的参数全部复制到device这个设备上，params是一个list类型，里面包含每一层的权重和偏差参数
def get_params(params,device):
    new_params = [ param.to(device) for param in params]
    #必须明确指出每层权重参数需要梯度
    for p in new_params:
        p.requires_grad_()
    return new_params
new_params = get_params(params,d2l.torch.try_gpu(0))
print('第一层偏差：',new_params[1])
print('第一层偏差梯度：',new_params[1].grad)

#数据同步，data是一个list列表,表示将data中所有元素都复制到gpu0上面，然后在gpu0上面进行data中所有元素求和，然后再把求和的结果广播（复制）到其他设备上面，这里是对每个设备在每个批量中的子批量求出的梯度求和然后把求和的结果复制到所有gpu上面
def allreduce(data):
    for i in range(1,len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1,len(data)):
        data[i][:] = data[0].to(data[i].device)

#data = [torch.ones(size=(1,2),device=d2l.torch.try_gpu(i))*(i+1) for i in range(1)]
data1 = [torch.ones(size=(1,2),device=torch.device('cuda:0')),torch.ones(size=(1,2),device=torch.device('cpu'))]
print('allreduce之前：\n',data1[0],'\n',data1[1])
allreduce(data1)
print('allreduce之后：\n',data1[0],'\n',data1[1])

#数据平均拆分到不同gpu上面
data = torch.arange(20).reshape(4,5)
#devices = [torch.device('cuda:0'),torch.device('cpu')]
devices = [torch.device('cuda:0')]
scatter = nn.parallel.scatter(data,devices)
print('data = ',data)
print('devices = ',devices)
print('scatter = ',scatter)
#将小批量样本数据和标签平均拆分到不同gpu上面，让每个gpu处理每个小批量中子样本数据
def split_batch(X,y,devices):
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X,devices),nn.parallel.scatter(y,devices))
#将小批量样本用多个gpu并行训练
def train_batch(X,y,AllDevices_params,All_devices,lr):
    # 将小批量样本数据和标签平均拆分到不同gpu上面，让每个gpu处理每个小批量中子样本数据
    X_shareds,y_shareds = split_batch(X,y,All_devices)
    #每个gpu训练自己上面的子样本数据，然后对训练出来的结果与对应的标签求loss，然后再对子样本的每个样本的loss求和。ls是一个列表类型
    # 在每个GPU上分别计算损失
    ls = [loss(LeNet(X_shared,device_params),y_shared).sum() for X_shared,y_shared,device_params in zip(X_shareds,y_shareds,AllDevices_params)]
    for l in ls:# 反向传播在每个GPU上分别执行
        #对每个gpu上面的权重参数求梯度
        l.backward()
    # 将每个GPU的所有梯度相加，并将其广播到所有GPU
    with torch.no_grad():
        #遍历网络每一个层
        for i in range(len(AllDevices_params[0])):
            #将每个gpu中对应的层的梯度求和，通常在第一个gpu上面计算，然后再广播到所有gpu上面
            allreduce([AllDevices_params[c][i].grad for c in range(len(All_devices))])
    # 在每个GPU上分别更新模型参数
    for device_params in AllDevices_params:
        #将每个gpu上面的网络权重参数进行权重更新,因此达到了所有gpu上面的权重参数值都是相同的
        d2l.torch.sgd(device_params,lr,X.shape[0])

def train(num_gpu,batch_size,lr):
    train_iter,test_iter = d2l.torch.load_data_fashion_mnist(batch_size)
    devices = [d2l.torch.try_gpu(i) for i in range(num_gpu)]
    #给每个gpu上面复制相同的网络权重参数
    # 将模型参数复制到num_gpu个GPU
    devices_params = [get_params(params,device)for device in devices]
    epochs = 10
    animator = d2l.torch.Animator('epoch','test acc',xlim=[1,epochs])
    timer = d2l.torch.Timer()
    for epoch in range(epochs):
        #处理一个epoch开始时间
        timer.start()
        for X,y in train_iter:
            # 为单个小批量执行多GPU训练
            train_batch(X,y,devices_params,devices,lr)
            torch.cuda.synchronize()
        #处理完一个epoch（一轮）所花的时间，计算训练完一遍数据集所花的时间
        timer.stop()
        #训练数据集每轮训练完后对测试数据集进行测试，输出测试精度大小
        # 在GPU0上评估模型
        animator.add(epoch+1,d2l.torch.evaluate_accuracy_gpu(lambda x : LeNet(x,devices_params[0]),test_iter,devices[0]))
    print(f'测试精度：{animator.Y[0][-1]:.2f},{timer.avg():.1f}秒/轮,在{str(devices)}')

train(1,256,0.2)#使用一个gpu设备，batch_size为256，lr为0.2
