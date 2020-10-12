import time
import torch
import torch.nn as nn
import os
import torch.optim as optim
from torchvision import datasets, transforms
import xlwt

workbook = xlwt.Workbook(encoding='utf-8')
sheet = workbook.add_sheet('origin')
workbook2 = xlwt.Workbook(encoding='utf-8')
sheet2 = workbook2.add_sheet('5*5')
workbook3 = xlwt.Workbook(encoding='utf-8')
sheet3 = workbook3.add_sheet('7*7')
workbook4 = xlwt.Workbook(encoding='utf-8')
sheet4 = workbook4.add_sheet('0.5')
workbook5 = xlwt.Workbook(encoding='utf-8')
sheet5 = workbook5.add_sheet('0.02')
workbook6 = xlwt.Workbook(encoding='utf-8')
sheet6 = workbook6.add_sheet('50')
workbook7 = xlwt.Workbook(encoding='utf-8')
sheet7 = workbook7.add_sheet('500')
workbook8 = xlwt.Workbook(encoding='utf-8')
sheet8 = workbook8.add_sheet('0.3')
workbook9 = xlwt.Workbook(encoding='utf-8')
sheet9 = workbook9.add_sheet('0.5')
col = ('train_loss','train_acc','test_loss','test_acc')
for i in range(0,4):
    sheet.write(0,i,col[i])

batch_size = 150
max_epoch = 100
learning_rate = 0.15    # 学习率可调
class_num = 10

batch_size2 = 50
batch_size3 = 500

learning_rate2 = 0.5
learning_rate3 = 0.02


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 导入训练集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),   # 把灰度范围从0-255变换到0-1之间
                       transforms.Normalize(mean=(0.1307,), std=(0.3081,))   # 数据标准化，可以加快模型的收敛
                   ])),
    batch_size=batch_size, shuffle=True)

# 导入测试集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                         ])),
    batch_size=batch_size, shuffle=True)

# 定义网络结构"Conv-ReLU-Conv-ReLU-Pooling-FC-ReLU-Dropout-FC"
class Net(nn.Module):
    def __init__(self, class_num):
        super(Net,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),   # Conv
            nn.ReLU(inplace=True))      # ReLU
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3),  # Conv
            nn.ReLU(inplace=True),      # ReLU
            nn.MaxPool2d(kernel_size=2, stride=2))      # Pooling
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=128*12*12, out_features=128),     # FC
            nn.ReLU(inplace=True),      # ReLU
            nn.Dropout(0.1)             # Dropout
    )
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=128, out_features=class_num)      # FC
        )
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0),-1)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# 定义测试函数
def test_accuracy(model, criterion):
    correct_num = 0
    loss = 0
    torch.cuda.empty_cache()
    # calculate test accuracy and loss
    for x, label in test_loader:
        with torch.autograd.no_grad():
            x, label = x.to(DEVICE), label.to(DEVICE)
            out = model(x)
            loss += criterion(out, label)
            pred = out.argmax(dim=1)
            correct_num += pred.eq(label).float().sum().item()

    total_num = len(test_loader.dataset)
    acc = correct_num / total_num
    loss = loss / (total_num / batch_size)

    return acc, loss


# 定义模型训练函数
def train(model):
    model.train()
    epoch_start = 0
    if os.path.exists('./checkpoint/ckpt.pth'):
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        model.load_state_dict(checkpoint['model'])
        epoch_start = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_start, max_epoch):
        correct_num = 0
        loss_sum = 0
        total_num = len(train_loader.dataset)

        stime = time.time()  # start timing

        for batch_idx, (x, label) in enumerate(train_loader):
            x, label = x.to(DEVICE), label.to(DEVICE)
            out = model(x)          # 数据集输入网络进行计算,size[batch_size, class_num]
            pred = out.argmax(dim=1)    # size[batch_size]
            correct_num += pred.eq(label).float().sum().item()  # float
            loss = criterion(out, label)
            loss_sum += loss
            optimizer.zero_grad()   # 所有参数梯度清零
            loss.backward()     # 反向传播，计算梯度

            optimizer.step()    # 参数更新

            # calculate training accuracy and loss
            if (batch_idx + 1) == int(total_num / batch_size) and batch_idx != 0:
                print("epoch:{}  batch:{}  train_loss_ave:{}  train_acc:{}".format
                      (epoch + 1, batch_idx + 1, loss_sum.item() / (batch_idx + 1), correct_num / total_num))
                sheet.write(epoch+1,0,loss_sum.item() / (batch_idx + 1))
                sheet.write(epoch+1,1,correct_num / total_num)

        acc, loss = test_accuracy(model, criterion)
        sheet.write(epoch+1,2,loss.item())
        sheet.write(epoch+1,3,acc)

        etime = time.time()  # end timing

        # 保存模型
        state = {
            'model': model.state_dict(),
            'epoch': epoch + 1,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')

        print("test_acc:{}  test_loss_ave:{}  time:{}  DEVICE:{}".format(acc, loss.item(), etime - stime,DEVICE))

        workbook.save('./data.xls')
# 定义主函数
def main():
    model = Net(class_num)
    model.to(DEVICE)
    train(model)

if __name__ == '__main__':
    main()





