# 导入模块
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import os

# 结果文件夹
images_dir = 'images'
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# 超参数设置
image_size = 28 * 28
h_dim = 400
z_dim = 20
epochs = 20
batch_size = 64
learning_rate = 1e-3

# 获取数据集
dataset = torchvision.datasets.MNIST(root='./data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# 按照batch_size大小加载数据，并随机打乱
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

# GPU或CPU设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义VAE类
class VAE(nn.Module):
    def __init__(self, image_size=28 * 28, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    # 编码，学习高斯分布均值和方差
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    # 将高斯分布均值和方差重表示，生成隐变量z，若x~N(mu,var*var)分布，则(x-mu)/var=z~N(0,1)分布
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 解码隐变量z
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    # 计算重构值和隐变量z的分布参数
    def forward(self, x):
        mu, log_var = self.encode(x)  # 从原始样本x中学习隐变量z的分布，即学习服从高斯分布均值与方差
        z = self.reparameterize(mu, log_var)  # 将高斯分布均值与方差参数重表示，生成隐变量z
        x_reconst = self.decode(z)  # 解码隐变量z，生成重构x’
        return x_reconst, mu, log_var  # 返回重构值和隐变量的分布参数


# VAE实例化
model = VAE().to(device)

# 选择Adam优化器，传入VAE模型参数和学习率
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
for epoch in range(epochs):
    for i, (x, _) in enumerate(data_loader):
        # 前向传播
        x = x.to(device).view(-1,
                              image_size)  # batch_size*1*28*28--->batch_size*image_size  其中，image_size=1*28*28=784
        x_reconst, mu, log_var = model(x)  # 将上一步得到的x输入模型进行前向传播计算，得到重构值、服从高斯分布的隐变量z的分布参数（均值和方差）

        # 计算重构损失和KL散度
        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # 反向传播，计算误差
        loss = reconst_loss + kl_div

        # 清空上一步的残余更新参数值
        optimizer.zero_grad()

        # 误差反向传播
        loss.backward()

        # VAE模型参数更新
        optimizer.step()

        # 打印结果
        if (i + 1) % 10 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                  .format(epoch + 1, epochs, i + 1, len(data_loader), reconst_loss.item(), kl_div.item()))
    with torch.no_grad():
        # 生成随机数z，z大小是batch_size*z_dim=128*20
        z = torch.randn(batch_size, z_dim).to(device)

        # 对随机数z进行解码输出
        out = model.decode(z).view(-1, 1, 28, 28)

        # 保存采样结果
        save_image(out, os.path.join(images_dir, 'sampled-{}.png'.format(epoch + 1)))

        # 将batch_size*748的x输入模型进行前向传播计算，获取重构值out
        out, _, _ = model(x)

        # 将输入与输出拼接在一起输出保存 batch_size*1*28*（28+28)=batch*1*28*56
        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(images_dir, 'reconst-{}.png'.format(epoch + 1)))
