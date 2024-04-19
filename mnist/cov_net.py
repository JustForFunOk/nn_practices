import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

        # input: [1, 28, 28]
        self.conv1 = nn.Conv2d(
            in_channels=1,  # 灰度图，通道数为1
            out_channels=10,  # 卷积核的深度，越大越能表示复杂的特征
            kernel_size=5,  # 卷积核的大小，5x5
            stride=1,  # default, 步长
            padding=0,  # default
            padding_mode='zeros',  # default
            dilation=1,  # default
            groups=1,  # default
            bias=True  # default
        )
        # conv1 weight size:
        # conv1 output: [10, 24, 24]

        self.conv2 = nn.Conv2d(
            in_channels=10,  # 上一层卷积层输出的通道数，10
            out_channels=20,  # 继续提取更深层次的特征
            kernel_size=5,  # 卷积核的大小，5x5
        )

        # conv2 output: [20, 20, 20]

        self.conv2_drop = nn.Dropout2d(
            p=0.5,  # default，随机将上层输出置为0，防止过拟合
            inplace=False  # default
        )

        self.fc1 = nn.Linear(320, 50)  # 320?
        self.fc2 = nn.Linear(50, 10)  # 输出10分类

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
