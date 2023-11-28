import paddle
from paddle.nn import initializer as init
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return paddle.nn.Conv2D(in_channels=in_planes, out_channels=out_planes,
        kernel_size=3, stride=stride, padding=1, bias_attr=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
       init_xavier_uniform_ = init.XavierUniform(fan_in=np.sqrt(2))
       init_xavier_uniform_(m.weight)
       init_Constant = init.Constant(value=0)
       init_Constant(m.bias)
    elif classname.find('BatchNorm') != -1:
        init_Constant = init.Constant(value=1)
        init_Constant(m.weight)
        init_Constant = init.Constant(value=0)
        init_Constant(m.bias)


class wide_basic(paddle.nn.Layer):

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = paddle.nn.BatchNorm2D(num_features=in_planes)
        self.conv1 = paddle.nn.Conv2D(in_channels=in_planes, out_channels=
            planes, kernel_size=3, padding=1, bias_attr=True)
        self.dropout = paddle.nn.Dropout(p=dropout_rate)
        self.bn2 = paddle.nn.BatchNorm2D(num_features=planes)
        self.conv2 = paddle.nn.Conv2D(in_channels=planes, out_channels=
            planes, kernel_size=3, stride=stride, padding=1, bias_attr=True)
        self.shortcut = paddle.nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = paddle.nn.Sequential(paddle.nn.Conv2D(
                in_channels=in_planes, out_channels=planes, kernel_size=1,
                stride=stride, bias_attr=True))

    def forward(self, x):
        out = self.dropout(self.conv1(paddle.nn.functional.relu(x=self.bn1(x)))
            )
        out = self.conv2(paddle.nn.functional.relu(x=self.bn2(out)))
        out += self.shortcut(x)
        return out


class Wide_ResNet(paddle.nn.Layer):

    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16
        assert (depth - 4) % 6 == 0, 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor
        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n,
            dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n,
            dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n,
            dropout_rate, stride=2)
        self.bn1 = paddle.nn.BatchNorm2D(num_features=nStages[3], momentum=
            1 - 0.9)
        self.linear = paddle.nn.Linear(in_features=nStages[3], out_features
            =num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return paddle.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = paddle.nn.functional.relu(x=self.bn1(out))
        out = paddle.nn.functional.avg_pool2d(kernel_size=8, x=out,
            exclusive=False)
        out = out.reshape((out.shape[0], -1))
        out = self.linear(out)
        return out


if __name__ == '__main__':
    net = Wide_ResNet(28, 10, 0.3, 10)
    y = net(paddle.randn(shape=[1, 3, 32, 32]))
    print(y.shape)
