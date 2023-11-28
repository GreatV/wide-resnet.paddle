import paddle
from paddle.nn import initializer as init
import numpy as np


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)


def cfg(depth):
    depth_lst = [11, 13, 16, 19]
    assert depth in depth_lst, 'Error : VGGnet depth should be either 11, 13, 16, 19'
    cf_dict = {'11': [64, 'mp', 128, 'mp', 256, 256, 'mp', 512, 512, 'mp', 
        512, 512, 'mp'], '13': [64, 64, 'mp', 128, 128, 'mp', 256, 256,
        'mp', 512, 512, 'mp', 512, 512, 'mp'], '16': [64, 64, 'mp', 128, 
        128, 'mp', 256, 256, 256, 'mp', 512, 512, 512, 'mp', 512, 512, 512,
        'mp'], '19': [64, 64, 'mp', 128, 128, 'mp', 256, 256, 256, 256,
        'mp', 512, 512, 512, 512, 'mp', 512, 512, 512, 512, 'mp']}
    return cf_dict[str(depth)]


def conv3x3(in_planes, out_planes, stride=1):
    return paddle.nn.Conv2D(in_channels=in_planes, out_channels=out_planes,
        kernel_size=3, stride=stride, padding=1, bias_attr=True)


class VGG(paddle.nn.Layer):

    def __init__(self, depth, num_classes):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg(depth))
        self.linear = paddle.nn.Linear(in_features=512, out_features=
            num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_planes = 3
        for x in cfg:
            if x == 'mp':
                layers += [paddle.nn.MaxPool2D(kernel_size=2, stride=2)]
            else:
                layers += [conv3x3(in_planes, x), paddle.nn.BatchNorm2D(
                    num_features=x), paddle.nn.ReLU()]
                in_planes = x
        layers += [paddle.nn.AvgPool2D(kernel_size=1, stride=1, exclusive=
            False)]
        return paddle.nn.Sequential(*layers)


if __name__ == '__main__':
    net = VGG(16, 10)
    y = net(paddle.randn(shape=[1, 3, 32, 32]))
    print(y.shape)
