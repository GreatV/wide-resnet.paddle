import paddle
from networks.wide_resnet import Wide_ResNet
if __name__ == '__main__':
    model = Wide_ResNet(28, 10, 0.3, 10)
    x = paddle.randn(shape=[1, 3, 32, 32])
    try:
        x = paddle.static.InputSpec.from_tensor(x)
        paddle.jit.save(model, input_spec=(x,), path="./model")
        print('[JIT] paddle.jit.save successed.')
        exit(0)
    except Exception as e:
        print('[JIT] paddle.jit.save failed.')
        raise e
