#实现自动求导的功能，如果C++包引入了，就使用C++包来加速运算，否则就使用torch中的来实现自动求导功能
import torch
import torch.sparse as sparse

sparse_conv2d_imported = True
try:
    import sparse_conv2d      #即引入c++包

except ImportError:
    sparse_conv2d_imported = False


class AddmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, weight: sparse.FloatTensor, dense_weight_placeholder, inp):
        if bias is None:
            out = sparse.mm(weight, inp)
        else:
            out = sparse.addmm(bias, weight, inp)
        ctx.save_for_backward(bias, weight, inp)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        bias, weight, inp = ctx.saved_tensors
        grad_bias = grad_input = None
        if bias is not None:
            grad_bias = grad_output.sum(1).reshape((-1, 1))
        grad_weight = grad_output.mm(inp.t())
        if ctx.needs_input_grad[3]:
            grad_input = torch.mm(weight.t(), grad_output)

        return grad_bias, None, grad_weight, grad_input


if sparse_conv2d_imported:
    class SparseConv2dFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, weight, dense_weight_placeholder, kernel_size, bias, stride, padding):
            out, f_input, fgrad_input = sparse_conv2d.forward(inp, weight, kernel_size, bias, stride, padding)
            ctx.save_for_backward(inp, weight, f_input, fgrad_input)
            ctx.kernel_size = kernel_size
            ctx.stride = stride
            ctx.padding = padding
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input, grad_weight, grad_bias = sparse_conv2d.backward(grad_output,
                                                                        ctx.saved_tensors[0],
                                                                        ctx.saved_tensors[1],
                                                                        ctx.kernel_size,
                                                                        ctx.stride,
                                                                        ctx.padding,
                                                                        ctx.saved_tensors[2],
                                                                        ctx.saved_tensors[3],
                                                                        (True, True, True))
            return grad_input, None, grad_weight, None, grad_bias, None, None


    class DenseConv2dFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, weight, kernel_size, bias, stride, padding):
            weight2d = weight.data.reshape((weight.size(0), -1))
            out, f_input, fgrad_input = sparse_conv2d.forward(inp, weight2d, kernel_size, bias, stride, padding)
            ctx.save_for_backward(inp, weight2d, f_input, fgrad_input, weight)
            ctx.kernel_size = kernel_size
            ctx.stride = stride
            ctx.padding = padding
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input, grad_weight2d, grad_bias = sparse_conv2d.backward(grad_output,
                                                                          ctx.saved_tensors[0],
                                                                          ctx.saved_tensors[1],
                                                                          ctx.kernel_size,
                                                                          ctx.stride,
                                                                          ctx.padding,
                                                                          ctx.saved_tensors[2],
                                                                          ctx.saved_tensors[3],
                                                                          (True, True, True))
            grad_weight = grad_weight2d.reshape_as(ctx.saved_tensors[4])
            return grad_input, grad_weight, None, grad_bias, None, None

else:
    class SparseConv2dFunction(torch.autograd.Function):
        @staticmethod
        def apply(inp, weight, dense_weight_placeholder, kernel_size, bias, stride, padding):
            size_4d = (weight.size(0), -1, *kernel_size)
            with torch.no_grad():
                dense_weight_placeholder.zero_()
                dense_weight_placeholder.add_(weight.to_dense())
            return torch.nn.functional.conv2d(inp, dense_weight_placeholder.view(size_4d), bias, stride, padding)


    class DenseConv2dFunction(torch.autograd.Function):
        @staticmethod
        def apply(inp, weight, kernel_size, bias, stride, padding):
            size_4d = (weight.size(0), -1, *kernel_size)
            return torch.nn.functional.conv2d(inp, weight.reshape(size_4d), bias, stride, padding)
            #返回值：一个Tensor变量
            #作用：在输入图像input中使用filters做卷积运算


import torch.nn as nn

import torch

class AlexNet(nn.Module):

    def __init__(self):

        super(AlexNet, self).__init__()

        self.features=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # 使用卷积层，输入为3，输出为64，核大小为11，步长为4
            nn.ReLU(inplace=True), # 使用激活函数
            nn.MaxPool2d(kernel_size=3, stride=2), # 使用最大池化，这里的大小为3，步长为2

            nn.Conv2d(64, 192, kernel_size=5, padding=2), # 使用卷积层，输入为64，输出为192，核大小为5，步长为2

            nn.ReLU(inplace=True),# 使用激活函数

            nn.MaxPool2d(kernel_size=3, stride=2), # 使用最大池化，这里的大小为3，步长为2

            nn.Conv2d(192, 384, kernel_size=3, padding=1), # 使用卷积层，输入为192，输出为384，核大小为3，步长为1

            nn.ReLU(inplace=True),# 使用激活函数

            nn.Conv2d(384, 256, kernel_size=3, padding=1),# 使用卷积层，输入为384，输出为256，核大小为3，步长为1

            nn.ReLU(inplace=True),# 使用激活函数

            nn.Conv2d(256, 256, kernel_size=3, padding=1),# 使用卷积层，输入为256，输出为256，核大小为3，步长为1

            nn.ReLU(inplace=True),# 使用激活函数

            nn.MaxPool2d(kernel_size=3, stride=2)) # 使用最大池化，这里的大小为3，步长为2

        self.avgpool=nn.AdaptiveAvgPool2d((6, 6))

        self.classifier=nn.Sequential(

        nn.Dropout(),# 使用Dropout来减缓过拟合

        nn.Linear(256 * 6 * 6, 4096), # 全连接，输出为4096

        nn.ReLU(inplace=True),# 使用激活函数

        nn.Dropout(),# 使用Dropout来减缓过拟合

        nn.Linear(4096, 4096), # 维度不变，因为后面引入了激活函数，从而引入非线性

        nn.ReLU(inplace=True), # 使用激活函数

        nn.Linear(4096, 1000)) #ImageNet默认为1000个类别，所以这里进行1000个类别分类

    def forward(self, x):

        x=self.features(x)

        x=self.avgpool(x)

        x=torch.flatten(x, 1)

        x=self.classifier(x)

        return x

    def alexnet(num_classes, device, pretrained_weights=""):

        net=AlexNet() # 定义AlexNet

        if pretrained_weights: # 判断预训练模型路径是否为空，如果不为空则加载

            net.load_state_dict(torch.load(pretrained_weights,map_location=device))

            num_fc=net.classifier[6].in_features # 获取输入到全连接层的输入维度信息

            net.classifier[6]=torch.nn.Linear(in_features=num_fc, out_features=num_classes) # 根据数据集的类别数来指定最后输出的out_features数目

        return net