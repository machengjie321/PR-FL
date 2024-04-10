import os
import torch
from setuptools import setup
#setuptools 最大的优势是它在包管理能力方面的增强。它可以使用一种更加透明的方法来查找、下载并安装依赖包
from torch.utils import cpp_extension

setup(name="sparse_conv2d",
      ext_modules=[cpp_extension.CppExtension("sparse_conv2d",
                                              [os.path.join("cpp_extension", "forward_backward.cpp")],
                                              extra_compile_args=["-std=c++14", "-fopenmp"])],
      cmdclass={"build_ext": cpp_extension.BuildExtension})

