# -*- coding: utf-8 -*- #
# ----------------------------------------
# File Name: yolov5_model.py
# Author: 谭康
# modifier:谭康
# Version: v00
# Created: ...
# Modification: 2023/05/31
# Description: yolov5框架所用到的类，勿随意改动，详细作用请自行了解yolov5框架
# ----------------------------------------
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

""" 创建模型集合（即多个神经网络模型的组合）"""
class Ensemble(nn.ModuleList):
    # ModuleList 是 PyTorch 中的一个容器模块，它存储子模块的列表
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        遍历 self 中的所有子模块（即每个单独的模型），并使用相同的输入 x 对它们进行调用。
        同时将 augment、profile 和 visualize 参数传递给每个子模块的 forward 方法
        子模块返回的结果被存储在列表 y 中，其中每个元素都是相应模型的输出（取第一个元素 [0]）
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # 将所有模型的输出沿维度 1（通常是特征维度）连接起来，形成一个新的张量 y。

        y = torch.cat(y, 1)   # 这被称为 NMS (Non-Maximum Suppression) 集合
        return y, None  # inference, train output



"""加载模型 提供一个接口来加载预训练的YOLOv5模型 """
class DetectMultiBackend(nn.Module):
    # 它是一个 torch.nn.Module 的子类，具有自己的 forward 方法。
    """
    weights: 模型权重文件的路径，默认是 'yolov5s.pt'。
    device: 推理设备，可以是 CPU 或 GPU，默认是 CPU。
    dnn: 是否使用 DNN 后端，默认是 False。
    fp16: 是否使用半精度浮点数（FP16）推理，默认是 False。
    data: 数据配置文件，默认是 None。
    fuse: 是否融合模型中的卷积和批归一化层，默认是 True。
    """
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, fp16=False, data=None, fuse=True):
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        fp16 &= True
        stride = 32  # 步长设置为32
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        # 创建一个容器示例，用于存储模型权重文件
        model = Ensemble()
        # 加载模型权重文件
        file = Path(str(weights).strip().replace("'", ''))
        # YOLOv5 的网络结构已经在模型的定义文件中定义好了，并且在保存模型时，这些结构信息已经被序列化并保存到了.pt 文件中
        # 这里既加载了网络结构也加载了权重参数
        ckpt = torch.load(file)
        # 提取模型名称
        names = ckpt["model"].names
        # 非法类别
        illegal = ckpt.get("illegal", [])
        # 优先使用ema模型，如果不存在则使用原始模型 'model', 默认先放在cpu上   # NOTE: 因为 EMA 模型通常能提供更好的性能
        ckpt = (ckpt.get('ema') or ckpt['model']).to("cpu").float()  # FP32 model
        # 打印模型的架构
        # print("ckpt['model']", ckpt)
        ckpt.stride = torch.tensor([32.])
        # 将模型调用 fuse() 方法后, 转换为评估模式并放到容器中 ，fuse() 返回融合后的模型对象 NOTE: 这种融合可以在推理时减少计算量，从而提高推理速度
        model.append(ckpt.fuse().eval())
        # 设置所有模块的 inplace 属性为 True，这可以节省内存。
        # NOTE:在 PyTorch 中，某些操作（如激活函数、池化层等）可以被配置为原地（in-place）操作。
        # 原地操作意味着这些操作会在输入张量上直接进行修改，而不是创建新的输出张量。这可以节省内存，因为不需要额外分配存储空间来保存输出张量。
        for m in model.modules():
            m.inplace = True
        # model[-1] 表示获取 model 容器中的最后一个子模块，并将其赋值给 model 变量
        model = model[-1]
        # model.stride, 这是模型的步长（stride），通常是一个张量或列表，表示模型中各个层的步长值。
        # NOTE: 确定模型的步长，并确保其最小值为 32。
        # 这对于一些特定的任务（如目标检测）非常重要，因为步长会影响特征图的分辨率。
        stride = max(int(model.stride.max()), 32)
        # 获取类别的名称
        # names = model.module.names if hasattr(model, 'module') else model.names
        # 根据 fp16 参数的值来决定是否将模型转换为半精度格式
        model.half() if fp16 else model.float()
        # 这是一个动态量化函数，可以将模型中的某些层（如线性层）量化为 8 位整数（QINT8）。
        # 量化可以进一步减少模型的大小和推理时间，但可能会稍微降低精度。
        # model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear},  # {torch.nn.Linear}：指定要量化的层类型，这里是线性层
        #                                             dtype=torch.qint8)
        print(f"detect model to {device}")
        # 将模型移动到指定的设备（CPU 或 GPU）。to(device) 方法会递归地将模型及其所有子模块、参数和缓冲区移动到指定的设备。
        self.model = model.to(device)
        # 将所有局部变量添加到类实例的属性中。这意味着你可以通过 self 访问这些变量，而不需要显式地为每个变量赋值。
        self.__dict__.update(locals())  # locals()：返回一个包含当前局部变量的字典。
        # NOTE : 这种做法虽然简洁，但也有一些潜在的风险：
        # 它会覆盖类实例中已有的属性。
        # 难以追踪哪些变量被添加到了类实例中，增加了调试的难度。
        # 可能会导致命名冲突或意外的行为。


    """ 模型的前向传播 """
    def forward(self, im):
        """
        Args:
            im: 输入图像张量，形状为 [batch_size, channels, height, width] 的 PyTorch 张量。
        Returns:
        """
        # 如果启用了半精度浮点数（FP16）推理，并且输入张量 im 不是 FP16 格式，则将其转换为 FP16。
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        # 调用 self.model 的前向传播方法，传入输入张量 im，并获取模型的输出 y
        # TODO 前向传播推理方法在哪里？
        y = self.model(im)

        if isinstance(y, (list, tuple)):
            # 如果是单个元素的列表或元组（即 len(y) == 1），则只返回第一个元素，并调用 self.from_numpy 进行转换。
            # 如果是多个元素的列表或元组，则对每个元素都调用 self.from_numpy 进行转换，并返回一个包含所有转换后结果的列表。
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            # 如果输出不是列表或元组，则直接调用 self.from_numpy 进行转换。
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x



if __name__ == '__main__':
    ckpt = r'C:\Users\Administrator\Desktop\yolov5\yolov5n.pt'
    DetectMultiBackend(ckpt)
