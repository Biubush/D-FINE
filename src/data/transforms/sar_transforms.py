"""
SAR图像特定的预处理和数据增强变换

这个模块包含了针对SAR图像特性的增强方法:
1. 斑点噪声处理
2. SAR图像特定的对比度增强
3. 直方图均衡化
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance
import cv2

from ...core import register
from ._transforms import Transform


@register()
class SARSpeckleNoise(Transform):
    """添加SAR图像特有的斑点噪声

    Args:
        prob (float): 应用此变换的概率
        intensity (float): 噪声强度，默认为0.1
    """
    def __init__(self, prob=0.5, intensity=0.1):
        super().__init__()
        self.prob = prob
        self.intensity = intensity
        
    def _apply_image(self, img):
        if random.random() > self.prob:
            return img
            
        # 将PIL图像转换为numpy数组
        if isinstance(img, Image.Image):
            img_np = np.array(img).astype(np.float32)
        else:
            img_np = img.copy()
            
        # 添加乘性斑点噪声（符合SAR图像特性）
        noise = np.random.gamma(
            shape=1.0,
            scale=self.intensity,
            size=img_np.shape[:2]
        )
        
        # 对每个通道应用噪声
        if len(img_np.shape) == 3:
            noise = np.expand_dims(noise, axis=2)
            noise = np.repeat(noise, img_np.shape[2], axis=2)
            
        img_np = img_np * noise
        
        # 裁剪值到合理范围
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        # 转回PIL图像
        if isinstance(img, Image.Image):
            return Image.fromarray(img_np)
        return img_np
        
    def _apply_coords(self, coords):
        # 噪声不改变坐标
        return coords


@register()
class SARHistogramEqualization(Transform):
    """对SAR图像进行直方图均衡化，增强对比度
    
    Args:
        prob (float): 应用此变换的概率
        clip_limit (float): 对比度限制，默认为2.0
        tile_grid_size (tuple): 瓦片网格大小，默认为(8, 8)
    """
    def __init__(self, prob=0.5, clip_limit=2.0, tile_grid_size=(8, 8)):
        super().__init__()
        self.prob = prob
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def _apply_image(self, img):
        if random.random() > self.prob:
            return img
            
        # 将PIL图像转换为numpy数组
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img.copy()
            
        # 创建CLAHE对象（限制对比度的自适应直方图均衡化）
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, 
            tileGridSize=self.tile_grid_size
        )
        
        # 对每个通道应用CLAHE
        if len(img_np.shape) == 3:
            img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
            img_np = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        else:
            img_np = clahe.apply(img_np)
            
        # 转回PIL图像
        if isinstance(img, Image.Image):
            return Image.fromarray(img_np)
        return img_np
        
    def _apply_coords(self, coords):
        # 直方图均衡化不改变坐标
        return coords


@register()
class SAREdgeEnhancement(Transform):
    """增强SAR图像中的边缘，有助于目标边界检测
    
    Args:
        prob (float): 应用此变换的概率
        alpha (float): 边缘增强强度，默认为0.5
    """
    def __init__(self, prob=0.5, alpha=0.5):
        super().__init__()
        self.prob = prob
        self.alpha = alpha
        
    def _apply_image(self, img):
        if random.random() > self.prob:
            return img
            
        # 将PIL图像转换为numpy数组
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img.copy()
            
        # 转为灰度图
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np.copy()
            
        # 使用Laplacian算子检测边缘
        edge = cv2.Laplacian(gray, cv2.CV_64F)
        edge = np.uint8(np.absolute(edge))
        
        # 将边缘添加回原图
        if len(img_np.shape) == 3:
            edge = np.expand_dims(edge, axis=2)
            edge = np.repeat(edge, img_np.shape[2], axis=2)
            
        img_np = np.clip(img_np + self.alpha * edge, 0, 255).astype(np.uint8)
        
        # 转回PIL图像
        if isinstance(img, Image.Image):
            return Image.fromarray(img_np)
        return img_np
        
    def _apply_coords(self, coords):
        # 边缘增强不改变坐标
        return coords


@register()
class SARBrightness(Transform):
    """针对SAR图像的亮度调整
    
    Args:
        prob (float): 应用此变换的概率
        factor_range (tuple): 亮度调整因子范围，默认为(0.5, 1.5)
    """
    def __init__(self, prob=0.5, factor_range=(0.7, 1.3)):
        super().__init__()
        self.prob = prob
        self.factor_range = factor_range
        
    def _apply_image(self, img):
        if random.random() > self.prob:
            return img
            
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
            
        factor = random.uniform(*self.factor_range)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
        
        return img
        
    def _apply_coords(self, coords):
        # 亮度调整不改变坐标
        return coords 