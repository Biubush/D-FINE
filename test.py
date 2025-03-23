"""
SAR飞机检测模型测试脚本
用于测试训练好的D-FINE模型并生成带有标注的检测结果

使用方法:
    python test.py -c CONFIG -r WEIGHTS -i INPUT [-o OUTPUT] [-t THRESHOLD] [-d DEVICE]

示例:
    # 使用GPU进行推理
    python test.py -c configs/dfine/custom/sar_dfine_s.yml -r output/sar_dfine_s/best_stg2.pth -i test_image.jpg -o result.jpg

    # 使用CPU进行推理
    python test.py -c configs/dfine/custom/sar_dfine_s.yml -r output/sar_dfine_s/best_stg2.pth -i test_image.jpg -d cpu

    # 调整检测阈值
    python test.py -c configs/dfine/custom/sar_dfine_s.yml -r output/sar_dfine_s/best_stg2.pth -i test_image.jpg -t 0.6
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.core import YAMLConfig

# SAR飞机数据集的类别名称
SAR_AIRCRAFT_CLASSES = ['A220', 'A320/321', 'A330', 'ARJ21', 'Boeing737', 'Boeing787', 'other']

# 为不同类别定义不同颜色
CLASS_COLORS = [
    (255, 0, 0),    # 红色 - A220
    (0, 255, 0),    # 绿色 - A320/321
    (0, 0, 255),    # 蓝色 - A330
    (255, 255, 0),  # 黄色 - ARJ21
    (255, 0, 255),  # 洋红色 - Boeing737
    (0, 255, 255),  # 青色 - Boeing787
    (255, 165, 0)   # 橙色 - other
]

class ModelTester:
    """SAR飞机检测模型测试器
    
    该类负责加载训练好的模型，处理图像，并生成带有检测结果的可视化输出。
    每个检测到的飞机都会用不同颜色的边界框标注，并显示类别名称和置信度。
    
    属性:
        device (str): 运行设备，如'cuda:0'或'cpu'
        config_path (str): 模型配置文件路径
        weights_path (str): 模型权重文件路径
        model: 加载好的模型
        transform: 图像预处理转换
    """
    
    def __init__(self, config_path, weights_path, device="cuda:0"):
        """初始化模型测试器
        
        Args:
            config_path (str): 配置文件路径，通常为YAML格式
            weights_path (str): 权重文件路径，通常为.pth格式
            device (str): 运行设备，如'cuda:0'或'cpu'
        """
        self.device = device
        self.config_path = config_path
        self.weights_path = weights_path
        self.model = self._load_model()
        self.transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
    
    def _load_model(self):
        """加载模型
        
        从配置文件和权重文件加载模型，并将其设置为评估模式
        
        Returns:
            加载好的模型
        """
        print(f"加载配置: {self.config_path}")
        print(f"加载权重: {self.weights_path}")
        
        # 加载配置和权重
        cfg = YAMLConfig(self.config_path, resume=self.weights_path)
        
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        # 加载训练模式状态并转换为部署模式
        checkpoint = torch.load(self.weights_path, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
        
        cfg.model.load_state_dict(state)
        
        # 创建推理模型
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs
        
        # 实例化模型并移至指定设备
        model = Model().to(self.device)
        model.eval()  # 设置为评估模式
        print("模型加载成功")
        return model
    
    def process_image(self, image_path, output_path="result.jpg", conf_threshold=0.4):
        """处理单张图像
        
        加载图像，运行模型推理，并生成带有检测结果的可视化输出。
        
        Args:
            image_path (str): 输入图像路径
            output_path (str): 输出图像路径，默认为"result.jpg"
            conf_threshold (float): 置信度阈值，默认为0.4
        
        Returns:
            检测结果列表: [(类别, 置信度, 边界框), ...]
            每个元素为三元组，分别是:
            - 类别名称 (str)
            - 置信度分数 (float)
            - 边界框坐标 [x1, y1, x2, y2] (list)
        """
        # 加载图像
        print(f"处理图像: {image_path}")
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        print(f"图像尺寸: {w}x{h}")
        orig_size = torch.tensor([[w, h]]).to(self.device)
        
        # 预处理图像
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 模型推理
        print("执行模型推理...")
        with torch.no_grad():
            labels, boxes, scores = self.model(img_tensor, orig_size)
        
        # 可视化结果
        result_image = self._visualize_detections(
            image, labels[0], boxes[0], scores[0], conf_threshold
        )
        
        # 保存结果
        result_image.save(output_path)
        print(f"结果已保存至: {output_path}")
        
        # 返回检测结果
        results = []
        for i, score in enumerate(scores[0]):
            if score > conf_threshold:
                label_idx = labels[0][i].item()
                label_name = SAR_AIRCRAFT_CLASSES[label_idx] if 0 <= label_idx < len(SAR_AIRCRAFT_CLASSES) else f"cls:{label_idx}"
                box = boxes[0][i].tolist()
                results.append((label_name, score.item(), box))
        
        return results
    
    def _visualize_detections(self, image, labels, boxes, scores, conf_threshold=0.4):
        """可视化检测结果
        
        在图像上绘制检测框、类别名称和置信度分数。
        不同类别使用不同颜色的框，文字大小会根据框的大小自适应调整。
        
        Args:
            image (PIL.Image): PIL图像对象
            labels (torch.Tensor): 类别标签张量
            boxes (torch.Tensor): 边界框坐标张量
            scores (torch.Tensor): 置信度分数张量
            conf_threshold (float): 置信度阈值，默认为0.4
        
        Returns:
            PIL.Image: 带有检测结果的图像
        """
        # 创建副本以保留原始图像
        result_im = image.copy()
        width, height = result_im.size
        
        # 筛选高于阈值的检测框
        mask = scores > conf_threshold
        filtered_labels = labels[mask]
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        
        # 如果没有检测到物体，直接返回原图
        if len(filtered_labels) == 0:
            print("未检测到任何物体!")
            return result_im
        
        # 准备绘制
        result_im = result_im.convert("RGBA")
        overlay = Image.new('RGBA', result_im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # 尝试加载字体
        font = None
        for font_name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "simhei.ttf", "simsun.ttc"]:
            try:
                font = ImageFont.truetype(font_name, 15)  # 基础字体大小
                break
            except:
                continue
        
        # 处理每个检测框
        for i, (label_idx, box, score) in enumerate(zip(filtered_labels, filtered_boxes, filtered_scores)):
            # 获取类别和对应颜色
            label_idx = label_idx.item()
            if 0 <= label_idx < len(SAR_AIRCRAFT_CLASSES):
                label_name = SAR_AIRCRAFT_CLASSES[label_idx]
                color_rgb = CLASS_COLORS[label_idx]
            else:
                label_name = f"cls:{label_idx}"
                color_rgb = (255, 255, 255)
            
            # 边界框坐标
            x1, y1, x2, y2 = box.tolist()
            
            # 计算框的大小，用于自适应字体和线宽
            box_width = x2 - x1
            box_height = y2 - y1
            box_size = min(box_width, box_height)
            
            # 自适应线宽和字体大小
            line_width = max(2, int(box_size / 50))
            font_size = max(12, int(box_size / 15))
            
            # 尝试重新加载合适大小的字体
            if font:
                try:
                    font = ImageFont.truetype(font._file, font_size)
                except:
                    pass
            
            # 创建标签文本
            text = f"{label_name}: {score.item():.2f}"
            
            # 计算文本大小
            if font:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            else:
                text_width = len(text) * font_size // 2
                text_height = font_size
            
            # 确保文本在图像内部
            text_x = max(0, min(width - text_width, x1))
            text_y = max(0, y1 - text_height) if y1 > text_height else y1
            
            # 边界框 - 带透明度
            box_color = tuple(list(color_rgb) + [200])  # 透明度为200/255
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)
            
            # 文本背景 - 带透明度
            text_bg = [text_x, text_y, text_x + text_width, text_y + text_height]
            text_bg_color = tuple(list(color_rgb) + [180])  # 透明度为180/255
            draw.rectangle(text_bg, fill=text_bg_color)
            
            # 文本颜色 - 根据背景亮度选择
            text_color = (255, 255, 255, 255) if sum(color_rgb) < 380 else (0, 0, 0, 255)
            
            # 绘制文本
            if font:
                draw.text((text_x, text_y), text, fill=text_color, font=font)
            else:
                draw.text((text_x, text_y), text, fill=text_color)
        
        # 合并图层并转回RGB
        result = Image.alpha_composite(result_im, overlay).convert("RGB")
        return result


def print_help_info():
    """打印脚本详细的帮助信息"""
    help_text = """
SAR飞机检测模型测试工具
==========================

该工具用于测试使用D-FINE框架训练的SAR飞机检测模型，并生成带有检测结果的可视化输出。

支持的飞机类别:
  - A220
  - A320/321
  - A330
  - ARJ21
  - Boeing737
  - Boeing787
  - other

参数说明:
  -c, --config     [必需] 配置文件路径，通常为YAML格式
  -r, --resume     [必需] 权重文件路径，通常为.pth格式
  -i, --input      [必需] 输入图像路径
  -o, --output     [可选] 输出图像路径，默认为"result.jpg"
  -t, --threshold  [可选] 置信度阈值，默认为0.4
  -d, --device     [可选] 运行设备，如'cuda:0'或'cpu'，默认为'cuda:0'
  -h, --help       显示此帮助信息

使用示例:
  # 使用默认参数
  python test.py -c configs/dfine/custom/sar_dfine_s.yml -r output/sar_dfine_s/best_stg2.pth -i test_image.jpg

  # 自定义输出路径
  python test.py -c configs/dfine/custom/sar_dfine_s.yml -r output/sar_dfine_s/best_stg2.pth -i test_image.jpg -o custom_output.jpg

  # 使用CPU进行推理
  python test.py -c configs/dfine/custom/sar_dfine_s.yml -r output/sar_dfine_s/best_stg2.pth -i test_image.jpg -d cpu

  # 调整检测阈值
  python test.py -c configs/dfine/custom/sar_dfine_s.yml -r output/sar_dfine_s/best_stg2.pth -i test_image.jpg -t 0.6

输出格式:
  程序会在终端打印检测结果，并保存带有标注的图像。
  检测结果包括每个检测框的类别、置信度和边界框坐标。
  不同类别的飞机会用不同颜色的框标注，以便区分。
"""
    print(help_text)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="SAR飞机检测模型测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False  # 禁用默认的帮助信息
    )
    
    # 添加参数
    parser.add_argument("-c", "--config", type=str, 
                        help="[必需] 配置文件路径，通常为YAML格式")
    parser.add_argument("-r", "--resume", type=str, 
                        help="[必需] 权重文件路径，通常为.pth格式")
    parser.add_argument("-i", "--input", type=str, 
                        help="[必需] 输入图像路径")
    parser.add_argument("-o", "--output", type=str, default="result.jpg", 
                        help="[可选] 输出图像路径，默认为'result.jpg'")
    parser.add_argument("-t", "--threshold", type=float, default=0.4, 
                        help="[可选] 置信度阈值，默认为0.4")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", 
                        help="[可选] 运行设备，如'cuda:0'或'cpu'，默认为'cuda:0'")
    parser.add_argument("-h", "--help", action="store_true", 
                        help="显示帮助信息")
    
    args = parser.parse_args()
    
    # 如果请求帮助或未提供必要参数，显示帮助信息
    if args.help or (not args.config or not args.resume or not args.input):
        print_help_info()
        return
    
    # 输出版本信息
    print("\nSAR飞机检测模型测试工具 v1.0")
    print("=" * 40)
    
    # 初始化测试器
    tester = ModelTester(args.config, args.resume, args.device)
    
    # 处理图像并获取结果
    results = tester.process_image(args.input, args.output, args.threshold)
    
    # 打印检测结果
    if results:
        print("\n检测结果:")
        print("=" * 50)
        print(f"{'类别':<10} {'置信度':<10} {'边界框'}")
        print("-" * 50)
        for label, score, box in results:
            print(f"{label:<10} {score:.4f}     [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]")
        print("=" * 50)
        print(f"\n共检测到 {len(results)} 个目标\n")
    else:
        print("\n未检测到任何目标\n")


if __name__ == "__main__":
    main() 