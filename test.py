"""
目标检测模型测试脚本
用于测试训练好的D-FINE模型并生成带有标注的检测结果

使用方法:
    python test.py -c CONFIG -r WEIGHTS -i INPUT [-o OUTPUT] [-t THRESHOLD] [-d DEVICE]

示例:
    # 使用GPU进行推理
    python test.py -c configs/dfine/custom/dfine_s.yml -r output/dfine_s/best_stg2.pth -i test_image.jpg -o result.jpg

    # 使用CPU进行推理
    python test.py -c configs/dfine/custom/dfine_s.yml -r output/dfine_s/best_stg2.pth -i test_image.jpg -d cpu

    # 调整检测阈值
    python test.py -c configs/dfine/custom/dfine_s.yml -r output/dfine_s/best_stg2.pth -i test_image.jpg -t 0.6
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont, ImageOps
import colorsys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.core import YAMLConfig

def generate_colors(n):
    """生成n个在色环上均匀分布的颜色
    
    Args:
        n (int): 需要生成的颜色数量
    
    Returns:
        list: 包含n个RGB颜色元组的列表
    """
    colors = []
    for i in range(n):
        # 使用HSV色彩空间，在色环上均匀分布
        hue = i / n
        # 设置饱和度和亮度为固定值，避免颜色太暗或太亮
        saturation = 0.8
        value = 0.9
        # 转换为RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # 转换为0-255范围
        rgb = tuple(int(x * 255) for x in rgb)
        colors.append(rgb)
    return colors

def _get_font(size):
    """获取指定大小的字体，按优先级尝试加载不同字体"""
    font_names = [
        "arial.ttf", "Arial.ttf", "DejaVuSans.ttf", 
        "simhei.ttf", "simsun.ttc", "NotoSansCJK-Regular.ttc",
        "NotoSansSC-Regular.otf", "NotoSansSC-Regular.ttf"
    ]
    
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except:
            continue
    
    # 如果找不到任何字体，返回默认字体
    return ImageFont.load_default()

def _calculate_font_size(image_size, text_length):
    """根据图像大小和文本长度计算合适的字体大小"""
    base_size = min(image_size) // 40  # 基础字体大小
    # 根据文本长度调整字体大小
    adjusted_size = base_size * (1 + 0.1 * (text_length - 10))
    return max(12, min(adjusted_size, 40))  # 限制字体大小在12-40之间

def _get_text_color(bg_color):
    """根据背景色计算文本颜色
    
    使用相对亮度公式计算背景色的亮度，并返回合适的文本颜色
    如果背景色较亮，返回黑色；如果背景色较暗，返回白色
    
    Args:
        bg_color (tuple): RGB颜色元组，如(255, 0, 0)
    
    Returns:
        tuple: 文本颜色RGB元组，如(0, 0, 0)或(255, 255, 255)
    """
    # 使用相对亮度公式：Y = 0.299R + 0.587G + 0.114B
    brightness = (0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]) / 255
    return (0, 0, 0) if brightness > 0.5 else (255, 255, 255)

class ModelTester:
    """目标检测模型测试器
    
    该类负责加载训练好的模型，处理图像，并生成带有检测结果的可视化输出。
    每个检测到的目标都会用不同颜色的边界框标注，并显示类别名称和置信度。
    
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
        
        # 加载数据集标注信息
        self.dataset_info = self._load_dataset_info()
        
        # 从标注文件中获取类别信息
        self.classes = self._get_classes_from_annotations()
        
        # 根据类别数量生成颜色
        self.class_colors = generate_colors(len(self.classes))
    
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
    
    def _load_dataset_info(self):
        """加载数据集的标注信息"""
        cfg = YAMLConfig(self.config_path)
        
        # 获取所有可能的数据集路径
        dataset_paths = []
        
        # 从配置文件中获取数据集路径
        if 'train_dataloader' in cfg.yaml_cfg:
            dataset_paths.append(cfg.yaml_cfg['train_dataloader']['dataset']['img_folder'])
        if 'val_dataloader' in cfg.yaml_cfg:
            dataset_paths.append(cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'])
        if 'test_dataloader' in cfg.yaml_cfg:
            dataset_paths.append(cfg.yaml_cfg['test_dataloader']['dataset']['img_folder'])
        
        # 获取标注文件路径
        ann_files = []
        if 'train_dataloader' in cfg.yaml_cfg:
            ann_files.append(cfg.yaml_cfg['train_dataloader']['dataset']['ann_file'])
        if 'val_dataloader' in cfg.yaml_cfg:
            ann_files.append(cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'])
        if 'test_dataloader' in cfg.yaml_cfg:
            ann_files.append(cfg.yaml_cfg['test_dataloader']['dataset']['ann_file'])
        
        # 加载所有标注文件
        annotations = {}
        for ann_file in ann_files:
            if os.path.exists(ann_file):
                print(f"加载标注文件: {ann_file}")
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                    # 创建图像文件名到标注的映射
                    for img in data['images']:
                        img_anns = []
                        for ann in data['annotations']:
                            if ann['image_id'] == img['id']:
                                img_anns.append({
                                    'category_id': ann['category_id'],
                                    'bbox': ann['bbox']  # [x, y, width, height]
                                })
                        if img_anns:
                            annotations[os.path.basename(img['file_name'])] = img_anns
        
        # 创建数据集信息字典
        dataset_info = {
            'img_folders': dataset_paths,
            'annotations': annotations
        }
        
        print(f"已加载 {len(annotations)} 张图片的标注信息")
        return dataset_info
    
    def _get_image_annotation(self, image_path):
        """获取图像的原始标注信息"""
        img_name = os.path.basename(image_path)
        
        # 检查图片是否在数据集中
        for folder in self.dataset_info['img_folders']:
            if os.path.exists(os.path.join(folder, img_name)):
                return self.dataset_info['annotations'].get(img_name, None)
        
        return None
    
    def _convert_coco_bbox_to_xyxy(self, bbox):
        """将COCO格式的bbox [x, y, w, h]转换为[x1, y1, x2, y2]格式"""
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    
    def _create_split_line(self, height, width=5, color=None):
        """创建分割线图像"""
        line = Image.new('RGB', (width, height), color or (128, 128, 128))
        # 创建反色效果
        if color is None:
            line = ImageOps.invert(line)
        return line
    
    def _get_classes_from_annotations(self):
        """从标注文件中获取类别信息"""
        cfg = YAMLConfig(self.config_path)
        
        # 获取所有标注文件
        ann_files = []
        if 'train_dataloader' in cfg.yaml_cfg:
            ann_files.append(cfg.yaml_cfg['train_dataloader']['dataset']['ann_file'])
        if 'val_dataloader' in cfg.yaml_cfg:
            ann_files.append(cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'])
        if 'test_dataloader' in cfg.yaml_cfg:
            ann_files.append(cfg.yaml_cfg['test_dataloader']['dataset']['ann_file'])
        
        # 从第一个存在的标注文件中读取类别信息
        for ann_file in ann_files:
            if os.path.exists(ann_file):
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                    if 'categories' in data:
                        # 按id排序类别
                        categories = sorted(data['categories'], key=lambda x: x['id'])
                        return [cat['name'] for cat in categories]
        
        # 如果没有找到类别信息，返回默认类别
        print("警告：未找到类别信息，使用默认类别")
        return ['A220', 'A320/321', 'A330', 'ARJ21', 'Boeing737', 'Boeing787', 'other']
    
    def process_image(self, image_path, output_path="result.jpg", conf_threshold=0.4):
        """处理单张图像并生成对比结果"""
        # 加载图像
        print(f"处理图像: {image_path}")
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        print(f"图像尺寸: {w}x{h}")
        
        # 获取原始标注
        annotations = self._get_image_annotation(image_path)
        
        # 运行模型推理
        orig_size = torch.tensor([[w, h]]).to(self.device)
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        print("执行模型推理...")
        with torch.no_grad():
            labels, boxes, scores = self.model(img_tensor, orig_size)
        
        # 创建两个图像副本用于可视化
        gt_image = image.copy()
        pred_image = image.copy()
        
        # 在gt_image上绘制原始标注
        if annotations:
            print("发现原始标注，正在可视化...")
            for ann in annotations:
                bbox = self._convert_coco_bbox_to_xyxy(ann['bbox'])
                category_id = ann['category_id']
                self._draw_box(gt_image, bbox, category_id, 1.0, is_gt=True)
        
        # 在pred_image上绘制预测结果
        self._visualize_detections(pred_image, labels[0], boxes[0], scores[0], conf_threshold)
        
        # 创建并排显示的图像（包含标签区域）
        label_height = max(40, h // 20)  # 自适应标签区域高度
        combined_width = w * 2 + 5  # 5像素用于分割线
        combined_height = h + label_height  # 增加标签区域高度
        combined_image = Image.new('RGB', (combined_width, combined_height))
        
        # 粘贴图像和分割线
        combined_image.paste(gt_image, (0, 0))
        split_line = self._create_split_line(h)
        combined_image.paste(split_line, (w, 0))
        combined_image.paste(pred_image, (w + 5, 0))
        
        # 添加标签
        draw = ImageDraw.Draw(combined_image)
        
        # 计算标签字体大小
        label_font_size = _calculate_font_size((w, h), 10)
        font = _get_font(label_font_size)
        
        # 计算标签文本位置
        gt_text = "Origin"
        pred_text = "Detect"
        
        # 计算文本大小
        gt_text_bbox = draw.textbbox((0, 0), gt_text, font=font)
        pred_text_bbox = draw.textbbox((0, 0), pred_text, font=font)
        
        gt_text_width = gt_text_bbox[2] - gt_text_bbox[0]
        pred_text_width = pred_text_bbox[2] - pred_text_bbox[0]
        
        # 计算标签位置（居中）
        gt_x = (w - gt_text_width) // 2
        pred_x = w + 5 + (w - pred_text_width) // 2
        label_y = h + (label_height - (gt_text_bbox[3] - gt_text_bbox[1])) // 2
        
        # 绘制标签背景
        padding = 8
        for text, x in [(gt_text, gt_x), (pred_text, pred_x)]:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            bg_bbox = [
                x - padding,
                label_y - padding,
                x + (text_bbox[2] - text_bbox[0]) + padding,
                label_y + (text_bbox[3] - text_bbox[1]) + padding
            ]
            draw.rectangle(bg_bbox, fill=(0, 0, 0))
        
        # 绘制标签文本（黑色背景配白色文字）
        draw.text((gt_x, label_y), gt_text, fill=(255, 255, 255), font=font)
        draw.text((pred_x, label_y), pred_text, fill=(255, 255, 255), font=font)
        
        # 保存结果
        combined_image.save(output_path)
        print(f"结果已保存至: {output_path}")
        
        # 返回检测结果列表
        results = []
        for i, score in enumerate(scores[0]):
            if score > conf_threshold:
                label_idx = labels[0][i].item()
                label_name = self.classes[label_idx] if 0 <= label_idx < len(self.classes) else f"cls:{label_idx}"
                box = boxes[0][i].tolist()
                results.append((label_name, score.item(), box))
        
        return results
    
    def _draw_box(self, image, bbox, category_id, score, is_gt=False):
        """在图像上绘制边界框"""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # 确保类别ID在有效范围内
        category_id = min(max(category_id, 0), len(self.classes) - 1)
        
        # 获取类别颜色和名称
        color = self.class_colors[category_id]
        label = self.classes[category_id]
        
        # 添加置信度到标签（如果不是ground truth）
        if not is_gt:
            label = f"{label} {score:.2f}"
        
        # 计算合适的字体大小（根据图像大小和框的大小）
        box_width = bbox[2] - bbox[0]
        box_height = bbox[3] - bbox[1]
        box_size = min(box_width, box_height)
        base_size = min(width, height) // 40
        font_size = max(base_size, min(box_size // 15, 40))  # 根据框的大小调整字体大小
        font = _get_font(font_size)
        
        # 绘制边界框
        line_width = max(2, min(box_width, box_height) // 50)
        draw.rectangle(bbox, outline=color, width=line_width)
        
        # 计算文本大小和位置
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 计算文本位置
        text_x = bbox[0]
        text_y = bbox[1] - text_height - 4
        if text_y < 0:
            text_y = bbox[3]
        
        # 绘制文本背景
        padding = 4
        bg_bbox = [
            text_x - padding,
            text_y - padding,
            text_x + text_width + padding,
            text_y + text_height + padding
        ]
        draw.rectangle(bg_bbox, fill=color)
        
        # 根据背景色选择文本颜色
        text_color = _get_text_color(color)
        
        # 绘制文本
        draw.text((text_x, text_y), label, fill=text_color, font=font)
    
    def _visualize_detections(self, image, labels, boxes, scores, conf_threshold=0.4):
        """可视化检测结果"""
        # 筛选高于阈值的检测框
        mask = scores > conf_threshold
        filtered_labels = labels[mask]
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        
        # 如果没有检测到物体，直接返回
        if len(filtered_labels) == 0:
            print("未检测到任何物体!")
            return
        
        # 绘制每个检测框
        for label, box, score in zip(filtered_labels, filtered_boxes, filtered_scores):
            self._draw_box(image, box.tolist(), label.item(), score.item())


def print_help_info():
    """打印脚本详细的帮助信息"""
    help_text = """
目标检测模型测试工具
==========================

该工具用于测试使用D-FINE框架训练的目标检测模型，并生成带有检测结果的可视化输出。

支持的目标类别:
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
  python test.py -c configs/dfine/custom/dfine_s.yml -r output/dfine_s/best_stg2.pth -i test_image.jpg

  # 自定义输出路径
  python test.py -c configs/dfine/custom/dfine_s.yml -r output/dfine_s/best_stg2.pth -i test_image.jpg -o custom_output.jpg

  # 使用CPU进行推理
  python test.py -c configs/dfine/custom/dfine_s.yml -r output/dfine_s/best_stg2.pth -i test_image.jpg -d cpu

  # 调整检测阈值
  python test.py -c configs/dfine/custom/dfine_s.yml -r output/dfine_s/best_stg2.pth -i test_image.jpg -t 0.6

输出格式:
  程序会在终端打印检测结果，并保存带有标注的图像。
  检测结果包括每个检测框的类别、置信度和边界框坐标。
  不同类别的目标会用不同颜色的框标注，以便区分。
"""
    print(help_text)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="目标检测模型测试工具",
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
    print("\n目标检测模型测试工具 v1.0")
    print("=" * 40)
    
    # 初始化测试器
    tester = ModelTester(args.config, args.resume, args.device)
    
    # 处理图像并获取结果
    result = tester.process_image(args.input, args.output, args.threshold)
    
    # 打印检测结果
    if result:
        print("\n检测结果:")
        print("=" * 50)
        print(f"{'类别':<10} {'置信度':<10} {'边界框'}")
        print("-" * 50)
        for label, score, box in result:
            print(f"{label:<10} {score:.4f}     [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]")
        print("=" * 50)
        print(f"\n共检测到 {len(result)} 个目标\n")
    else:
        print("\n未检测到任何目标\n")


if __name__ == "__main__":
    main() 