#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D-FINE训练结果分析与可视化
"""

import os
import argparse
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import torch
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import yaml

# 设置中文支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像负号'-'显示为方块的问题

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='D-FINE训练结果分析与可视化')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='训练输出目录路径，例如：output/sar_dfine_s_2')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='结果保存目录路径，默认为输入目录下的summary子目录')
    parser.add_argument('--dpi', type=int, default=150,
                        help='图表DPI，影响输出图片清晰度')
    parser.add_argument('--show', action='store_true',
                        help='是否显示图表（而不仅是保存）')
    parser.add_argument('--figure-size', type=float, nargs=2, default=[10, 6],
                        help='图表尺寸，默认为(10, 6)')
    return parser.parse_args()

def read_log_file(log_path):
    """读取日志文件并解析为JSON格式的数据列表"""
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                data.append(entry)
            except json.JSONDecodeError:
                continue
    return data

def extract_metrics(log_data):
    """从日志数据中提取关键训练和评估指标"""
    epochs = []
    train_losses = []
    train_loss_components = {}
    test_metrics = {}
    lr_values = []
    
    for entry in log_data:
        epoch = entry.get('epoch')
        if epoch is None:
            continue
            
        epochs.append(epoch)
        
        # 提取学习率
        lr = entry.get('train_lr')
        if lr is not None:
            lr_values.append(lr)
            
        # 提取训练损失
        loss = entry.get('train_loss')
        if loss is not None:
            train_losses.append(loss)
            
        # 提取训练损失的各个组成部分
        for key, value in entry.items():
            if key.startswith('train_loss_') and key != 'train_loss':
                if key not in train_loss_components:
                    train_loss_components[key] = []
                train_loss_components[key].append(value)
                
        # 提取测试指标
        for key, value in entry.items():
            if key.startswith('test_'):
                if key not in test_metrics:
                    test_metrics[key] = []
                test_metrics[key].append(value)
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'train_loss_components': train_loss_components,
        'test_metrics': test_metrics,
        'lr_values': lr_values
    }

def load_eval_results(eval_dir):
    """加载评估结果文件"""
    eval_files = {}
    
    if not os.path.exists(eval_dir):
        return None
        
    # 查找所有评估结果文件
    for file_name in os.listdir(eval_dir):
        if file_name.endswith('.pth'):
            file_path = os.path.join(eval_dir, file_name)
            try:
                eval_data = torch.load(file_path, map_location='cpu')
                if file_name == 'latest.pth':
                    key = 'latest'
                else:
                    # 尝试从文件名提取epoch数
                    match = re.match(r'(\d+)\.pth', file_name)
                    if match:
                        key = int(match.group(1))
                    else:
                        key = file_name
                        
                eval_files[key] = eval_data
            except Exception as e:
                print(f"无法加载评估文件 {file_name}: {e}")
    
    return eval_files

def analyze_sample_images(sample_dir):
    """分析样本图像目录"""
    if not os.path.exists(sample_dir):
        return None
        
    # 计算样本数量和图像分辨率统计信息
    image_sizes = []
    image_count = 0
    
    for file_name in os.listdir(sample_dir):
        if file_name.endswith(('.jpg', '.png', '.jpeg', '.webp')):
            image_count += 1
            
            # 这里可以加入PIL读取图像尺寸的代码
            # 但为简化，这里仅返回图像数量
    
    return {
        'image_count': image_count
    }

def plot_training_loss(metrics, save_path, show=False, dpi=150, figsize=(10, 6)):
    """绘制训练损失曲线"""
    plt.figure(figsize=figsize)
    plt.plot(metrics['epochs'], metrics['train_losses'], '-o', markersize=3, label='总损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()

def plot_loss_components(metrics, save_path, show=False, dpi=150, figsize=(12, 8)):
    """绘制损失组成部分"""
    # 筛选主要损失组件（不包括辅助损失）
    main_components = {k: v for k, v in metrics['train_loss_components'].items() 
                       if not ('aux' in k or 'dn' in k or 'enc' in k or 'pre' in k)}
    
    plt.figure(figsize=figsize)
    for key, values in main_components.items():
        # 提取组件名称，去掉'train_loss_'前缀
        label = key.replace('train_loss_', '')
        plt.plot(metrics['epochs'], values, '-o', markersize=3, label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss值')
    plt.title('主要损失组件变化')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()

def plot_accuracy_metrics(metrics, save_path, show=False, dpi=150, figsize=(10, 6)):
    """绘制准确率相关指标"""
    plt.figure(figsize=figsize)
    
    # COCO评估指标
    for key, values in metrics['test_metrics'].items():
        if key == 'test_coco_eval_bbox':
            label_mapping = {
                0: 'AP50:95',
                1: 'AP50',
                2: 'AP75',
                3: 'AP_S',
                4: 'AP_M',
                5: 'AP_L'
            }
            
            # 转换为numpy数组以便操作
            values_array = np.array(values)
            
            # 绘制不同的AP指标
            for i, metric_name in label_mapping.items():
                if i < values_array.shape[1]:
                    plt.plot(metrics['epochs'], values_array[:, i], '-o', markersize=3, label=metric_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('AP值')
    plt.title('COCO评估指标')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()

def plot_learning_rate(metrics, save_path, show=False, dpi=150, figsize=(10, 6)):
    """绘制学习率变化曲线"""
    plt.figure(figsize=figsize)
    plt.plot(metrics['epochs'], metrics['lr_values'], '-o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('学习率')
    plt.title('学习率变化曲线')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')  # 对数坐标更适合显示学习率变化
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()

def plot_pr_curves(eval_results, save_path, epoch_key='latest', show=False, dpi=150, figsize=(12, 10)):
    """绘制PR曲线"""
    if eval_results is None or epoch_key not in eval_results:
        return
    
    eval_data = eval_results[epoch_key]
    
    # COCO评估结果通常包含不同IoU阈值的PR曲线
    # 这里我们关注AP50和AP75两个关键指标
    if hasattr(eval_data, 'prec') and hasattr(eval_data, 'rec'):
        precisions = eval_data.prec
        recalls = eval_data.rec
        
        # COCO格式的precision维度是[IoU_thresholds, recall_thresholds, class_ids, area_ranges, max_detections]
        # 通常关注的是AP50 (IoU=0.5) 和 AP75 (IoU=0.75)
        iou_thresholds = [0, 5]  # 索引0对应IoU=0.5，索引5对应IoU=0.75
        category_ids = range(precisions.shape[2])  # 所有类别
        area_range_idx = 0  # 通常是all area
        max_det_idx = 2  # 通常是100 detections
        
        plt.figure(figsize=figsize)
        
        for i, iou_idx in enumerate(iou_thresholds):
            plt.subplot(1, 2, i+1)
            
            # 计算平均PR曲线（所有类别的平均）
            avg_precision = np.zeros_like(precisions[iou_idx, :, 0, area_range_idx, max_det_idx])
            valid_classes = 0
            
            for cat_idx in category_ids:
                precision = precisions[iou_idx, :, cat_idx, area_range_idx, max_det_idx]
                recall = recalls[iou_idx, :, cat_idx, area_range_idx, max_det_idx]
                
                # 检查数据是否有效
                if not np.all(np.isnan(precision)):
                    plt.plot(recall, precision, alpha=0.3, linewidth=0.5)
                    avg_precision += precision
                    valid_classes += 1
            
            if valid_classes > 0:
                avg_precision /= valid_classes
                plt.plot(
                    recalls[iou_idx, :, 0, area_range_idx, max_det_idx],
                    avg_precision,
                    'r-', linewidth=2, label='平均PR曲线'
                )
                
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'PR曲线 (IoU={0.5 if i==0 else 0.75})')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlim([0, 1])
            plt.ylim([0, 1.05])
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        if show:
            plt.show()
        plt.close()

def plot_loss_heatmap(metrics, save_path, show=False, dpi=150, figsize=(14, 10)):
    """绘制损失热力图"""
    # 准备损失组件数据
    components = {}
    for key, values in metrics['train_loss_components'].items():
        # 简化组件名称
        short_key = key.replace('train_loss_', '')
        # 只选择主要损失组件和一些重要的辅助损失
        if not any(x in short_key for x in ['_dn_pre', '_enc_', 'aux_']):
            components[short_key] = values
    
    # 创建数据框
    df = pd.DataFrame(components)
    
    # 添加epoch列
    df['epoch'] = metrics['epochs']
    
    # 重塑数据为热力图格式
    df_melted = df.melt(id_vars=['epoch'], var_name='Component', value_name='Loss')
    
    # 创建Seaborn热力图
    plt.figure(figsize=figsize)
    pivot_table = df_melted.pivot(index='Component', columns='epoch', values='Loss')
    
    # 使用更友好的颜色映射和日志尺度
    heatmap = sns.heatmap(pivot_table, cmap='viridis', robust=True, 
                           cbar_kws={'label': 'Loss值'})
    
    plt.title('损失组件随Epoch变化热力图')
    plt.xlabel('Epoch')
    plt.ylabel('损失组件')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()

def plot_performance_overview(metrics, eval_results, save_path, show=False, dpi=150, figsize=(15, 10)):
    """绘制性能概览图，结合多个关键指标"""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. 训练损失 (左上角)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(metrics['epochs'], metrics['train_losses'], '-o', markersize=2, color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 学习率 (右上角)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(metrics['epochs'], metrics['lr_values'], '-o', markersize=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('学习率')
    ax2.set_title('学习率变化')
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. AP指标 (中上)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'test_coco_eval_bbox' in metrics['test_metrics']:
        values_array = np.array(metrics['test_metrics']['test_coco_eval_bbox'])
        label_mapping = {
            0: 'AP50:95',
            1: 'AP50',
            2: 'AP75'
        }
        colors = ['red', 'orange', 'purple']
        
        for i, (metric_idx, metric_name) in enumerate(label_mapping.items()):
            if metric_idx < values_array.shape[1]:
                ax3.plot(metrics['epochs'], values_array[:, metric_idx], 
                         '-o', markersize=2, label=metric_name, color=colors[i])
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AP值')
    ax3.set_title('COCO评估指标')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    # 4. 主要损失组件 (左下)
    ax4 = fig.add_subplot(gs[1, 0])
    main_components = {
        'vfl': metrics['train_loss_components'].get('train_loss_vfl', []),
        'bbox': metrics['train_loss_components'].get('train_loss_bbox', []),
        'giou': metrics['train_loss_components'].get('train_loss_giou', []),
        'fgl': metrics['train_loss_components'].get('train_loss_fgl', [])
    }
    
    for name, values in main_components.items():
        if values:  # 确保有数据
            ax4.plot(metrics['epochs'], values, '-', label=name)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss值')
    ax4.set_title('主要损失组件')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend()
    
    # 5. 小目标、中目标、大目标AP (右下)
    ax5 = fig.add_subplot(gs[1, 1:])
    if 'test_coco_eval_bbox' in metrics['test_metrics']:
        values_array = np.array(metrics['test_metrics']['test_coco_eval_bbox'])
        size_mapping = {
            3: 'AP_S (小目标)',
            4: 'AP_M (中目标)',
            5: 'AP_L (大目标)'
        }
        colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
        
        for i, (metric_idx, metric_name) in enumerate(size_mapping.items()):
            if metric_idx < values_array.shape[1]:
                ax5.plot(metrics['epochs'], values_array[:, metric_idx], 
                         '-o', markersize=2, label=metric_name, color=colors[i])
    
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('AP值')
    ax5.set_title('不同尺寸目标的AP')
    ax5.grid(True, linestyle='--', alpha=0.7)
    ax5.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()

def create_model_card(metrics, eval_results, train_samples, val_samples, save_path, dpi=150):
    """创建模型信息卡片"""
    # 获取最终性能指标
    final_metrics = {}
    if metrics['test_metrics'] and 'test_coco_eval_bbox' in metrics['test_metrics']:
        values = metrics['test_metrics']['test_coco_eval_bbox'][-1]  # 最后一个epoch的值
        metric_names = ['AP50:95', 'AP50', 'AP75', 'AP_S', 'AP_M', 'AP_L']
        for i, name in enumerate(metric_names):
            if i < len(values):
                final_metrics[name] = values[i]
    
    # 创建画布
    fig = plt.figure(figsize=(12, 8))
    
    # 设置标题
    plt.suptitle('D-FINE 模型性能报告', fontsize=18, fontweight='bold', y=0.98)
    
    # 防止子图重叠
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9, bottom=0.05, left=0.05, right=0.95)
    
    # 定义颜色
    info_color = '#e6f2ff'  # 浅蓝色
    metric_color = '#e6ffe6'  # 浅绿色
    
    # 基本信息部分 - 顶部区域
    ax_info = plt.subplot2grid((6, 2), (0, 0), colspan=2, rowspan=2)
    ax_info.axis('off')
    
    # 绘制基本信息背景
    ax_info.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=info_color, edgecolor='gray', 
                                   alpha=0.5, transform=ax_info.transAxes))
    
    # 基本训练信息 - 分两列显示
    info_text = [
        f"总训练轮次: {max(metrics['epochs']) if metrics['epochs'] else 'N/A'}",
        f"训练样本数: {train_samples['image_count'] if train_samples else 'N/A'}",
        f"验证样本数: {val_samples['image_count'] if val_samples else 'N/A'}",
        f"最终学习率: {metrics['lr_values'][-1] if metrics['lr_values'] else 'N/A':.6f}",
        f"最终损失值: {metrics['train_losses'][-1] if metrics['train_losses'] else 'N/A':.4f}"
    ]
    
    # 左列信息
    for i, text in enumerate(info_text[:3]):
        ax_info.text(0.05, 0.8 - i*0.25, text, fontsize=12, transform=ax_info.transAxes)
    
    # 右列信息
    for i, text in enumerate(info_text[3:]):
        ax_info.text(0.55, 0.8 - i*0.25, text, fontsize=12, transform=ax_info.transAxes)
    
    # 性能指标部分 - 底部区域
    ax_metrics = plt.subplot2grid((6, 2), (2, 0), colspan=2, rowspan=4)
    ax_metrics.axis('off')
    
    # 绘制性能指标背景
    ax_metrics.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=metric_color, edgecolor='gray', 
                                      alpha=0.5, transform=ax_metrics.transAxes))
    
    # 性能指标标题
    ax_metrics.text(0.5, 0.95, '关键性能指标', fontsize=14, fontweight='bold', 
                   ha='center', transform=ax_metrics.transAxes)
    
    # 性能指标条形图
    if final_metrics:
        # 定义指标颜色
        colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#8F00FF', '#FF6D01']
        
        # 绘制指标条形图 - 一行一个指标
        metrics_data = list(final_metrics.items())
        
        for i, ((name, value), color) in enumerate(zip(metrics_data, colors)):
            y_pos = 0.8 - i * 0.13  # 每个指标的垂直位置
            
            # 绘制指标名称
            ax_metrics.text(0.05, y_pos, f"{name}:", fontsize=12, ha='left', va='center')
            
            # 绘制条形图
            bar_width = value * 0.55  # 根据值计算条形图宽度
            ax_metrics.add_patch(plt.Rectangle((0.2, y_pos-0.04), bar_width, 0.08, 
                                             facecolor=color, alpha=0.8))
            
            # 绘制指标值
            ax_metrics.text(0.2 + bar_width + 0.02, y_pos, f"{value:.4f}", 
                           fontsize=12, va='center')
    
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)

def load_dataset_config(input_dir):
    """加载数据集配置文件并获取训练和验证样本数量"""
    # 尝试查找配置文件
    config_paths = [
        input_dir / 'config.yml',  # 训练目录可能包含配置文件的副本
        Path('configs/dataset/sar_airplane_detection.yml')  # 默认配置文件路径
    ]
    
    config_file = None
    for path in config_paths:
        if path.exists():
            config_file = path
            break
    
    if config_file is None:
        print("警告：未找到数据集配置文件，无法获取准确的样本数量")
        return {'train_samples': 0, 'val_samples': 0}
    
    # 加载配置文件
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取数据集文件路径
    train_ann_file = config.get('train_dataloader', {}).get('dataset', {}).get('ann_file', '')
    val_ann_file = config.get('val_dataloader', {}).get('dataset', {}).get('ann_file', '')
    
    train_samples = 0
    val_samples = 0
    
    # 从COCO json文件中获取样本数量
    if train_ann_file and Path(train_ann_file).exists():
        with open(train_ann_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            train_samples = len(train_data.get('images', []))
    else:
        print(f"警告：未找到训练集标注文件: {train_ann_file}")
    
    if val_ann_file and Path(val_ann_file).exists():
        with open(val_ann_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
            val_samples = len(val_data.get('images', []))
    else:
        print(f"警告：未找到验证集标注文件: {val_ann_file}")
    
    return {'train_samples': train_samples, 'val_samples': val_samples}

def main():
    """主函数"""
    args = parse_arguments()
    
    input_dir = Path(args.input_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / 'summary'
    
    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True)
    
    # 读取日志文件
    log_path = input_dir / 'log.txt'
    if not log_path.exists():
        print(f"错误：日志文件不存在 - {log_path}")
        return
    
    log_data = read_log_file(log_path)
    metrics = extract_metrics(log_data)
    
    # 加载评估结果
    eval_dir = input_dir / 'eval'
    eval_results = load_eval_results(eval_dir)
    
    # 获取数据集样本数量
    dataset_info = load_dataset_config(input_dir)
    
    # 分析样本图像（作为备选方案）
    train_samples = analyze_sample_images(input_dir / 'train_samples')
    val_samples = analyze_sample_images(input_dir / 'val_samples')
    
    # 使用COCO文件中的样本数量（如果可用），否则使用样本目录的数量
    train_count = dataset_info['train_samples'] or (train_samples['image_count'] if train_samples else 0)
    val_count = dataset_info['val_samples'] or (val_samples['image_count'] if val_samples else 0)
    
    # 更新样本信息
    train_samples = {'image_count': train_count} if train_count > 0 else train_samples
    val_samples = {'image_count': val_count} if val_count > 0 else val_samples
    
    # 生成图表
    plot_training_loss(metrics, output_dir / 'training_loss.png', args.show, args.dpi, tuple(args.figure_size))
    plot_loss_components(metrics, output_dir / 'loss_components.png', args.show, args.dpi, tuple(args.figure_size))
    plot_accuracy_metrics(metrics, output_dir / 'accuracy_metrics.png', args.show, args.dpi, tuple(args.figure_size))
    plot_learning_rate(metrics, output_dir / 'learning_rate.png', args.show, args.dpi, tuple(args.figure_size))
    plot_pr_curves(eval_results, output_dir / 'pr_curves.png', 'latest', args.show, args.dpi, tuple(args.figure_size))
    plot_loss_heatmap(metrics, output_dir / 'loss_heatmap.png', args.show, args.dpi, (14, 10))
    plot_performance_overview(metrics, eval_results, output_dir / 'performance_overview.png', args.show, args.dpi, (15, 10))
    create_model_card(metrics, eval_results, train_samples, val_samples, output_dir / 'model_card.png', args.dpi)
    
    print(f"分析完成！结果已保存至 {output_dir}")

if __name__ == "__main__":
    main()