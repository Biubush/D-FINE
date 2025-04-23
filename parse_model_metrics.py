import os
import json
from pathlib import Path

def find_best_epoch(log_data):
    """从日志数据中找出最佳epoch（基于最高AP50:95）"""
    best_epoch = 0
    best_ap = 0.0
    
    for entry in log_data:
        if 'test_coco_eval_bbox' in entry:
            ap50_95 = entry['test_coco_eval_bbox'][0]  # 假设第一个元素是AP50:95
            if ap50_95 > best_ap:
                best_ap = ap50_95
                best_epoch = entry['epoch']
    return best_epoch, best_ap

def parse_log_file(log_path):
    """解析单个log.txt文件"""
    metrics = {}
    with open(log_path, 'r') as f:
        log_data = [json.loads(line) for line in f if line.strip()]
    
    if not log_data:
        return None
        
    best_epoch, best_ap = find_best_epoch(log_data)
    
    # 获取最佳epoch的完整数据
    best_entry = next((e for e in log_data if e.get('epoch') == best_epoch), None)
    if not best_entry:
        return None
    
    # 获取总epoch数
    total_epoch = max([entry['epoch'] for entry in log_data if 'epoch' in entry])
    
    return {
        'best_epoch': best_epoch,
        'total_epoch': total_epoch,
        'ap50_95': best_ap,
        'ap50': best_entry['test_coco_eval_bbox'][1],
        'ap75': best_entry['test_coco_eval_bbox'][2],
        'train_loss': best_entry.get('train_loss'),
        'lr': best_entry.get('train_lr'),
        'weight_path': str(
            (Path(log_path).parent / 'best_stg2.pth').resolve() 
            if (Path(log_path).parent / 'best_stg2.pth').exists()
            else (Path(log_path).parent / 'best_stg1.pth').resolve()
        )
    }

def generate_model_metrics(output_dir='output'):
    """生成模型指标JSON文件"""
    model_metrics = {}
    output_path = Path(output_dir)
    
    for model_dir in output_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        log_path = model_dir / 'log.txt'
        if not log_path.exists():
            continue
            
        metrics = parse_log_file(log_path)
        if metrics:
            model_name = model_dir.name
            model_metrics[model_name] = metrics
    
    # 写入JSON文件
    output_json = output_path / 'model_metrics.json'
    with open(output_json, 'w') as f:
        json.dump(model_metrics, f, indent=2)
    
    return str(output_json)

if __name__ == '__main__':
    output_file = generate_model_metrics()
    print(f"模型指标已保存至：{output_file}")
