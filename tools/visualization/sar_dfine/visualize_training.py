import json
import os

def parse_log_file(log_path):
    """解析训练日志文件"""
    data = {
        'epochs': [],
        'train_lr': [],
        'train_loss': [],
        'train_loss_vfl': [],
        'train_loss_bbox': [],
        'train_loss_giou': [],
        'train_loss_fgl': [],
        'train_loss_vfl_aux_0': [],
        'train_loss_bbox_aux_0': [],
        'train_loss_giou_aux_0': [],
        'train_loss_fgl_aux_0': [],
        'train_loss_ddf_aux_0': [],
        'test_coco_eval_bbox': []
    }
    
    with open(log_path, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                data['epochs'].append(log_entry['epoch'])
                data['train_lr'].append(log_entry['train_lr'])
                data['train_loss'].append(log_entry['train_loss'])
                data['train_loss_vfl'].append(log_entry['train_loss_vfl'])
                data['train_loss_bbox'].append(log_entry['train_loss_bbox'])
                data['train_loss_giou'].append(log_entry['train_loss_giou'])
                data['train_loss_fgl'].append(log_entry['train_loss_fgl'])
                data['train_loss_vfl_aux_0'].append(log_entry['train_loss_vfl_aux_0'])
                data['train_loss_bbox_aux_0'].append(log_entry['train_loss_bbox_aux_0'])
                data['train_loss_giou_aux_0'].append(log_entry['train_loss_giou_aux_0'])
                data['train_loss_fgl_aux_0'].append(log_entry['train_loss_fgl_aux_0'])
                data['train_loss_ddf_aux_0'].append(log_entry['train_loss_ddf_aux_0'])
                data['test_coco_eval_bbox'].append(log_entry['test_coco_eval_bbox'])
            except:
                continue
    return data

def generate_visualization_data():
    """生成所有模型的可视化数据"""
    models = ['x', 'l', 'm', 's', 'n']
    visualization_data = {}
    
    for model in models:
        log_path = f'output/sar_dfine_{model}/log.txt'
        if os.path.exists(log_path):
            data = parse_log_file(log_path)
            visualization_data[f'sar_dfine_{model}'] = data
            
    # 将数据写入JavaScript文件
    js_content = f"""
// 训练数据
const trainingData = {json.dumps(visualization_data, indent=2)};

// 更新图表函数
function updateCharts() {{
    const modelName = document.getElementById('modelSelect').value;
    const data = trainingData[modelName];
    
    if (!data) return;
    
    // 更新损失曲线
    charts.loss.setOption({{
        xAxis: {{
            data: data.epochs
        }},
        series: [
            {{
                name: 'Total Loss',
                data: data.train_loss
            }},
            {{
                name: 'VFL Loss',
                data: data.train_loss_vfl
            }},
            {{
                name: 'Bbox Loss',
                data: data.train_loss_bbox
            }},
            {{
                name: 'GIoU Loss',
                data: data.train_loss_giou
            }},
            {{
                name: 'FGL Loss',
                data: data.train_loss_fgl
            }}
        ]
    }});
    
    // 更新学习率曲线
    charts.lr.setOption({{
        xAxis: {{
            data: data.epochs
        }},
        series: [
            {{
                data: data.train_lr
            }}
        ]
    }});
    
    // 更新辅助损失曲线
    charts.auxLoss.setOption({{
        xAxis: {{
            data: data.epochs
        }},
        series: [
            {{
                name: 'VFL Aux',
                data: data.train_loss_vfl_aux_0
            }},
            {{
                name: 'Bbox Aux',
                data: data.train_loss_bbox_aux_0
            }},
            {{
                name: 'GIoU Aux',
                data: data.train_loss_giou_aux_0
            }},
            {{
                name: 'FGL Aux',
                data: data.train_loss_fgl_aux_0
            }},
            {{
                name: 'DDF Aux',
                data: data.train_loss_ddf_aux_0
            }}
        ]
    }});
    
    // 更新评估指标曲线
    charts.eval.setOption({{
        xAxis: {{
            data: data.epochs
        }},
        series: [
            {{
                name: 'AP',
                data: data.test_coco_eval_bbox.map(d => d[0])
            }},
            {{
                name: 'AP50',
                data: data.test_coco_eval_bbox.map(d => d[1])
            }},
            {{
                name: 'AP75',
                data: data.test_coco_eval_bbox.map(d => d[2])
            }},
            {{
                name: 'APS',
                data: data.test_coco_eval_bbox.map(d => d[4])
            }},
            {{
                name: 'APM',
                data: data.test_coco_eval_bbox.map(d => d[5])
            }},
            {{
                name: 'APL',
                data: data.test_coco_eval_bbox.map(d => d[6])
            }}
        ]
    }});
}}

// 初始加载
document.addEventListener('DOMContentLoaded', updateCharts);
"""
    
    # 将生成的JavaScript代码添加到HTML模板中
    with open(r'tools\visualization\sar_dfine\visualization_template.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 在</script>标签前插入生成的JavaScript代码
    html_content = html_content.replace('</script>', js_content + '\n</script>')
    
    # 保存更新后的HTML文件
    with open(r'output\training_visualization.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    generate_visualization_data()