import json
import os
import math
import glob
from collections import defaultdict

def find_best_epoch_params():
    """查找每个模型规格的最佳epoch参数并返回比较数据"""
    models = ['x', 'l', 'm', 's', 'n']
    best_params = {}
    all_metrics = defaultdict(list)  # 存储所有指标的值，用于确定最大最小值
    
    for model in models:
        early_stop_log = f'output/sar_dfine_{model}/early_stop_log.txt'
        train_log = f'output/sar_dfine_{model}/log.txt'
        
        # 检查文件是否存在
        if not os.path.exists(early_stop_log) or not os.path.exists(train_log):
            print(f"警告: 模型 {model} 的日志文件不存在，跳过")
            continue
        
        # 读取最佳epoch
        try:
            with open(early_stop_log, 'r') as f:
                early_stop_data = json.loads(f.readline().strip())
                best_epoch = early_stop_data.get('best_epoch')
                if best_epoch is None:
                    print(f"警告: 模型 {model} 的early_stop_log.txt中没有best_epoch字段，跳过")
                    continue
        except (json.JSONDecodeError, IOError) as e:
            print(f"错误: 无法解析模型 {model} 的early_stop_log.txt: {e}")
            continue
        
        # 从训练日志中查找对应epoch的参数
        best_epoch_params = None
        try:
            with open(train_log, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if log_entry.get('epoch') == best_epoch:
                            best_epoch_params = log_entry
                            break
                    except json.JSONDecodeError:
                        continue
        except IOError as e:
            print(f"错误: 无法读取模型 {model} 的log.txt: {e}")
            continue
        
        if best_epoch_params is None:
            print(f"警告: 在模型 {model} 的log.txt中找不到epoch {best_epoch} 的数据，跳过")
            continue
        
        # 存储最佳参数 - 将模型名称简化为单个字母
        best_params[model.upper()] = {
            "best_epoch": best_epoch,
            "params": best_epoch_params,
            "best_AP50_95": early_stop_data.get("best_AP50:95", 0)
        }
        
        # 收集所有指标的值，用于后续确定图表的最大最小值
        for key, value in best_epoch_params.items():
            if isinstance(value, (int, float)) and key != 'epoch' and key != 'n_parameters':
                all_metrics[key].append(value)
            elif key == 'test_coco_eval_bbox' and isinstance(value, list):
                for i, v in enumerate(value):
                    all_metrics[f"AP_{i}"].append(v)
    
    return best_params, all_metrics

def generate_comparison_html():
    """生成模型比较的HTML可视化页面"""
    best_params, all_metrics = find_best_epoch_params()
    
    if not best_params:
        print("错误: 没有找到任何有效的模型数据")
        return
    
    # 获取要比较的模型名称列表
    model_names = list(best_params.keys())
    # 模型名称已经是单个字母，不需要额外处理显示名称
    model_display_names = model_names
    
    # 定义颜色映射（红黄蓝绿紫）
    colors = ['#ff4d4f', '#ffc53d', '#1890ff', '#52c41a', '#722ed1']
    color_mapping = {model: colors[i % len(colors)] for i, model in enumerate(model_names)}
    
    # 创建AP比较数据
    ap_data = {}
    ap_metric_names = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large']
    ap_indices = [0, 1, 2, 4, 5, 6]  # 根据日志示例观察到的COCO评估指标索引
    
    for model_name in model_names:
        model_data = best_params[model_name]
        coco_metrics = model_data['params'].get('test_coco_eval_bbox', [0] * 12)
        
        if len(coco_metrics) < 7:  # 确保有足够的指标
            print(f"警告: 模型 {model_name} 的COCO评估指标不完整")
            continue
            
        ap_data[model_name] = {
            ap_metric_names[i]: coco_metrics[ap_indices[i]]
            for i in range(len(ap_metric_names))
        }
    
    # 获取损失指标数据
    loss_metrics = [
        'train_loss', 'train_loss_vfl', 'train_loss_bbox', 
        'train_loss_giou', 'train_loss_fgl'
    ]
    
    loss_data = {}
    for model_name in model_names:
        loss_data[model_name] = {
            metric: best_params[model_name]['params'].get(metric, 0)
            for metric in loss_metrics
        }
    
    # 辅助损失指标数据
    aux_loss_metrics = [
        'train_loss_vfl_aux_0', 'train_loss_bbox_aux_0', 
        'train_loss_giou_aux_0', 'train_loss_fgl_aux_0', 
        'train_loss_ddf_aux_0'
    ]
    
    aux_loss_data = {}
    for model_name in model_names:
        aux_loss_data[model_name] = {
            metric: best_params[model_name]['params'].get(metric, 0)
            for metric in aux_loss_metrics
        }
    
    # 其他参数数据
    other_metrics = [
        'train_lr', 'n_parameters', 'best_epoch'
    ]
    
    other_data = {}
    for model_name in model_names:
        other_data[model_name] = {
            'train_lr': best_params[model_name]['params'].get('train_lr', 0),
            'n_parameters': best_params[model_name]['params'].get('n_parameters', 0),
            'best_epoch': best_params[model_name]['best_epoch'],
            'best_AP50_95': best_params[model_name]['best_AP50_95']
        }
    
    # 确定所有指标的最大值和最小值，用于动态调整图表范围
    metric_ranges = {}
    for metric, values in all_metrics.items():
        if values:
            min_val = min(values)
            max_val = max(values)
            # 为了更好的可视化效果，稍微扩展范围
            range_extension = (max_val - min_val) * 0.1
            metric_ranges[metric] = {
                'min': max(0, min_val - range_extension),  # 不小于0
                'max': max_val + range_extension
            }
    
    # 为AP指标设置范围
    ap_ranges = {}
    for i, name in enumerate(ap_metric_names):
        metric_key = f"AP_{ap_indices[i]}"
        if metric_key in metric_ranges:
            ap_ranges[name] = metric_ranges[metric_key]
        else:
            ap_ranges[name] = {'min': 0, 'max': 1}  # 默认AP范围0-1
    
    # 生成HTML内容
    with open(r'tools\visualization\sar_dfine\comparison_template.html', 'r', encoding='utf-8') as f:
        template = f.read()
    
    # 准备JavaScript数据
    js_content = f"""
// 模型数据
const modelNames = {json.dumps(model_names)};
const modelDisplayNames = {json.dumps(model_display_names)};
const colorMapping = {json.dumps(color_mapping)};

// AP指标数据
const apData = {json.dumps(ap_data)};
const apRanges = {json.dumps(ap_ranges)};

// 损失指标数据
const lossData = {json.dumps(loss_data)};
const lossRanges = {json.dumps({k: metric_ranges.get(k, {'min': 0, 'max': 1}) for k in loss_metrics})};

// 辅助损失数据
const auxLossData = {json.dumps(aux_loss_data)};
const auxLossRanges = {json.dumps({k: metric_ranges.get(k, {'min': 0, 'max': 1}) for k in aux_loss_metrics})};

// 其他参数数据
const otherData = {json.dumps(other_data)};

// 格式化数字显示（小数保留4位有效数字，整数不显示小数点）
function formatNumber(num) {{
    if (Number.isInteger(num)) {{
        return num.toString();
    }} else {{
        return num.toFixed(4);
    }}
}}

// 初始化所有图表
function initCharts() {{
    // 初始化AP柱状图
    initAPBarCharts();
    
    // 初始化损失雷达图
    initLossRadarChart();
    
    // 初始化辅助损失分组柱状图
    initAuxLossBarChart();
    
    // 初始化综合对比图
    initComprehensiveComparisonChart();
    
    // 初始化参数表格
    initParameterTable();
}}

// 初始化AP柱状图
function initAPBarCharts() {{
    const apMetrics = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large'];
    const apTitles = ['AP', 'AP50', 'AP75', 'AP Small', 'AP Medium', 'AP Large'];
    
    // 为每个AP指标创建单独的图表
    apMetrics.forEach((metric, index) => {{
        const chartContainer = document.getElementById(`ap${{index + 1}}Chart`);
        const chart = echarts.init(chartContainer);
        
        const options = {{
            title: {{
                text: apTitles[index],
                left: 'center'
            }},
            tooltip: {{
                trigger: 'axis',
                formatter: function(params) {{
                    let result = params[0].name + '<br/>';
                    params.forEach(param => {{
                        result += param.seriesName + ': ' + formatNumber(param.value) + '<br/>';
                    }});
                    return result;
                }}
            }},
            xAxis: {{
                type: 'category',
                data: modelDisplayNames,
                axisLabel: {{
                    interval: 0,
                    // 移除旋转，使标签水平显示
                    fontWeight: 'bold'
                }}
            }},
            yAxis: {{
                type: 'value',
                min: apRanges[metric].min,
                max: apRanges[metric].max,
                axisLabel: {{
                    showMinLabel: false,
                    showMaxLabel: false
                }}
            }},
            series: [
                {{
                    name: apTitles[index],
                    type: 'bar',
                    data: modelNames.map(model => apData[model]?.[metric] || 0),
                    itemStyle: {{
                        color: params => colorMapping[modelNames[params.dataIndex]]
                    }},
                    label: {{
                        show: true,
                        position: 'top',
                        formatter: function(params) {{
                            return formatNumber(params.value);
                        }}
                    }}
                }}
            ],
            grid: {{
                containLabel: true,
                left: '3%',
                right: '3%',
                bottom: '15%'
            }}
        }};
        
        chart.setOption(options);
        window.addEventListener('resize', () => chart.resize());
    }});
}}

// 初始化损失雷达图 - 优化后的版本，动态调整每个指标的范围
function initLossRadarChart() {{
    const chart = echarts.init(document.getElementById('lossRadarChart'));
    
    const lossMetrics = ['train_loss', 'train_loss_vfl', 'train_loss_bbox', 'train_loss_giou', 'train_loss_fgl'];
    const lossNames = ['Total Loss', 'VFL Loss', 'Bbox Loss', 'GIoU Loss', 'FGL Loss'];
    
    // 为每个指标计算最小最大值
    const metricMinMax = lossMetrics.map(metric => {{
        // 收集所有模型在该指标上的值
        const values = modelNames.map(model => lossData[model]?.[metric] || 0);
        const min = Math.min(...values);
        const max = Math.max(...values);
        // 设置一定的边界以便更好地显示对比
        const range = max - min;
        const padding = range * 0.1;
        return {{
            min: Math.max(0, min - padding),
            max: max + padding
        }};
    }});
    
    // 构建雷达图的指标配置
    const indicator = lossMetrics.map((metric, index) => {{
        return {{
            name: lossNames[index],
            min: metricMinMax[index].min,
            max: metricMinMax[index].max
        }};
    }});
    
    const series = modelNames.map(model => {{
        return {{
            value: lossMetrics.map(metric => lossData[model]?.[metric] || 0),
            name: model,
            itemStyle: {{
                color: colorMapping[model]
            }}
        }};
    }});
    
    const options = {{
        title: {{
            text: '损失函数雷达图对比',
            left: 'center'
        }},
        tooltip: {{
            trigger: 'item',
            formatter: function(params) {{
                let result = params.name + '<br/>';
                lossMetrics.forEach((metric, index) => {{
                    result += lossNames[index] + ': ' + formatNumber(params.value[index]) + '<br/>';
                }});
                return result;
            }}
        }},
        legend: {{
            top: '25px',
            data: modelDisplayNames
        }},
        radar: {{
            indicator: indicator,
            center: ['50%', '55%'],
            radius: '70%',
            shape: 'circle',
            splitArea: {{
                areaStyle: {{
                    color: ['rgba(250,250,250,0.3)', 'rgba(235,235,235,0.3)']
                }}
            }}
        }},
        series: [
            {{
                type: 'radar',
                data: series,
                lineStyle: {{
                    width: 2
                }},
                emphasis: {{
                    lineStyle: {{
                        width: 4
                    }}
                }}
            }}
        ]
    }};
    
    chart.setOption(options);
    window.addEventListener('resize', () => chart.resize());
}}

// 初始化辅助损失分组柱状图 - 替代原来的散点图
function initAuxLossBarChart() {{
    const chart = echarts.init(document.getElementById('auxLossChart'));
    
    const auxMetrics = ['train_loss_vfl_aux_0', 'train_loss_bbox_aux_0', 'train_loss_giou_aux_0', 'train_loss_fgl_aux_0', 'train_loss_ddf_aux_0'];
    const auxNames = ['VFL Aux', 'Bbox Aux', 'GIoU Aux', 'FGL Aux', 'DDF Aux'];
    
    // 准备分组柱状图的系列数据
    const series = modelNames.map(model => {{
        return {{
            name: model,
            type: 'bar',
            data: auxMetrics.map(metric => auxLossData[model]?.[metric] || 0),
            itemStyle: {{
                color: colorMapping[model]
            }},
            label: {{
                show: false,
                position: 'top',
                formatter: params => formatNumber(params.value)
            }}
        }};
    }});
    
    // 计算每个指标的最大值和最小值
    const yAxisRange = {{
        min: 0,
        max: 0
    }};
    
    auxMetrics.forEach((metric, metricIndex) => {{
        const values = modelNames.map(model => auxLossData[model]?.[metric] || 0);
        const max = Math.max(...values);
        if (max > yAxisRange.max) yAxisRange.max = max;
    }});
    
    // 增加一点边界以便更好地显示
    yAxisRange.max = yAxisRange.max * 1.1;
    
    const options = {{
        title: {{
            text: '辅助损失函数对比',
            left: 'center'
        }},
        tooltip: {{
            trigger: 'axis',
            formatter: function(params) {{
                const metricIndex = params[0].dataIndex;
                const metricName = auxNames[metricIndex];
                let result = metricName + '<br/>';
                params.forEach(param => {{
                    result += param.seriesName + ': ' + formatNumber(param.value) + '<br/>';
                }});
                return result;
            }}
        }},
        legend: {{
            top: '25px',
            data: modelDisplayNames
        }},
        grid: {{
            left: '3%',
            right: '4%',
            bottom: '15%',
            containLabel: true
        }},
        xAxis: {{
            type: 'category',
            data: auxNames,
            axisLabel: {{
                interval: 0,
                rotate: 30
            }}
        }},
        yAxis: {{
            type: 'value',
            min: yAxisRange.min,
            max: yAxisRange.max,
            name: '损失值',
            axisLabel: {{
                showMinLabel: false,
                showMaxLabel: false
            }}
        }},
        series: series
    }};
    
    chart.setOption(options);
    window.addEventListener('resize', () => chart.resize());
}}

// 初始化综合对比图（平行坐标系）
function initComprehensiveComparisonChart() {{
    const chart = echarts.init(document.getElementById('comprehensiveChart'));
    
    // 准备平行坐标轴
    const dimensions = [
        'best_AP50_95', 'n_parameters', 'train_lr', 'best_epoch'
    ];
    
    const dimensionNames = [
        'AP50:95', '参数量', '学习率', '最佳Epoch'
    ];
    
    // 为平行坐标轴准备数据
    const data = modelNames.map(model => {{
        const modelData = otherData[model];
        return [
            modelData.best_AP50_95 || 0,
            modelData.n_parameters || 0,
            modelData.train_lr || 0,
            modelData.best_epoch || 0,
            model // 用于图例显示
        ];
    }});
    
    // 确定每个维度的最大最小值
    const dimensionRanges = dimensions.map((dim, index) => {{
        const values = data.map(d => d[index]);
        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min;
        // 为了更好的可视化效果，稍微扩展范围
        return {{
            min: Math.max(0, min - range * 0.1),
            max: max + range * 0.1
        }};
    }});
    
    const options = {{
        title: {{
            text: '综合参数对比',
            left: 'center'
        }},
        legend: {{
            top: '25px',
            data: modelDisplayNames
        }},
        tooltip: {{
            trigger: 'item'
        }},
        parallelAxis: dimensions.map((dim, index) => {{
            return {{
                dim: index,
                name: dimensionNames[index],
                min: dimensionRanges[index].min,
                max: dimensionRanges[index].max,
                nameLocation: 'end',
                nameGap: 15,
                axisLabel: {{
                    showMinLabel: false,
                    showMaxLabel: false
                }}
            }};
        }}),
        parallel: {{
            left: '5%',
            right: '13%',
            bottom: '10%',
            top: '20%',
            parallelAxisDefault: {{
                nameLocation: 'end',
                nameGap: 20
            }}
        }},
        series: [
            {{
                type: 'parallel',
                lineStyle: {{
                    width: 4
                }},
                data: data.map((d, index) => {{
                    return {{
                        value: d,
                        lineStyle: {{
                            color: colorMapping[modelNames[index]]
                        }}
                    }};
                }})
            }}
        ]
    }};
    
    chart.setOption(options);
    window.addEventListener('resize', () => chart.resize());
}}

// 初始化参数表格
function initParameterTable() {{
    const tableContainer = document.getElementById('parameterTable');
    
    // 创建表格标题行
    let tableHTML = '<table class="parameter-table"><thead><tr><th>参数</th>';
    
    // 添加模型名称作为列标题
    modelDisplayNames.forEach(name => {{
        tableHTML += `<th>${{name}}</th>`;
    }});
    
    tableHTML += '</tr></thead><tbody>';
    
    // 添加基本参数行
    const basicParams = [
        {{'key': 'best_epoch', 'name': '最佳Epoch'}},
        {{'key': 'best_AP50_95', 'name': 'AP50:95'}},
        {{'key': 'n_parameters', 'name': '参数量'}},
        {{'key': 'train_lr', 'name': '学习率'}}
    ];
    
    basicParams.forEach(param => {{
        tableHTML += `<tr><td>${{param.name}}</td>`;
        
        modelNames.forEach(model => {{
            const value = otherData[model][param.key] || 0;
            tableHTML += `<td style="color:${{colorMapping[model]}}">${{formatNumber(value)}}</td>`;
        }});
        
        tableHTML += '</tr>';
    }});
    
    // 添加AP指标行
    const apMetrics = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large'];
    const apNames = ['AP', 'AP50', 'AP75', 'AP Small', 'AP Medium', 'AP Large'];
    
    apMetrics.forEach((metric, index) => {{
        tableHTML += `<tr><td>${{apNames[index]}}</td>`;
        
        modelNames.forEach(model => {{
            const value = apData[model]?.[metric] || 0;
            tableHTML += `<td style="color:${{colorMapping[model]}}">${{formatNumber(value)}}</td>`;
        }});
        
        tableHTML += '</tr>';
    }});
    
    // 添加损失指标行
    const lossMetrics = ['train_loss', 'train_loss_vfl', 'train_loss_bbox', 'train_loss_giou', 'train_loss_fgl'];
    const lossNames = ['Total Loss', 'VFL Loss', 'Bbox Loss', 'GIoU Loss', 'FGL Loss'];
    
    lossMetrics.forEach((metric, index) => {{
        tableHTML += `<tr><td>${{lossNames[index]}}</td>`;
        
        modelNames.forEach(model => {{
            const value = lossData[model]?.[metric] || 0;
            tableHTML += `<td style="color:${{colorMapping[model]}}">${{formatNumber(value)}}</td>`;
        }});
        
        tableHTML += '</tr>';
    }});
    
    // 添加辅助损失指标行
    const auxLossMetrics = ['train_loss_vfl_aux_0', 'train_loss_bbox_aux_0', 'train_loss_giou_aux_0', 'train_loss_fgl_aux_0', 'train_loss_ddf_aux_0'];
    const auxLossNames = ['VFL Aux Loss', 'Bbox Aux Loss', 'GIoU Aux Loss', 'FGL Aux Loss', 'DDF Aux Loss'];
    
    auxLossMetrics.forEach((metric, index) => {{
        tableHTML += `<tr><td>${{auxLossNames[index]}}</td>`;
        
        modelNames.forEach(model => {{
            const value = auxLossData[model]?.[metric] || 0;
            tableHTML += `<td style="color:${{colorMapping[model]}}">${{formatNumber(value)}}</td>`;
        }});
        
        tableHTML += '</tr>';
    }});
    
    tableHTML += '</tbody></table>';
    
    // 插入表格到容器
    tableContainer.innerHTML = tableHTML;
}}

// 初始化综合评估文本
function initComprehensiveEvaluation() {{
    const container = document.getElementById('evaluationContainer');
    
    // 计算模型排名（基于AP50:95）
    const modelRankings = [...modelNames].sort((a, b) => {{
        return (otherData[b].best_AP50_95 || 0) - (otherData[a].best_AP50_95 || 0);
    }});
    
    // 计算参数量与性能的比例
    const efficiencyData = modelNames.map(model => {{
        const ap = otherData[model].best_AP50_95 || 0;
        const params = otherData[model].n_parameters || 1; // 避免除以0
        return {{
            model: model,
            efficiency: ap / (params / 1000000), // AP per million parameters
            ap: ap,
            params: params
        }};
    }}).sort((a, b) => b.efficiency - a.efficiency);
    
    // 生成评估文本
    let evaluationHTML = '<div class="evaluation-text">';
    
    // 模型排名评估
    evaluationHTML += '<h3>模型性能排名</h3><p>';
    modelRankings.forEach((model, index) => {{
        const ap = formatNumber(otherData[model].best_AP50_95 || 0);
        evaluationHTML += `<span style="color:${{colorMapping[model]}}">${{model}}</span>: ${{ap}}`;
        if (index < modelRankings.length - 1) {{
            evaluationHTML += ' > ';
        }}
    }});
    evaluationHTML += '</p>';
    
    // 参数效率评估
    evaluationHTML += '<h3>参数效率排名 (AP/百万参数)</h3><p>';
    efficiencyData.forEach((data, index) => {{
        const efficiency = formatNumber(data.efficiency);
        evaluationHTML += `<span style="color:${{colorMapping[data.model]}}">${{data.model}}</span>: ${{efficiency}}`;
        if (index < efficiencyData.length - 1) {{
            evaluationHTML += ' > ';
        }}
    }});
    evaluationHTML += '</p>';
    
    // 训练效率评估
    evaluationHTML += '<h3>训练效率分析</h3><p>';
    const sortedByEpoch = [...modelNames].sort((a, b) => {{
        return (otherData[a].best_epoch || 0) - (otherData[b].best_epoch || 0);
    }});
    
    sortedByEpoch.forEach((model, index) => {{
        const epoch = otherData[model].best_epoch || 0;
        evaluationHTML += `<span style="color:${{colorMapping[model]}}">${{model}}</span>: ${{epoch}}轮`;
        if (index < sortedByEpoch.length - 1) {{
            evaluationHTML += ' < ';
        }}
    }});
    evaluationHTML += '</p>';
    
    // 总体评估
    evaluationHTML += '<h3>综合评估结论</h3><p>';
    evaluationHTML += '基于以上对比数据，我们可以得出以下结论：</p><ul>';
    
    // 性能最佳模型
    const bestModel = modelRankings[0];
    evaluationHTML += `<li>性能最佳的模型是 <span style="color:${{colorMapping[bestModel]}}">${{bestModel}}</span>，其AP50:95达到了${{formatNumber(otherData[bestModel].best_AP50_95 || 0)}}。</li>`;
    
    // 参数效率最佳模型
    const mostEfficientModel = efficiencyData[0].model;
    evaluationHTML += `<li>参数效率最高的模型是 <span style="color:${{colorMapping[mostEfficientModel]}}">${{mostEfficientModel}}</span>，每百万参数可提供更高的AP值。</li>`;
    
    // 训练效率最佳模型
    const fastestModel = sortedByEpoch[0];
    evaluationHTML += `<li>训练收敛最快的模型是 <span style="color:${{colorMapping[fastestModel]}}">${{fastestModel}}</span>，仅需${{otherData[fastestModel].best_epoch || 0}}轮即可达到最佳性能。</li>`;
    
    evaluationHTML += '</ul></div>';
    
    // 插入评估文本
    container.innerHTML = evaluationHTML;
}}

// 文档加载完成后初始化所有图表
document.addEventListener('DOMContentLoaded', function() {{
    initCharts();
    initComprehensiveEvaluation();
}});
"""
    
    # 在</script>标签前插入生成的JavaScript代码
    html_content = template.replace('</script>', js_content + '\n</script>')
    
    # 保存更新后的HTML文件
    with open(r'output\model_comparison.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"比较分析页面已生成: output/model_comparison.html")

if __name__ == '__main__':
    generate_comparison_html() 