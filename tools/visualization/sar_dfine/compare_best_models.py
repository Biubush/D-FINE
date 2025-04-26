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
    
    # 读取YOLO模型指标缓存
    try:
        with open('tools/visualization/sar_dfine/metrics_cache.json', 'r', encoding='utf-8') as f:
            metrics_cache = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"错误: 无法读取metrics_cache.json: {e}")
        metrics_cache = {
            "模型": [],
            "mAP@0.5:0.95": [],
            "mAP@0.5": [],
            "mAP@0.75": []
        }
    
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
    
    # 读取YOLO模型指标缓存
    try:
        with open('tools/visualization/sar_dfine/metrics_cache.json', 'r', encoding='utf-8') as f:
            metrics_cache = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"错误: 无法读取metrics_cache.json: {e}")
        metrics_cache = {
            "模型": [],
            "mAP@0.5:0.95": [],
            "mAP@0.5": [],
            "mAP@0.75": []
        }
    
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

// YOLO模型数据
const yoloModels = {json.dumps(metrics_cache['模型'])};
const yoloMap = {json.dumps(metrics_cache['mAP@0.5:0.95'])};
const yoloMap50 = {json.dumps(metrics_cache['mAP@0.5'])};
const yoloMap75 = {json.dumps(metrics_cache['mAP@0.75'])};

// SAR-D-FINE模型数据
const sarModels = {json.dumps(model_names)};
const sarMap = {json.dumps([ap_data[model]['AP'] for model in model_names])};
const sarMap50 = {json.dumps([ap_data[model]['AP50'] for model in model_names])};
const sarMap75 = {json.dumps([ap_data[model]['AP75'] for model in model_names])};

// 合并数据为单个系列
const combinedData = {{
    map: [...yoloMap.map((value, index) => ({{
        value: value,
        name: yoloModels[index],
        itemStyle: {{ color: '#D3D3D3' }}  // 浅灰色
    }})), ...sarMap.map((value, index) => ({{
        value: value,
        name: 'D-FINE-' + sarModels[index].toUpperCase(),
        itemStyle: {{ color: colorMapping[sarModels[index]] }}
    }}))],
    map50: [...yoloMap50.map((value, index) => ({{
        value: value,
        name: yoloModels[index],
        itemStyle: {{ color: '#D3D3D3' }}  // 浅灰色
    }})), ...sarMap50.map((value, index) => ({{
        value: value,
        name: 'D-FINE-' + sarModels[index].toUpperCase(),
        itemStyle: {{ color: colorMapping[sarModels[index]] }}
    }}))],
    map75: [...yoloMap75.map((value, index) => ({{
        value: value,
        name: yoloModels[index],
        itemStyle: {{ color: '#D3D3D3' }}  // 浅灰色
    }})), ...sarMap75.map((value, index) => ({{
        value: value,
        name: 'D-FINE-' + sarModels[index].toUpperCase(),
        itemStyle: {{ color: colorMapping[sarModels[index]] }}
    }}))]
}};

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
function formatNumber(num, isLearningRate = false) {{
    if (isLearningRate) {{
        // 对学习率使用科学计数法，保留2位小数
        return num.toExponential(2);
    }}
    if (Number.isInteger(num)) {{
        return num.toString();
    }} else {{
        return num.toFixed(4);
    }}
}}

// 计算坐标轴范围的辅助函数
function calculateAxisRange(values, padding = 0.1) {{
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    
    // 如果数值差异很小（小于1%），则扩大显示范围以突出差异
    if (range / max < 0.01) {{
        const mid = (max + min) / 2;
        const halfRange = Math.max(range, max * 0.01) / 2;
        return {{
            min: mid - halfRange * (1 + padding),
            max: mid + halfRange * (1 + padding)
        }};
    }}
    
    // 否则使用常规的范围计算
    return {{
        min: min - range * padding,
        max: max + range * padding
    }};
}}

// 初始化所有图表
function initCharts() {{
    // 初始化AP柱状图
    initAPBarCharts();
    
    // 初始化YOLO对比图表
    initYOLOComparisonCharts();
    
    // 初始化损失雷达图
    initLossRadarChart();
    
    // 初始化辅助损失雷达图
    initAuxLossRadarChart();
    
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

// 初始化YOLO对比图表
function initYOLOComparisonCharts() {{
    // 计算各指标的范围
    const mapRange = calculateAxisRange([...yoloMap, ...sarMap]);
    const map50Range = calculateAxisRange([...yoloMap50, ...sarMap50]);
    const map75Range = calculateAxisRange([...yoloMap75, ...sarMap75]);

    // 初始化mAP@0.5:0.95图表
    const mapChart = echarts.init(document.getElementById('yoloMapChart'));
    mapChart.setOption({{
        title: {{
            text: 'mAP@0.5:0.95 对比',
            left: 'center'
        }},
        tooltip: {{
            trigger: 'axis',
            axisPointer: {{
                type: 'line',
                axis: 'y',
                animation: true,
                lineStyle: {{
                    color: '#666',
                    type: 'dashed'
                }}
            }},
            formatter: function(params) {{
                return params[0].name + '<br/>' + 
                       'mAP@0.5:0.95: ' + formatNumber(params[0].value);
            }},
            position: function (pos, params, el, elRect, size) {{
                const obj = {{top: 10}};
                obj[['left', 'right'][+(pos[0] < size.viewSize[0] / 2)]] = 30;
                return obj;
            }}
        }},
        grid: {{
            left: '3%',
            right: '4%',
            bottom: '15%',
            containLabel: true
        }},
        xAxis: {{
            type: 'category',
            data: combinedData.map.map(item => item.name),
            axisLabel: {{
                interval: 0,
                rotate: 30
            }}
        }},
        yAxis: {{
            type: 'value',
            min: mapRange.min,
            max: mapRange.max,
            axisLabel: {{
                showMinLabel: false,
                showMaxLabel: false,
                formatter: value => formatNumber(value)
            }},
            splitLine: {{
                show: true,
                lineStyle: {{
                    type: 'dashed',
                    opacity: 0.3
                }}
            }}
        }},
        series: [{{
            type: 'bar',
            data: combinedData.map,
            label: {{
                show: true,
                position: 'top',
                formatter: function(params) {{
                    return formatNumber(params.value);
                }}
            }}
        }}]
    }});

    // 初始化mAP@0.5图表
    const map50Chart = echarts.init(document.getElementById('yoloMap50Chart'));
    map50Chart.setOption({{
        title: {{
            text: 'mAP@0.5 对比',
            left: 'center'
        }},
        tooltip: {{
            trigger: 'axis',
            axisPointer: {{
                type: 'line',
                axis: 'y',
                animation: true,
                lineStyle: {{
                    color: '#666',
                    type: 'dashed'
                }}
            }},
            formatter: function(params) {{
                return params[0].name + '<br/>' + 
                       'mAP@0.5: ' + formatNumber(params[0].value);
            }},
            position: function (pos, params, el, elRect, size) {{
                const obj = {{top: 10}};
                obj[['left', 'right'][+(pos[0] < size.viewSize[0] / 2)]] = 30;
                return obj;
            }}
        }},
        grid: {{
            left: '3%',
            right: '4%',
            bottom: '15%',
            containLabel: true
        }},
        xAxis: {{
            type: 'category',
            data: combinedData.map50.map(item => item.name),
            axisLabel: {{
                interval: 0,
                rotate: 30
            }}
        }},
        yAxis: {{
            type: 'value',
            min: map50Range.min,
            max: map50Range.max,
            axisLabel: {{
                showMinLabel: false,
                showMaxLabel: false,
                formatter: value => formatNumber(value)
            }},
            splitLine: {{
                show: true,
                lineStyle: {{
                    type: 'dashed',
                    opacity: 0.3
                }}
            }}
        }},
        series: [{{
            type: 'bar',
            data: combinedData.map50,
            label: {{
                show: true,
                position: 'top',
                formatter: function(params) {{
                    return formatNumber(params.value);
                }}
            }}
        }}]
    }});

    // 初始化mAP@0.75图表
    const map75Chart = echarts.init(document.getElementById('yoloMap75Chart'));
    map75Chart.setOption({{
        title: {{
            text: 'mAP@0.75 对比',
            left: 'center'
        }},
        tooltip: {{
            trigger: 'axis',
            axisPointer: {{
                type: 'line',
                axis: 'y',
                animation: true,
                lineStyle: {{
                    color: '#666',
                    type: 'dashed'
                }}
            }},
            formatter: function(params) {{
                return params[0].name + '<br/>' + 
                       'mAP@0.75: ' + formatNumber(params[0].value);
            }},
            position: function (pos, params, el, elRect, size) {{
                const obj = {{top: 10}};
                obj[['left', 'right'][+(pos[0] < size.viewSize[0] / 2)]] = 30;
                return obj;
            }}
        }},
        grid: {{
            left: '3%',
            right: '4%',
            bottom: '15%',
            containLabel: true
        }},
        xAxis: {{
            type: 'category',
            data: combinedData.map75.map(item => item.name),
            axisLabel: {{
                interval: 0,
                rotate: 30
            }}
        }},
        yAxis: {{
            type: 'value',
            min: map75Range.min,
            max: map75Range.max,
            axisLabel: {{
                showMinLabel: false,
                showMaxLabel: false,
                formatter: value => formatNumber(value)
            }},
            splitLine: {{
                show: true,
                lineStyle: {{
                    type: 'dashed',
                    opacity: 0.3
                }}
            }}
        }},
        series: [{{
            type: 'bar',
            data: combinedData.map75,
            label: {{
                show: true,
                position: 'top',
                formatter: function(params) {{
                    return formatNumber(params.value);
                }}
            }}
        }}]
    }});

    // 添加窗口大小改变时的重绘
    window.addEventListener('resize', function() {{
        mapChart.resize();
        map50Chart.resize();
        map75Chart.resize();
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

// 初始化辅助损失雷达图
function initAuxLossRadarChart() {{
    const chart = echarts.init(document.getElementById('auxLossChart'));
    
    const auxMetrics = ['train_loss_vfl_aux_0', 'train_loss_bbox_aux_0', 'train_loss_giou_aux_0', 'train_loss_fgl_aux_0', 'train_loss_ddf_aux_0'];
    const auxNames = ['VFL Aux', 'Bbox Aux', 'GIoU Aux', 'FGL Aux', 'DDF Aux'];
    
    // 为每个指标计算最小最大值
    const metricMinMax = auxMetrics.map(metric => {{
        // 收集所有模型在该指标上的值
        const values = modelNames.map(model => auxLossData[model]?.[metric] || 0);
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
    const indicator = auxMetrics.map((metric, index) => {{
        return {{
            name: auxNames[index],
            min: metricMinMax[index].min,
            max: metricMinMax[index].max
        }};
    }});
    
    const series = modelNames.map(model => {{
        return {{
            value: auxMetrics.map(metric => auxLossData[model]?.[metric] || 0),
            name: model,
            itemStyle: {{
                color: colorMapping[model]
            }}
        }};
    }});
    
    const options = {{
        title: {{
            text: '辅助损失函数雷达图对比',
            left: 'center'
        }},
        tooltip: {{
            trigger: 'item',
            formatter: function(params) {{
                let result = params.name + '<br/>';
                auxMetrics.forEach((metric, index) => {{
                    result += auxNames[index] + ': ' + formatNumber(params.value[index]) + '<br/>';
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

// 初始化综合参数对比图
function initComprehensiveComparisonChart() {{
    const chart = echarts.init(document.getElementById('comprehensiveChart'));
    
    // 定义评价指标
    const metrics = [
        {{
            name: '性能指标',
            key: 'best_AP50_95',
            weight: 0.35
        }},
        {{
            name: '参数效率',
            key: 'efficiency',
            weight: 0.25
        }},
        {{
            name: '训练效率',
            key: 'training_efficiency',
            weight: 0.2
        }},
        {{
            name: '稳定性',
            key: 'stability',
            weight: 0.2
        }}
    ];
    
    // 数据归一化函数
    function normalize(value, min, max) {{
        if (max === min) return 1;
        return (value - min) / (max - min);
    }}
    
    // 计算每个模型的原始数据
    const rawScores = modelNames.map(model => {{
        const data = otherData[model];
        
        // 计算参数效率 (AP50:95 / log(参数量))
        const efficiency = (data.best_AP50_95 || 0) / Math.log(data.n_parameters || 1);
        
        // 计算训练效率 (AP50:95 / epoch)
        const training_efficiency = (data.best_AP50_95 || 0) / (data.best_epoch || 1);
        
        // 计算稳定性
        const stability = calculateStability(model);
        
        return {{
            model: model,
            scores: {{
                best_AP50_95: data.best_AP50_95 || 0,
                efficiency: efficiency,
                training_efficiency: training_efficiency,
                stability: stability
            }}
        }};
    }});
    
    // 获取每个指标的最大最小值
    const ranges = metrics.reduce((acc, metric) => {{
        const values = rawScores.map(score => score.scores[metric.key]);
        acc[metric.key] = {{
            min: Math.min(...values),
            max: Math.max(...values)
        }};
        return acc;
    }}, {{}});
    
    // 计算归一化后的得分和综合得分
    const modelScores = rawScores.map(raw => {{
        const normalizedScores = {{}};
        metrics.forEach(metric => {{
            normalizedScores[metric.key] = normalize(
                raw.scores[metric.key],
                ranges[metric.key].min,
                ranges[metric.key].max
            );
        }});
        
        // 计算综合得分
        const totalScore = metrics.reduce((sum, metric) => {{
            return sum + normalizedScores[metric.key] * metric.weight;
        }}, 0);
        
        return {{
            ...raw,
            normalizedScores,
            totalScore
        }};
    }});
    
    // 计算稳定性指标
    function calculateStability(model) {{
        const data = otherData[model];
        
        // 使用多个指标计算稳定性
        const trainLoss = data.train_loss || 0;
        const learningRate = data.train_lr || 0;
        const epoch = data.best_epoch || 1;
        
        // 损失稳定性（损失值越小越稳定）
        const lossStability = 1 / (1 + trainLoss);
        
        // 学习率稳定性（学习率越小越稳定）
        const lrStability = 1 / (1 + learningRate);
        
        // 收敛稳定性（收敛轮次越少越稳定）
        const epochStability = 1 / Math.sqrt(epoch);
        
        // 综合稳定性得分
        return (lossStability * 0.4 + lrStability * 0.3 + epochStability * 0.3);
    }}
    
    // 准备雷达图数据
    const radarData = modelScores.map(score => {{
        return {{
            name: score.model,
            value: metrics.map(metric => score.normalizedScores[metric.key]),
            itemStyle: {{
                color: colorMapping[score.model]
            }}
        }};
    }});
    
    // 准备散点图数据
    const scatterData = modelScores.map(score => {{
        // 计算相对效率值（相对于最大参数量的模型）
        const maxParams = Math.max(...modelNames.map(m => otherData[m].n_parameters));
        const relativeEfficiency = (score.scores.efficiency * maxParams) / 
                                 (otherData[score.model].n_parameters || 1);
        
        return {{
            name: score.model,
            value: [
                relativeEfficiency * 100,        // 转换为百分比 - 现在是x轴
                score.scores.best_AP50_95 * 100, // 转换为百分比 - 现在是y轴
                score.totalScore
            ],
            itemStyle: {{
                color: colorMapping[score.model]
            }}
        }};
    }});
    
    // 计算坐标轴范围
    const xValues = scatterData.map(d => d.value[0]);
    const yValues = scatterData.map(d => d.value[1]);
    
    const xRange = {{
        min: Math.min(...xValues),
        max: Math.max(...xValues)
    }};
    const yRange = {{
        min: Math.min(...yValues),
        max: Math.max(...yValues)
    }};
    
    // 扩展范围以突出差异
    const xPadding = (xRange.max - xRange.min) * 0.2;
    const yPadding = (yRange.max - yRange.min) * 0.2;
    
    // 计算趋势线
    const linearRegression = (data) => {{
        const x = data.map(d => d.value[0]);
        const y = data.map(d => d.value[1]);
        const n = x.length;
        
        const xy = x.map((xi, i) => xi * y[i]);
        const xx = x.map(xi => xi * xi);
        
        const sum_x = x.reduce((a, b) => a + b, 0);
        const sum_y = y.reduce((a, b) => a + b, 0);
        const sum_xy = xy.reduce((a, b) => a + b, 0);
        const sum_xx = xx.reduce((a, b) => a + b, 0);
        
        const slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        const intercept = (sum_y - slope * sum_x) / n;
        
        return [
            [xRange.min - xPadding, slope * (xRange.min - xPadding) + intercept],
            [xRange.max + xPadding, slope * (xRange.max + xPadding) + intercept]
        ];
    }};
    
    const trendlineData = linearRegression(scatterData);
    
    const options = {{
        title: [
            {{
                text: '模型综合性能雷达图',
                left: '5%',
                top: '5%'
            }},
            {{
                text: '性能-效率散点图',
                left: '55%',
                top: '5%'
            }}
        ],
        tooltip: {{
            trigger: 'item',
            formatter: function(params) {{
                if (params.componentType === 'series' && params.seriesType === 'radar') {{
                    let result = params.name + '<br/>';
                    metrics.forEach((metric, index) => {{
                        const normalizedValue = params.value[index];
                        const rawValue = rawScores.find(s => s.model === params.name).scores[metric.key];
                        result += `${{metric.name}}: ${{formatNumber(rawValue)}} (归一化: ${{formatNumber(normalizedValue)}})<br/>`;
                    }});
                    return result;
                }} else if (params.componentType === 'series' && params.seriesType === 'scatter') {{
                    return params.name + '<br/>' +
                           '相对效率: ' + params.value[0].toFixed(2) + '%<br/>' +
                           '性能: ' + params.value[1].toFixed(2) + '%<br/>' +
                           '综合得分: ' + formatNumber(params.value[2]);
                }}
            }}
        }},
        legend: {{
            data: modelDisplayNames,
            top: '10%'
        }},
        radar: {{
            indicator: metrics.map(metric => ({{
                name: metric.name,
                max: 1,  // 归一化后的最大值为1
                min: 0   // 归一化后的最小值为0
            }})),
            center: ['25%', '60%'],
            radius: '40%',
            splitArea: {{
                areaStyle: {{
                    color: ['rgba(250,250,250,0.3)', 'rgba(235,235,235,0.3)']
                }}
            }}
        }},
        series: [
            {{
                type: 'radar',
                data: radarData,
                emphasis: {{
                    lineStyle: {{
                        width: 4
                    }}
                }}
            }},
            {{
                type: 'scatter',
                coordinateSystem: 'cartesian2d',
                data: scatterData,
                symbolSize: function(data) {{
                    if (!data || !data.value || data.value.length < 3) return 20;
                    // 使用sigmoid函数平滑点的大小变化
                    const score = data.value[2];
                    const size = 30 / (1 + Math.exp(-5 * (score - 0.5))) + 15;
                    return size;
                }},
                label: {{
                    show: true,
                    formatter: function(params) {{
                        return params.name;
                    }},
                    position: 'right',
                    distance: 10,
                    textStyle: {{
                        fontSize: 12,
                        fontWeight: 'bold'
                    }}
                }},
                emphasis: {{
                    scale: true,
                    itemStyle: {{
                        shadowBlur: 10,
                        shadowColor: 'rgba(0,0,0,0.3)'
                    }}
                }},
                xAxisIndex: 0,
                yAxisIndex: 0
            }},
            {{
                type: 'line',
                coordinateSystem: 'cartesian2d',
                data: trendlineData,
                showSymbol: false,
                lineStyle: {{
                    type: 'dashed',
                    opacity: 0.5,
                    width: 2
                }}
            }}
        ],
        grid: [
            {{
                left: '55%',
                top: '20%',
                right: '8%',    // 增加右侧边距
                bottom: '10%',
                containLabel: true
            }}
        ],
        xAxis: [
            {{
                gridIndex: 0,
                type: 'value',
                name: '相对参数效率 (%)',
                nameLocation: 'center',
                nameGap: 45,
                min: xRange.min - xPadding,
                max: xRange.max + xPadding,
                interval: (xRange.max - xRange.min) / 5,
                axisLabel: {{
                    formatter: function(value) {{
                        return value.toFixed(2) + '%';
                    }},
                    margin: 12,
                    rotate: 0
                }},
                splitLine: {{
                    show: true,
                    lineStyle: {{
                        type: 'dashed',
                        opacity: 0.3
                    }}
                }}
            }}
        ],
        yAxis: [
            {{
                gridIndex: 0,
                type: 'value',
                name: '性能 (AP50:95 %)',
                nameLocation: 'center',
                nameGap: 55,    // 进一步增加y轴标题间距
                position: 'left',  // 确保y轴在左侧
                offset: 5,      // 向左偏移
                min: yRange.min - yPadding,
                max: yRange.max + yPadding,
                interval: (yRange.max - yRange.min) / 5,
                axisLabel: {{
                    formatter: function(value) {{
                        return value.toFixed(2) + '%';
                    }},
                    margin: 20,  // 增加标签与轴的距离
                    align: 'right',
                    padding: [0, 15, 0, 0]  // 右侧内边距，使文字远离轴线
                }},
                splitLine: {{
                    show: true,
                    lineStyle: {{
                        type: 'dashed',
                        opacity: 0.3
                    }}
                }}
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
            const formattedValue = param.key === 'train_lr' ? 
                formatNumber(value, true) : formatNumber(value);
            tableHTML += `<td style="color:${{colorMapping[model]}}">${{formattedValue}}</td>`;
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