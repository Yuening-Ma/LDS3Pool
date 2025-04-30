import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import glob
import re
import json
from collections import OrderedDict
from scipy.signal import savgol_filter

# 定义方法名称映射和绘图顺序
METHOD_MAPPING = OrderedDict([
    ('MaxPool', 'MaxPool'),
    ('AvgPool', 'AvgPool'),
    ('S3Pool', 'S3Pool'),
    ('BDSLocPool2d_C', 'LDS3Pool_C'),
    ('BDSLocPool2d_pca', 'LDS3Pool_PCA'),
    ('BDSLocPool2d_hilbert', 'LDS3Pool_Hilbert'),
])

# 颜色映射
COLOR_MAPPING = {
    'MaxPool': 'cyan',
    'AvgPool': 'gold',
    'S3Pool': 'yellowgreen',
    'BDSLocPool2d_C': 'violet',
    'BDSLocPool2d_pca': 'blue',
    'BDSLocPool2d_hilbert': 'red',
}

# 选择要绘制的模型和数据集组合
MODEL_DATASET = "LeNet5_CIFAR10"
# MODEL_DATASET = "VGG11_OxfordIIITPet"

# 标记样式
MARKERS = ['o', 's', 'p', 'D', 'v', '^', '*', 'x', '+', '|', '_', '.']

# 平滑设置
WINDOW_LENGTH = 11  # 平滑窗口长度，必须是奇数
POLYORDER = 3      # 多项式阶数，必须小于窗口长度F

def smooth_data(y, window_length=WINDOW_LENGTH, polyorder=POLYORDER):
    """使用Savitzky-Golay滤波器平滑数据"""
    if len(y) < window_length:
        # 如果数据点太少，使用简单的移动平均
        return np.convolve(y, np.ones(min(5, len(y)))/min(5, len(y)), mode='same')
    return savgol_filter(y, window_length, polyorder)

def extract_accuracy_from_tensorboard(log_dir, accuracy_type='train'):
    """从TensorBoard日志中提取训练或验证准确率数据"""
    # 确定正确的日志目录
    accuracy_dir = 'accuracy_train' if accuracy_type == 'train' else 'accuracy_val'
    
    # 查找目录下的事件文件
    event_files = glob.glob(f"{log_dir}/{accuracy_dir}/events.out.tfevents.*")
    
    if not event_files:
        print(f"警告: 未找到{accuracy_type}准确率日志文件: {log_dir}/{accuracy_dir}/")
        return None, None
    
    # 使用第一个匹配的事件文件
    event_file = event_files[0]
    
    # 加载事件文件
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    # 提取准确率值和对应的步骤
    if 'accuracy' in ea.scalars.Keys():
        accuracy_events = ea.Scalars('accuracy')
        steps = [event.step for event in accuracy_events]
        accuracy_values = [event.value for event in accuracy_events]
        return steps, accuracy_values
    else:
        print(f"警告: 在 {event_file} 中未找到 'accuracy' 标量")
        return None, None

def get_params_from_json(date, exp_type):
    """从参数JSON文件中读取所有配置"""
    json_file = f"output/runs_{date}/para_{exp_type}_2025{date}.json"
    try:
        with open(json_file, 'r') as f:
            params = json.load(f)
        return params
    except Exception as e:
        print(f"警告: 无法读取参数文件 {json_file}: {e}")
        return []

def get_method_name(param_dict):
    """根据参数获取方法的完整名称，处理BDSLocPool2d的不同dim_reduction取值"""
    pooling_method = param_dict.get('pooling_method', '')
    
    # 如果是BDSLocPool2d，根据dim_reduction分为不同方法
    if pooling_method == 'BDSLocPool2d' and 'dim_reduction' in param_dict:
        dim_reduction = param_dict.get('dim_reduction', '')
        return f"{pooling_method}_{dim_reduction}"
    
    return pooling_method

def collect_data_by_methods(date, model_dataset, specific_seed=None, accuracy_type='train'):
    """从指定的runs_{date}目录中收集所有方法的数据，可以指定特定的随机种子"""
    # 用于存储按方法分组的数据
    all_method_data = {}
    
    # 处理BASE和BDSL两种类型
    for exp_type in ['base', 'bdsl']:
        EXP_TYPE = exp_type.upper()
        
        # 读取参数配置
        params_list = get_params_from_json(date, exp_type)
        if not params_list:
            print(f"警告: 无法获取日期 {date} 的 {exp_type} 参数，跳过")
            continue
        
        # 过滤出符合MODEL_DATASET和指定种子的参数
        filtered_params = []
        for param in params_list:
            if param.get('model_dataset') == model_dataset:
                # 如果指定了种子，则只保留该种子的参数
                if specific_seed is None or param.get('random_seed') == specific_seed:
                    # 只保留BDSLocPool2d_hilbert，过滤掉BDSLocPool2d_pca和BDSLocPool2d_C
                    if param.get('pooling_method') == 'BDSLocPool2d':
                        if param.get('dim_reduction') == 'hilbert':
                            filtered_params.append(param)
                    else:
                        # 非BDSLocPool2d的方法直接保留
                        filtered_params.append(param)
        
        if not filtered_params:
            if specific_seed:
                print(f"警告: 日期 {date} 的 {exp_type} 实验中没有找到 {model_dataset} 的 seed={specific_seed} 数据")
            else:
                print(f"警告: 日期 {date} 的 {exp_type} 实验中没有找到 {model_dataset} 的数据")
            continue
        
        # 按池化方法分组，把BDSLocPool2d的不同dim_reduction作为不同方法
        method_groups = {}
        for param in filtered_params:
            # 获取完整的方法名称（包括dim_reduction信息）
            method_name = get_method_name(param)
            # 如果已经有这个方法了，而且我们有指定的种子，那么以种子为准
            # 否则使用index较小的那个
            if method_name in method_groups:
                if specific_seed is not None:
                    # 如果指定了种子，使用该种子的参数
                    if param.get('random_seed') == specific_seed:
                        method_groups[method_name] = param
                elif param.get('index', 0) < method_groups[method_name].get('index', 0):
                    # 否则使用index较小的
                    method_groups[method_name] = param
            else:
                method_groups[method_name] = param
        
        seed_info = f"seed={specific_seed}" if specific_seed is not None else "所有可用种子"
        print(f"日期 {date}, 实验 {exp_type}, {seed_info}, 找到方法: {list(method_groups.keys())}")
        
        # 遍历每个方法
        for method_name, param_dict in method_groups.items():
            # 获取运行索引
            run_idx = param_dict.get('index', 0)
            run_id = str(run_idx).rjust(4, '0')
            
            # 构建日志目录路径
            log_dir = f"output/runs_{date}/{EXP_TYPE}/{run_id}"
            
            # 提取准确率数据
            steps, acc_values = extract_accuracy_from_tensorboard(log_dir, accuracy_type)
            if steps is None or acc_values is None:
                print(f"警告: 无法提取 {log_dir} 的{accuracy_type}准确率数据")
                continue
            
            # 存储数据
            all_method_data[method_name] = {
                'steps': steps,
                'accuracy': acc_values,
                'exp_type': exp_type,
                'seed': param_dict.get('random_seed', None)
            }
    
    return all_method_data

def plot_accuracy_curve(ax, date, model_dataset, specific_seed=None, accuracy_type='train'):
    """绘制指定类型的准确率曲线，可以指定特定的随机种子"""
    # 收集所有方法的数据
    all_method_data = collect_data_by_methods(date, model_dataset, specific_seed, accuracy_type)
    
    # 获取数据中所有的方法名称
    method_names = list(all_method_data.keys())
    
    # 按照预定义顺序绘制方法，若方法不在预定义映射中则按找到的顺序绘制
    plotted_methods = []
    
    # 首先绘制在METHOD_MAPPING中的方法
    for method_name, mapped_name in METHOD_MAPPING.items():
        if method_name in all_method_data:
            plotted_methods.append(method_name)
            data = all_method_data[method_name]
            steps = data['steps']
            acc_values = data['accuracy']
            seed = data['seed']
            
            # 平滑数据
            smoothed_values = smooth_data(acc_values)
            
            # 获取颜色和标记
            color = COLOR_MAPPING.get(method_name, 'blue')
            marker_idx = list(METHOD_MAPPING.keys()).index(method_name) % len(MARKERS)
            marker = MARKERS[marker_idx]
            
            # 绘制原始数据（淡色，无标记）
            ax.plot(steps, acc_values, color=color, linestyle='-', alpha=0.2, linewidth=1)
            
            # 绘制平滑后的数据（实线，带标记）
            seed_label = f" (seed={seed})" if seed is not None else ""
            ax.plot(steps, smoothed_values, color=color, linestyle='-', linewidth=2,
                    marker=marker, markevery=max(1, len(steps)//10), 
                    label=f"{mapped_name}{seed_label}", markersize=7)
    
    # 绘制不在METHOD_MAPPING中的方法
    for method_idx, method_name in enumerate([m for m in method_names if m not in plotted_methods]):
        data = all_method_data[method_name]
        steps = data['steps']
        acc_values = data['accuracy']
        seed = data['seed']
        
        # 平滑数据
        smoothed_values = smooth_data(acc_values)
        
        # 选择颜色和标记
        color_idx = (len(METHOD_MAPPING) + method_idx) % len(COLOR_MAPPING)
        color = f'C{color_idx}'
        marker = MARKERS[(len(METHOD_MAPPING) + method_idx) % len(MARKERS)]
        
        # 绘制原始数据（淡色，无标记）
        ax.plot(steps, acc_values, color=color, linestyle='-', alpha=0.2, linewidth=1)
        
        # 绘制平滑后的数据（实线，带标记）
        seed_label = f" (seed={seed})" if seed is not None else ""
        ax.plot(steps, smoothed_values, color=color, linestyle='-', linewidth=2,
                marker=marker, markevery=max(1, len(steps)//10), 
                label=f"{method_name}", markersize=7)
    
    # 设置子图标题和标签
    title = "Training Accuracy" if accuracy_type == 'train' else "Validation Accuracy"
    seed_title = f"(seed={specific_seed})" if specific_seed is not None else ""
    ax.set_title(f"{model_dataset} {title} {seed_title}", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    
    # 添加网格线，但只显示横向网格，更加淡化
    ax.grid(True, axis='y', linestyle='-', alpha=0.4)
    ax.grid(False, axis='x')
    
    # 添加图例，更加简洁
    ax.legend(loc='lower right', fontsize=10, frameon=False)

def main():
    # 设置日期和种子
    date = "0419_1"  # 可以从命令行参数获取
    specific_seed = 7  # 指定具体的随机种子，设为None则使用所有可用的种子中的第一个
    
    # 创建1行2列的子图布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 绘制训练集准确率
    plot_accuracy_curve(ax1, date, MODEL_DATASET, specific_seed, 'train')
    
    # 绘制验证集准确率
    plot_accuracy_curve(ax2, date, MODEL_DATASET, specific_seed, 'val')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表，确保输出目录存在
    output_dir = f"output/runs_{date}/analysis/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果指定了种子，在文件名中体现
    seed_suffix = f"_seed{specific_seed}" if specific_seed is not None else ""
    
    plt.savefig(f"{output_dir}{MODEL_DATASET}_accuracy_trends_smooth{seed_suffix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}{MODEL_DATASET}_accuracy_trends_smooth{seed_suffix}.svg", format='svg', bbox_inches='tight')
    # plt.savefig(f"{output_dir}{MODEL_DATASET}_accuracy_trends_smooth{seed_suffix}.pdf", format='pdf', bbox_inches='tight')
    
    print(f"已保存平滑版训练趋势图至 {output_dir}{MODEL_DATASET}_accuracy_trends_smooth{seed_suffix}.png")
    
    # 显示图表
    plt.close()  # 关闭而不是显示，避免在服务器运行时阻塞

if __name__ == "__main__":
    main()
