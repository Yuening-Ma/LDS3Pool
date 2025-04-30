#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import csv
from collections import defaultdict

def read_max_accuracy(event_file_path, tag="accuracy", step=99):
    """读取事件文件中的最大准确率和对应步骤"""
    try:
        ea = event_accumulator.EventAccumulator(event_file_path)
        ea.Reload()  # 加载事件文件

        if tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            max_accuracy = float("-inf")
            max_step = -1
            for event in events:
                if 0 <= event.step <= step and event.value > max_accuracy:
                    max_accuracy = event.value
                    max_step = event.step
            if max_accuracy != float("-inf"):
                return max_accuracy, max_step
    except Exception as e:
        print(f"读取文件 {event_file_path} 时出错: {e}")
    return None, None

def read_avg_epoch_time(event_file_path, step=99):
    """读取事件文件中的平均epoch时间"""
    try:
        ea = event_accumulator.EventAccumulator(event_file_path)
        ea.Reload()  # 加载事件文件

        tag = "epoch_time"
        if tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            total_time = 0
            count = 0
            for event in events:
                if 0 <= event.step <= step:
                    total_time += event.value
                    count += 1
            if count > 0:
                return total_time / count
    except Exception as e:
        print(f"读取文件 {event_file_path} 时出错: {e}")
    return None

def extract_bdsl_data(root_dir, date):
    """提取BDSL数据"""
    # 读取配置文件
    json_path = f'{root_dir}para_bdsl_2025{date}.json'
    with open(json_path, 'r') as f:
        bdsl_combinations = json.load(f)

    # 参数组合字典: {参数组合标识: [随机种子...]}
    param_groups = defaultdict(list)
    param_to_index = {}

    # 识别参数组合
    for config in bdsl_combinations:
        # 提取除random_seed和index外的参数，添加k字段作为分组条件
        param_key = (
            config["model_dataset"],
            config["pooling_method"],
            config["dim_reduction"],
            config.get("k", "None")  # 添加k字段，如果不存在则使用"None"
        )
        # 加入随机种子
        param_groups[param_key].append(config["random_seed"])
        # 记录参数到index的映射
        index = config["index"]
        param_to_index[(param_key, config["random_seed"])] = index

    # 存储结果
    results = []

    # 处理每个参数组合
    exp_dir = os.path.join(root_dir, "BDSL")
    for param_key, seeds in param_groups.items():
        model_dataset, pooling_method, dim_reduction, k_value = param_key
        
        # 确定epoch数量
        max_epoch = 150 if "OxfordIIITPet" in model_dataset else 100
        
        # 每个种子的数据
        seed_accs = []
        seed_times = []
        
        for seed in seeds:
            index = param_to_index[(param_key, seed)]
            exp_id = str(index).rjust(4, '0')
            exp_path = os.path.join(exp_dir, exp_id)
            
            # 验证集最大准确率
            accuracy_val_dir = os.path.join(exp_path, "accuracy_val")
            accuracy, _ = read_max_accuracy(accuracy_val_dir, step=max_epoch-1)
            
            # 平均epoch时间
            main_event_file = os.path.join(exp_path, "")
            avg_time = read_avg_epoch_time(main_event_file, step=max_epoch-1)
            
            if accuracy is not None:
                seed_accs.append(accuracy)
            if avg_time is not None:
                seed_times.append(avg_time)

        # 将参数组合数据添加到结果中
        param_str = f"{model_dataset}_{pooling_method}_{dim_reduction}_k{k_value}"
        results.append({
            "exp_type": "BDSL",
            "param_combination": param_str,
            "model_dataset": model_dataset,
            "pooling_method": pooling_method,
            "dim_reduction": dim_reduction,
            "k_value": k_value,  # 添加k值到结果中
            "seeds": seeds,
            "accuracies": seed_accs,
            "avg_epoch_times": seed_times
        })
        
    return results

def save_to_csv(data, output_file):
    """保存数据到CSV文件"""
    # 确定字段列表
    fields = list(data[0].keys())
    
    # 保存到CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(fields)
        # 写入数据
        for row in data:
            writer.writerow([row[field] for field in fields])
    
    print(f"数据已保存到 {output_file}")

def save_expanded_data(data, output_file):
    """保存展开后的数据到CSV文件，将accuracies和avg_epoch_times展开为单独的列"""
    # 分析数据确定最大种子数
    max_seeds = 0
    for item in data:
        max_seeds = max(max_seeds, len(item.get('accuracies', [])))
        max_seeds = max(max_seeds, len(item.get('avg_epoch_times', [])))
    
    # 创建表头
    headers = ['exp_type', 'param_combination', 'model_dataset', 'pooling_method', 'dim_reduction', 'k_value']
    for i in range(max_seeds):
        headers.append(f'accuracy_{i+1}')
    for i in range(max_seeds):
        headers.append(f'epoch_time_{i+1}')
    
    # 写入CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for item in data:
            row = [
                item.get('exp_type', ''),
                item.get('param_combination', ''),
                item.get('model_dataset', ''),
                item.get('pooling_method', ''),
                item.get('dim_reduction', ''),
                item.get('k_value', '')
            ]
            
            # 添加accuracies值
            accuracies = item.get('accuracies', [])
            for i in range(max_seeds):
                if i < len(accuracies):
                    row.append(accuracies[i])
                else:
                    row.append('')
            
            # 添加epoch_times值
            times = item.get('avg_epoch_times', [])
            for i in range(max_seeds):
                if i < len(times):
                    row.append(times[i])
                else:
                    row.append('')
            
            writer.writerow(row)
    
    print(f"展开的数据已保存到 {output_file}")

def main():
    date = "0426"
    root_dir = f"output/runs_{date}/"
    
    # 提取BDSL数据
    bdsl_data = extract_bdsl_data(root_dir, date)
    
    # 不再提取BASE数据
    combined_data = bdsl_data
    
    # 保存到root_dir目录下
    output_file = os.path.join(root_dir, f"extracted_k_data_{date}.csv")
    save_to_csv(combined_data, output_file)
    
    # 保存展开数据格式（每个随机种子对应单独的列）
    expanded_output_file = os.path.join(root_dir, f"extracted_k_data_expanded_{date}.csv")
    save_expanded_data(combined_data, expanded_output_file)
    
    # 输出提取的数据摘要
    print(f"BDSL实验: {len(bdsl_data)}个参数组合")

if __name__ == "__main__":
    main() 