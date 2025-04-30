#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import json
import seaborn as sns

# Set clean plotting style
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']  # For displaying Chinese if needed
plt.rcParams['axes.unicode_minus'] = False  # For correct display of minus sign

# Define method name mapping
METHOD_MAPPING = {
    'MaxPool': 'MaxPool',
    'AvgPool': 'AvgPool',
    'S3Pool': 'S3Pool',
    'BDSLocPool2d_C': 'LDS3Pool_C',
    'BDSLocPool2d_pca': 'LDS3Pool_PCA',
    'BDSLocPool2d_hilbert': 'LDS3Pool_Hilbert',
}

# Define color mapping
COLOR_MAPPING = {
    'MaxPool': 'red',
    'AvgPool': 'gold',
    'S3Pool': 'greenyellow',
    'BDSLocPool2d_C': 'cyan',
    'BDSLocPool2d_pca': 'lightyellow',
    'BDSLocPool2d_hilbert': 'violet',
}

def process_data(data_file):
    """
    Process data output from extract.py
    Returns: accuracy dataframes, time dataframe
    """
    # Read data
    df = pd.read_csv(data_file)
    
    # Define specific model_dataset order
    model_datasets = [
        "LeNet5_MNIST",
        "LeNet5_FashionMNIST",
        "LeNet5_CIFAR10",
        "LeNet5_CIFAR100",
        "VGG11_CIFAR10",
        "VGG11_CIFAR100", 
        "VGG11_STL10",
        "VGG11_OxfordIIITPet"
    ]
    
    # Ensure all model_datasets in the data are included
    all_model_datasets = sorted(df['model_dataset'].unique())
    for md in all_model_datasets:
        if md not in model_datasets:
            model_datasets.append(md)
    
    # Store different pooling methods
    pooling_methods = []
    for _, row in df.iterrows():
        if row['exp_type'] == 'BASE':
            method = row['pooling_method']
            if method not in pooling_methods:
                pooling_methods.append(method)
        else:  # BDSL
            method = f"{row['pooling_method']}_{row['dim_reduction']}"
            if method not in pooling_methods:
                pooling_methods.append(method)
    
    # Create result dataframes
    accuracy_mean = pd.DataFrame(index=pooling_methods, columns=model_datasets)
    accuracy_std = pd.DataFrame(index=pooling_methods, columns=model_datasets)
    accuracy_cv = pd.DataFrame(index=pooling_methods, columns=model_datasets)  # Coefficient of variation
    epoch_time = pd.DataFrame(index=pooling_methods, columns=model_datasets)
    
    # Fill data
    for _, row in df.iterrows():
        # Get method identifier
        if row['exp_type'] == 'BASE':
            method = row['pooling_method']
        else:  # BDSL
            method = f"{row['pooling_method']}_{row['dim_reduction']}"
        
        model_dataset = row['model_dataset']
        
        # Extract accuracy and time data
        accuracies = json.loads(row['accuracies'].replace("'", '"'))
        times = json.loads(row['avg_epoch_times'].replace("'", '"'))
        
        # Calculate statistics
        if accuracies:
            acc_mean = np.mean(accuracies)
            acc_std = np.std(accuracies, ddof=1)  # Use unbiased estimator
            acc_cv = acc_std / acc_mean if acc_mean > 0 else 0  # Coefficient of variation
            
            accuracy_mean.at[method, model_dataset] = acc_mean
            accuracy_std.at[method, model_dataset] = acc_std
            accuracy_cv.at[method, model_dataset] = acc_cv
        
        if times:
            epoch_time.at[method, model_dataset] = np.mean(times)
    
    # Ensure data type is float
    accuracy_mean = accuracy_mean.astype(float)
    accuracy_std = accuracy_std.astype(float)
    accuracy_cv = accuracy_cv.astype(float)
    epoch_time = epoch_time.astype(float)
    
    return accuracy_mean, accuracy_std, accuracy_cv, epoch_time

def format_x_labels(models_datasets):
    """Format labels: split 'Model_Dataset' into two lines"""
    formatted_labels = []
    for label in models_datasets:
        parts = label.split('_')
        if len(parts) == 2:
            formatted_labels.append(f"{parts[0]}\n{parts[1]}")
        else:
            formatted_labels.append(label)
    return formatted_labels

def add_best_method_markers(ax, df, models_datasets, bars_dict, methods):
    """Mark the best method for each model+dataset"""
    for i, model_dataset in enumerate(models_datasets):
        # Get values for current dataset across all methods
        dataset_values = df[model_dataset].values
        best_method_idx = np.argmax(dataset_values)
        # Highlight best method
        method_name = methods[best_method_idx]
        if method_name in bars_dict:
            bar = bars_dict[method_name][i]
            # Add star on top of the best method's bar
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                   '*', ha='center', va='center', color='black', fontsize=20)

def add_dataset_dividers(ax, models_datasets):
    """Add dataset dividers"""
    dataset_dividers = ["LeNet5", "VGG11"]
    prev_dataset = None
    for i, model_dataset in enumerate(models_datasets):
        current_dataset = model_dataset.split("_")[0]
        if prev_dataset and current_dataset != prev_dataset and current_dataset in dataset_dividers:
            ax.axvline(x=i-0.5, color='gray', linestyle='-', alpha=0.15)
        prev_dataset = current_dataset

def create_bar_chart(df_accuracy, df_std, methods_to_include, title, output_filename, hatches=None):
    """
    Create bar chart for specific methods with error bars
    """
    # Filter required rows
    df_filtered = df_accuracy.loc[methods_to_include]
    df_std_filtered = df_std.loc[methods_to_include]
    
    # Rename pooling methods (row labels)
    df_filtered.index = [METHOD_MAPPING.get(method, method) for method in df_filtered.index]
    df_std_filtered.index = [METHOD_MAPPING.get(method, method) for method in df_std_filtered.index]
    methods = df_filtered.index
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Get data
    models_datasets = df_filtered.columns  # Column headers (model+dataset)
    
    # Set grouped bar chart parameters
    n_methods = len(methods)
    n_models_datasets = len(models_datasets)
    width = 0.18  # Width of each bar
    
    # Calculate position for each group of bars
    indices = np.arange(n_models_datasets)
    
    # If no hatches provided, generate default values
    if hatches is None:
        hatches = ['', '..', '///', '\\\\\\', 'xxx', '+++'][:n_methods]
    
    # Draw bars for each method
    bars_dict = {}
    for i, method in enumerate(methods):
        # Find original method name for correct color
        original_method = [m for m in methods_to_include if METHOD_MAPPING.get(m, m) == method][0]
        
        offset = (i - n_methods / 2 + 0.5) * width
        values = df_filtered.loc[method].values
        errors = df_std_filtered.loc[method].values
        
        # Draw bar chart with error bars
        bars = ax.bar(indices + offset, values, width, label=method, 
                     color=COLOR_MAPPING.get(original_method, f'C{i}'), 
                     edgecolor='black', linewidth=0.5, alpha=0.85,
                     hatch=hatches[i % len(hatches)],
                     yerr=errors, capsize=5, error_kw={'elinewidth': 1, 'capthick': 1})
        bars_dict[method] = bars
    
    # Set chart title and labels
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xlabel('Model and Dataset', fontsize=14)
    
    # Set x-axis ticks and display in two lines
    ax.set_xticks(indices)
    ax.set_xticklabels(format_x_labels(models_datasets), fontsize=12)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Set y-axis range based on data
    min_val = max(0, df_filtered.values.min() * 0.9)  # Slightly lower than min value, but not less than 0
    ax.set_ylim(min_val, 1.05)
    
    # Add legend, positioned at bottom of chart
    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.2),
              ncol=n_methods, frameon=False)
    
    # Add grid lines, horizontal only
    ax.grid(True, axis='y', linestyle='-', alpha=0.2)
    ax.grid(False, axis='x')
    
    # Add dataset dividers
    add_dataset_dividers(ax, models_datasets)
    
    # Mark best methods
    add_best_method_markers(ax, df_filtered, models_datasets, bars_dict, methods)
    
    # Adjust layout for bottom legend
    plt.tight_layout(rect=[0, 0, 1, 1])
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save chart
    plt.savefig(f"{output_filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_filename}.svg", format='svg', bbox_inches='tight')
    
    print(f"Chart saved to {output_filename}.png")
    plt.close()

def create_tables(df_accuracy_mean, df_accuracy_std, df_accuracy_cv, df_epoch_time, output_dir):
    """
    Create and save tables
    """
    # Prepare output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Rename index labels before saving
    for df in [df_accuracy_mean, df_accuracy_std, df_accuracy_cv, df_epoch_time]:
        df.index = [METHOD_MAPPING.get(method, method) for method in df.index]
    
    # Save mean accuracy table
    mean_path = os.path.join(output_dir, "accuracy_mean.csv")
    df_accuracy_mean.to_csv(mean_path)
    print(f"Mean accuracy table saved to {mean_path}")
    
    # Save accuracy standard deviation table
    std_path = os.path.join(output_dir, "accuracy_std.csv")
    df_accuracy_std.to_csv(std_path)
    print(f"Accuracy standard deviation table saved to {std_path}")
    
    # Save coefficient of variation table
    cv_path = os.path.join(output_dir, "accuracy_cv.csv")
    df_accuracy_cv.to_csv(cv_path)
    print(f"Coefficient of variation table saved to {cv_path}")
    
    # Save epoch average time table
    time_path = os.path.join(output_dir, "epoch_time.csv")
    df_epoch_time.to_csv(time_path)
    print(f"Average epoch time table saved to {time_path}")

def create_heatmap(df, title, output_filename, fmt=".4f", cmap="YlGnBu"):
    """
    Create heatmap
    """
    # Handle NaN values in dataframe
    df_clean = df.copy()
    mask = df_clean.isna()
    
    plt.figure(figsize=(14, 6))
    
    # Use masked heatmap to hide NaN values
    ax = sns.heatmap(df_clean, annot=True, fmt=fmt, cmap=cmap, linewidths=.5, 
                    mask=mask, annot_kws={"size": 9})
    
    # Format x labels to split model and dataset names into two lines
    xlabels = [format_x_labels([col])[0] for col in df_clean.columns]
    ax.set_xticklabels(xlabels, fontsize=10, rotation=0)
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save chart
    plt.savefig(f"{output_filename}.png", dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_filename}.png")
    plt.close()

def create_box_chart(df_accuracy, methods_to_include, title, output_filename):
    """
    Create box chart for specific methods
    """
    # Filter required rows
    df_filtered = df_accuracy.loc[methods_to_include]
    
    # Rename pooling methods (row labels)
    df_filtered.index = [METHOD_MAPPING.get(method, method) for method in df_filtered.index]
    methods = df_filtered.index
    
    # Create plot with 2 rows and 4 columns
    fig, axs = plt.subplots(2, 4, figsize=(14, 6))
    axs = axs.flatten()  # Flatten the 2D array of axes for easier iteration
    
    # Get data
    models_datasets = df_filtered.columns  # Column headers (model+dataset)
    
    # Set grouped box chart parameters
    n_methods = len(methods)
    width = 0.25  # Width of each box (increased from 0.18)
    
    # Draw boxes for each model_dataset
    for idx, model_dataset in enumerate(models_datasets):
        ax = axs[idx]
        
        # Get values for current model_dataset
        values = df_filtered[model_dataset].values
        
        # Create data for boxplot - each method needs a list of values
        data = []
        positions = []
        all_accuracies = []  # Store all accuracy values for this subplot
        
        for i, method in enumerate(methods):
            # Find original method name for correct color
            original_method = [m for m in methods_to_include if METHOD_MAPPING.get(m, m) == method][0]
            
            # Get the 6 accuracy values from the original data
            data_file = f"output/runs_0419_1/extracted_data_0419_1.csv"
            df_original = pd.read_csv(data_file)
            
            # Handle BDSL methods
            if original_method.startswith('BDSLocPool2d'):
                # Extract the dimension reduction method
                dim_reduction = original_method.split('_')[-1]
                # Filter for BDSL methods with matching dimension reduction
                row = df_original[
                    (df_original['exp_type'] == 'BDSL') & 
                    (df_original['dim_reduction'] == dim_reduction) & 
                    (df_original['model_dataset'] == model_dataset)
                ]
            else:
                # Filter for traditional methods
                row = df_original[
                    (df_original['exp_type'] == 'BASE') & 
                    (df_original['pooling_method'] == original_method) & 
                    (df_original['model_dataset'] == model_dataset)
                ]
            
            if not row.empty:
                # Extract the 6 accuracy values
                accuracies = json.loads(row['accuracies'].iloc[0].replace("'", '"'))
                data.append(accuracies)
                positions.append(i)
                all_accuracies.extend(accuracies)  # Add to all accuracies list
        
        # Draw box plot for all methods at once
        box = ax.boxplot(data, positions=positions, widths=width,
                        patch_artist=True,
                        boxprops=dict(facecolor=COLOR_MAPPING.get(original_method, f'C{i}'),
                                    edgecolor='black', linewidth=0.5, alpha=0.85),
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='black', linewidth=0.5),
                        capprops=dict(color='black', linewidth=0.5),
                        flierprops=dict(marker='o', markersize=3, markerfacecolor='black', markeredgecolor='none'),
                        whis=1.5)
        
        # Set colors for each box
        for i, box in enumerate(box['boxes']):
            box.set_facecolor(COLOR_MAPPING.get(methods_to_include[i], f'C{i}'))
        
        # Set subplot title
        ax.set_title(format_x_labels([model_dataset])[0], fontsize=12)
        
        # Set y-axis range based on all accuracy values
        if all_accuracies:  # Only if we have data
            min_val = min(all_accuracies)
            max_val = max(all_accuracies)
            # Add small margins (1% of the range)
            range_val = max_val - min_val
            margin = range_val * 0.02
            ax.set_ylim(max(0, min_val - margin), min(1, max_val + margin))
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # Add grid lines, horizontal only
        ax.grid(True, axis='y', linestyle='-', alpha=0.2)
        ax.grid(False, axis='x')
        
        # Hide x-axis labels and ticks
        ax.set_xticks([])
        ax.set_xticklabels([])
        
        # Only show y-axis label for the first subplot in each row
        if idx % 4 == 0:  # First subplot in each row
            ax.set_ylabel('Accuracy', fontsize=12)
    
    # Add overall title with reduced distance
    fig.suptitle(title, fontsize=16, y=0.95)
    
    # Add legend at the bottom with reduced distance
    handles = [plt.Rectangle((0,0),1,1, facecolor=COLOR_MAPPING.get(m, 'C0'), 
                            edgecolor='black', linewidth=0.5) for m in methods_to_include]
    fig.legend(handles, methods, fontsize=12, loc='upper center', 
              bbox_to_anchor=(0.5, 0.02), ncol=n_methods, frameon=False)
    
    # Adjust layout with reduced spacing
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save chart
    plt.savefig(f"{output_filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_filename}.svg", format='svg', bbox_inches='tight')
    
    print(f"Box chart saved to {output_filename}.png")
    plt.close()

def main():
    # Input file
    date = "0419_1"
    extracted_data_file = f"output/runs_{date}/extracted_data_{date}.csv"
    
    # Output directory
    output_dir = f"output/runs_{date}/analysis"
    
    # Process data
    df_acc_mean, df_acc_std, df_acc_cv, df_epoch_time = process_data(extracted_data_file)
    
    # Note: we'll create copy of dataframes for plotting before renaming indices
    df_acc_mean_plot = df_acc_mean.copy()
    df_acc_std_plot = df_acc_std.copy()
    
    # Create tables - this modifies the dataframes' indices
    create_tables(df_acc_mean, df_acc_std, df_acc_cv, df_epoch_time, output_dir)
    
    # Create heatmaps - we use the already modified dataframes with LDS3Pool indices
    create_heatmap(df_acc_mean, "Validation Accuracy Mean", f"{output_dir}/heatmap_accuracy_mean", fmt=".4f")
    create_heatmap(df_acc_std, "Validation Accuracy Standard Deviation", f"{output_dir}/heatmap_accuracy_std", fmt=".4f")
    create_heatmap(df_acc_cv, "Validation Accuracy Coefficient of Variation", f"{output_dir}/heatmap_accuracy_cv", fmt=".4f")
    create_heatmap(df_epoch_time, "Average Epoch Time (seconds)", f"{output_dir}/heatmap_epoch_time", fmt=".2f", cmap="YlOrRd")
    
    # Create first bar chart: traditional methods and LDS3Pool_hilbert
    # Use original dataframes with BDSLocPool2d indices
    traditional_methods = [
        'MaxPool',
        'AvgPool',
        'S3Pool',
        'BDSLocPool2d_hilbert'
    ]
    create_bar_chart(
        df_acc_mean_plot, df_acc_std_plot,
        traditional_methods,
        'Validation Accuracy: Control Group vs. LDS3Pool-Hilbert',
        f"{output_dir}/validation_accuracy_traditional",
        hatches=['', '..', '///', '\\\\\\']
    )
    
    # Create second bar chart: three LDS3Pool methods
    lds3pool_methods = [
        'BDSLocPool2d_C',
        'BDSLocPool2d_pca',
        'BDSLocPool2d_hilbert'
    ]
    create_bar_chart(
        df_acc_mean_plot, df_acc_std_plot,
        lds3pool_methods,
        'Validation Accuracy: LDS3Pool Variants Comparison',
        f"{output_dir}/validation_accuracy_lds3pool",
        hatches=['', '..', '///']
    )

    # Create first box chart: traditional methods and LDS3Pool_hilbert
    create_box_chart(
        df_acc_mean_plot,
        traditional_methods,
        'Validation Accuracy: Control Group vs. LDS3Pool-Hilbert',
        f"{output_dir}/accuracy_box_plot_traditional"
    )
    
    # Create second box chart: three LDS3Pool methods
    create_box_chart(
        df_acc_mean_plot,
        lds3pool_methods,
        'Validation Accuracy: LDS3Pool Variants Comparison',
        f"{output_dir}/accuracy_box_plot_lds3pool"
    )

if __name__ == "__main__":
    main() 