#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import json
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def perform_paired_t_tests(data_file):
    """
    Perform paired t-tests between LDS3Pool_Hilbert and other methods
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
    
    # Methods to compare with LDS3Pool_Hilbert
    comparison_methods = ['MaxPool', 'AvgPool', 'S3Pool', 'BDSLocPool2d_pca', 'BDSLocPool2d_C']
    
    # Target method (LDS3Pool_Hilbert)
    target_method = 'BDSLocPool2d_hilbert'
    
    # Create result dataframes for t-values and p-values
    t_values = pd.DataFrame(index=comparison_methods, columns=model_datasets)
    p_values = pd.DataFrame(index=comparison_methods, columns=model_datasets)
    
    # For each model_dataset, perform paired t-tests
    for model_dataset in model_datasets:
        # Get target method data
        target_data = None
        for _, row in df.iterrows():
            if row['model_dataset'] == model_dataset and \
               ((row['exp_type'] == 'BDSL' and f"{row['pooling_method']}_{row['dim_reduction']}" == target_method) or \
                (row['exp_type'] == 'BASE' and row['pooling_method'] == target_method)):
                accuracies = json.loads(row['accuracies'].replace("'", '"'))
                target_data = np.array(accuracies)
                break
        
        if target_data is None or len(target_data) == 0:
            continue
        
        # Compare with each comparison method
        for comp_method in comparison_methods:
            comp_data = None
            for _, row in df.iterrows():
                if row['model_dataset'] == model_dataset and \
                   ((row['exp_type'] == 'BDSL' and f"{row['pooling_method']}_{row['dim_reduction']}" == comp_method) or \
                    (row['exp_type'] == 'BASE' and row['pooling_method'] == comp_method)):
                    accuracies = json.loads(row['accuracies'].replace("'", '"'))
                    comp_data = np.array(accuracies)
                    break
            
            if comp_data is None or len(comp_data) == 0:
                continue
            
            # Ensure same length for paired t-test
            min_len = min(len(target_data), len(comp_data))
            target_data_trimmed = target_data[:min_len]
            comp_data_trimmed = comp_data[:min_len]
            
            # Perform paired t-test
            t_stat, p_val = stats.ttest_rel(target_data_trimmed, comp_data_trimmed)
            
            # Store results (note: t_stat is positive if target method is better)
            # We want t_stat to be positive if target is better
            if np.mean(target_data_trimmed) < np.mean(comp_data_trimmed):
                t_stat = -t_stat
                
            t_values.at[comp_method, model_dataset] = t_stat
            p_values.at[comp_method, model_dataset] = p_val
    
    return t_values, p_values

def perform_paired_t_tests_across_datasets(accuracy_mean_path):
    """
    Perform paired t-tests across datasets, treating each model_dataset as a paired sample
    This function analyzes whether methods differ in performance across all model_dataset combinations
    """
    # Read the accuracy mean table
    df = pd.read_csv(accuracy_mean_path, index_col=0)
    
    # Rename LDS3Pool back to BDSLocPool2d to match internal naming
    df.index = [index.replace('LDS3Pool', 'BDSLocPool2d') for index in df.index]
    
    # Target method (LDS3Pool_Hilbert)
    target_method = 'BDSLocPool2d_Hilbert'

    print(df.index)
    
    # Get target method data across all datasets
    if target_method not in df.index:
        print(f"Target method {target_method} not found in accuracy mean table")
        return None, None
    
    target_data = df.loc[target_method].values
    
    # Methods to compare with LDS3Pool_Hilbert (excluding the target itself)
    comparison_methods = [method for method in df.index if method != target_method]
    
    # Create result dataframes for t-values and p-values
    # This is now a single row dataframe as we're comparing across all datasets
    t_values = pd.DataFrame(index=['across_datasets'], columns=comparison_methods)
    p_values = pd.DataFrame(index=['across_datasets'], columns=comparison_methods)
    
    # Perform paired t-test for each comparison method
    for comp_method in comparison_methods:
        if comp_method not in df.index:
            continue
        
        comp_data = df.loc[comp_method].values
        
        # Check if we have enough paired samples (non-NaN values)
        valid_indices = ~np.isnan(target_data) & ~np.isnan(comp_data)
        valid_target = target_data[valid_indices]
        valid_comp = comp_data[valid_indices]
        
        if len(valid_target) < 2:  # Need at least 2 samples for t-test
            continue
            
        # Perform paired t-test
        t_stat, p_val = stats.ttest_rel(valid_target, valid_comp)
        
        # We want t_stat to be positive if target is better
        if np.mean(valid_target) < np.mean(valid_comp):
            t_stat = -t_stat
        
        t_values.at['across_datasets', comp_method] = t_stat
        p_values.at['across_datasets', comp_method] = p_val
    
    # Transpose results to have methods as rows
    t_values = t_values.transpose()
    p_values = p_values.transpose()
    
    return t_values, p_values

def format_t_test_results(t_values, p_values):
    """
    Format t-test results with significance markers
    """
    # Create result dataframe
    result_df = t_values.copy()
    
    # Format t-values with significance markers
    for method in result_df.index:
        for model_dataset in result_df.columns:
            t_val = t_values.at[method, model_dataset]
            p_val = p_values.at[method, model_dataset]
            
            if pd.isna(t_val) or pd.isna(p_val):
                result_df.at[method, model_dataset] = ''
                continue
                
            # Format t-value with significance markers
            if t_val > 0:  # LDS3Pool_Hilbert is better
                if p_val < 0.01:
                    result_df.at[method, model_dataset] = f"{t_val:.3f}**"
                elif p_val < 0.05:
                    result_df.at[method, model_dataset] = f"{t_val:.3f}*"
                else:
                    result_df.at[method, model_dataset] = f"{t_val:.3f}"
            else:  # Other method is better
                result_df.at[method, model_dataset] = f"{t_val:.3f}"
    
    # Rename rows
    result_df.index = [index.replace('BDSLocPool2d', 'LDS3Pool') for index in result_df.index]
    
    return result_df

def format_across_datasets_results(t_values, p_values):
    """
    Format t-test results for across-datasets comparison
    """
    # Create result dataframe
    result_df = pd.DataFrame(index=t_values.index, columns=['t_value', 'p_value', 'significance'])
    
    # Format results
    for method in result_df.index:
        t_val = t_values.at[method, 'across_datasets']
        p_val = p_values.at[method, 'across_datasets']
        
        if pd.isna(t_val) or pd.isna(p_val):
            result_df.at[method, 't_value'] = ''
            result_df.at[method, 'p_value'] = ''
            result_df.at[method, 'significance'] = ''
            continue
        
        # Add formatted t and p values
        result_df.at[method, 't_value'] = f"{t_val:.3f}"
        result_df.at[method, 'p_value'] = f"{p_val:.3f}"
        
        # Add significance markers
        if t_val > 0:  # LDS3Pool_Hilbert is better
            if p_val < 0.01:
                result_df.at[method, 'significance'] = '**'
            elif p_val < 0.05:
                result_df.at[method, 'significance'] = '*'
            else:
                result_df.at[method, 'significance'] = ''
        else:  # Other method is better
            if p_val < 0.01:
                result_df.at[method, 'significance'] = '(-**)'
            elif p_val < 0.05:
                result_df.at[method, 'significance'] = '(-*)'
            else:
                result_df.at[method, 'significance'] = ''
    
    # Rename rows
    result_df.index = [index.replace('BDSLocPool2d', 'LDS3Pool') for index in result_df.index]
    
    return result_df

def create_heatmap(df, title, output_filename, cmap="RdBu_r"):
    """
    Create heatmap for t-values
    """
    # Handle string values in dataframe
    # Create a numeric dataframe for the heatmap colors
    df_numeric = df.applymap(lambda x: float(x.replace('*', '')) if isinstance(x, str) and x else np.nan)
    
    # Create a mask for NaN values
    mask = df_numeric.isna()
    
    plt.figure(figsize=(14, 8))
    
    # Use masked heatmap to hide NaN values
    ax = sns.heatmap(df_numeric, annot=df, fmt='', cmap=cmap, linewidths=.5, 
                    mask=mask, annot_kws={"size": 9}, center=0)
    
    # Format x labels to split model and dataset names into two lines
    xlabels = []
    for col in df.columns:
        parts = col.split('_')
        if len(parts) == 2:
            xlabels.append(f"{parts[0]}\n{parts[1]}")
        else:
            xlabels.append(col)
    
    ax.set_xticklabels(xlabels, fontsize=10, rotation=0)
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save chart
    plt.savefig(f"{output_filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_filename}.svg", format='svg', bbox_inches='tight')
    print(f"Heatmap saved to {output_filename}.png")
    plt.close()

def create_across_datasets_bar_chart(result_df, output_filename):
    """
    Create bar chart for t-test results across datasets
    """
    plt.figure(figsize=(10, 6))
    
    # Extract t-values as floats
    t_values = result_df['t_value'].astype(float)
    
    # Define colors based on significance
    colors = []
    for method in result_df.index:
        sig = result_df.at[method, 'significance']
        if '**' in sig:
            colors.append('darkgreen' if not '(-' in sig else 'darkred')
        elif '*' in sig:
            colors.append('green' if not '(-' in sig else 'red')
        else:
            colors.append('gray')
    
    # Create bar chart
    bars = plt.bar(result_df.index, t_values, color=colors)
    
    # Add zero line
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.ylabel('t-value', fontsize=12)
    plt.title('Paired t-test: LDS3Pool_Hilbert vs Other Methods Across All Datasets\n(* p<0.05, ** p<0.01)', fontsize=14)
    
    # Add annotations above bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        sig = result_df.iloc[i]['significance']
        if sig:
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + 0.1 if height > 0 else height - 0.3,
                    sig, ha='center', va='bottom', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save chart
    plt.savefig(f"{output_filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_filename}.svg", format='svg', bbox_inches='tight')
    print(f"Bar chart saved to {output_filename}.png")
    plt.close()

def main():
    # Input file
    date = "0419_1"
    extracted_data_file = f"output/runs_{date}/extracted_data_{date}.csv"
    
    # Output directory
    output_dir = f"output/runs_{date}/analysis"
    
    # Perform paired t-tests for individual model_dataset combinations
    t_values, p_values = perform_paired_t_tests(extracted_data_file)
    
    # Format results
    result_df = format_t_test_results(t_values, p_values)
    
    # Save results to CSV
    result_path = os.path.join(output_dir, "t_test_results.csv")
    result_df.to_csv(result_path)
    print(f"T-test results saved to {result_path}")
    
    # Create heatmap
    create_heatmap(
        result_df, 
        "Paired t-tests: LDS3Pool_Hilbert vs. Other Methods\n(* p<0.05, ** p<0.01)",
        f"{output_dir}/t_test_heatmap"
    )
    
    # Also save raw t-values and p-values
    t_values.to_csv(os.path.join(output_dir, "t_values_raw.csv"))
    p_values.to_csv(os.path.join(output_dir, "p_values_raw.csv"))
    
    # Perform paired t-tests across datasets using accuracy_mean table
    accuracy_mean_path = os.path.join(output_dir, "accuracy_mean.csv")
    if os.path.exists(accuracy_mean_path):
        print("Performing paired t-tests across datasets...")
        across_t_values, across_p_values = perform_paired_t_tests_across_datasets(accuracy_mean_path)
        
        if across_t_values is not None:
            # Format results
            across_result_df = format_across_datasets_results(across_t_values, across_p_values)
            
            # Save results to CSV
            across_result_path = os.path.join(output_dir, "t_test_across_datasets.csv")
            across_result_df.to_csv(across_result_path)
            print(f"Across-datasets t-test results saved to {across_result_path}")
            
            # Create bar chart
            create_across_datasets_bar_chart(
                across_result_df,
                f"{output_dir}/t_test_across_datasets_bar"
            )
    else:
        print(f"Accuracy mean table not found at {accuracy_mean_path}")

if __name__ == "__main__":
    main() 