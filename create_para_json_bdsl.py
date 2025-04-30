import json
import os
from datetime import datetime

def create_para_json_bdsl():
    # 创建保存参数的目录
    timestamp = datetime.now().strftime("%Y%m%d")
    para_dir = f"para_json"
    os.makedirs(para_dir, exist_ok=True)
    
    # 定义模型和数据集组合
    model_dataset_dict = {
        'LeNet5': [
            "MNIST", 
            "FashionMNIST", 
            "CIFAR10", 
            "CIFAR100"
        ],
        'VGG11':[
            "CIFAR10", 
            "CIFAR100",
            "OxfordIIITPet",
            "STL10", 
        ]
    }

    k_list = [
        1, 
        2, 
        3, 
        4
    ]
    
    model_dataset_pairs = []
    for model, datasets in model_dataset_dict.items():
        for dataset in datasets:
            model_dataset_pairs.append(model + '_' + dataset)
    
    # 定义其他参数范围
    dim_reductions = ['hilbert',
                    #    'pca', 'C'
    ]
    val_pools = [
        'SameAsTrain',
        # 'MaxPool', 
        # 'AvgPool'
    ]
    random_seeds = [7, 
                    42, 123, 1309, 5287, 31415
    ]
    trans_nums = [
        # 'e', 
        # 'pi', 
        'golden'
    ]
    
    # 生成所有参数组合
    para_list = []
    index = 0
    for model_dataset in model_dataset_pairs:
        for dim_reduction in dim_reductions:
            for val_pool in val_pools:
                for k in k_list:
                    for random_seed in random_seeds:
                        for trans_num in trans_nums:
                            para = {
                                'pooling_method': 'BDSLocPool2d',
                                'model_dataset': model_dataset,
                                'dim_reduction': dim_reduction,
                                'val_pool': val_pool,
                                'random_seed': random_seed,
                                'trans_num': trans_num,
                                'k': k,
                                'index': index
                            }
                            para_list.append(para)
                            index += 1
    # 保存参数到JSON文件
    json_path = os.path.join(para_dir, f"para_bdsl_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(para_list, f, indent=4)
    
    print(f"参数已保存到: {json_path}")
    print(f"共生成 {len(para_list)} 组参数")

if __name__ == "__main__":
    create_para_json_bdsl()
