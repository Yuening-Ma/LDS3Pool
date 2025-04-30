import json
import itertools
import os
from datetime import datetime

para_dir = f"para_json"
timestamp = datetime.now().strftime("%Y%m%d")

model_dataset_dict = {
                        'LeNet5':[
                                "MNIST", 
                                "FashionMNIST", 
                                "CIFAR10", 
                                "CIFAR100"
                        ],
                        'VGG11':[
                                "CIFAR10", 
                                "CIFAR100", 
                                "STL10", 
                                "OxfordIIITPet"
                        ]
                      }

model_dataset_pairs = []
for model, datasets in model_dataset_dict.items():
    for dataset in datasets:
        model_dataset_pairs.append(model + '_' + dataset)

del model, datasets, dataset

other_pooling_methods = [
                        "MaxPool", 
                        "AvgPool", 
                        "S3Pool", 
                        ]
random_seeds = [7, 
                # 42, 123, 1309, 5287, 31415
]

'''
其他池化方法的实验
'''
base_combinations = []
idx = 0
for model_dataset, pooling_method in itertools.product(model_dataset_pairs, other_pooling_methods):
    param_dict = {"model_dataset": model_dataset, 
                  "pooling_method": pooling_method}
    if pooling_method == 'S3Pool':
        for seed in random_seeds:
            param_dict_copy = param_dict.copy()
            param_dict_copy["random_seed"] = seed
            param_dict_copy["index"] = idx
            base_combinations.append(param_dict_copy)
            idx += 1
    else:
        for seed in random_seeds:
            param_dict_copy = param_dict.copy()
            param_dict_copy["random_seed"] = seed
            param_dict_copy["index"] = idx
            base_combinations.append(param_dict_copy)
            idx += 1
    

json_path = os.path.join(para_dir, f"para_base_{timestamp}.json")
with open(json_path, "w") as f:
    json.dump(base_combinations, f, indent=4)

print(f"Generated {len(base_combinations)} base parameter combinations. Saved to {json_path}")

