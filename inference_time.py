import json
from models import LeNet5, VGG
from pools import S3Pool2d, BDSLocPool2d
from torchvision import datasets
from torchvision import transforms
from torch import nn
import torch
from torchinfo import summary
from torchmetrics import Accuracy
from tqdm import tqdm
import time
import os
import pandas as pd
import numpy as np

# 设置配置文件和模型路径
config_file = 'para_json/para_bdsl_20250424.json'
output_path = 'output/'

with open(config_file, 'r') as f:
    para_json_list = json.load(f)

print('Number of para_json:', len(para_json_list))

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

'''
create pool layers and model
'''
def make_bdslpool_layer(param_dict, input_size=None):
    
    return BDSLocPool2d(
        input_size=input_size,
        pool_size=2,  # 固定为2
        dim_reduction=param_dict['dim_reduction'],
        val_pool=param_dict['val_pool'],
        random_seed=param_dict['random_seed'],
        trans_num = param_dict['trans_num']
    )

# 获取GPU内存使用情况函数
def get_gpu_memory_usage():
    if torch.cuda.is_available():
        # 返回以MB为单位的已分配内存
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

# 创建用于存储训练时间和推理时间的数据结构
train_times = {}
inference_times = {}
# 创建用于存储训练和推理batch数量的数据结构
train_batches = {}
inference_batches = {}

for para_json in para_json_list:
    para_json_idx = para_json['index']
    print(f"\n处理参数配置 {para_json_idx}:", para_json)

    model_name, dataset_name = para_json['model_dataset'].split('_')
    pooling_method = para_json['pooling_method']
    dim_reduction = para_json.get('dim_reduction', 'N/A')  # 获取dim_reduction，如果不存在则为N/A
    random_seed = para_json.get('random_seed', 0)  # 获取随机种子

    '''
    创建数据加载器
    '''
    BATCH_SIZE = 32

    if dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(root="./datasets/", train=True, download=False, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root="./datasets", train=False, download=False, transform=transforms.ToTensor())
        INPUT_SIZE = (BATCH_SIZE, 1, 28, 28)
    elif dataset_name == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(root="./datasets/", train=True, download=False, transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST(root="./datasets", train=False, download=False, transform=transforms.ToTensor())
        INPUT_SIZE = (BATCH_SIZE, 1, 28, 28)
    elif dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root="./datasets/", train=True, download=False, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(root="./datasets", train=False, download=False, transform=transforms.ToTensor())
        INPUT_SIZE = (BATCH_SIZE, 3, 32, 32)
    elif dataset_name == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root="./datasets/", train=True, download=False, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR100(root="./datasets", train=False, download=False, transform=transforms.ToTensor())
        INPUT_SIZE = (BATCH_SIZE, 3, 32, 32)
    elif dataset_name == 'STL10':
        train_dataset = datasets.STL10(root="./datasets/", split='train', download=False, transform=transforms.ToTensor())
        test_dataset = datasets.STL10(root="./datasets", split='test', download=False, transform=transforms.ToTensor())
        INPUT_SIZE = (BATCH_SIZE, 3, 96, 96)
    elif dataset_name == 'OxfordIIITPet':
        # 2分类而不是37分类
        train_dataset = datasets.OxfordIIITPet(root="./datasets/", 
                                            split='trainval', 
                                            target_types='binary-category',
                                            download=False, 
                                            transform=transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor()
                                            ]))
        test_dataset = datasets.OxfordIIITPet(root="./datasets", 
                                            split='test', 
                                            target_types='binary-category',
                                            download=False, 
                                            transform=transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor()
                                            ]))
        INPUT_SIZE = (BATCH_SIZE, 3, 224, 224)

    # 创建数据加载器，丢弃不完整的batch
    drop_last = True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                              num_workers=8, drop_last=drop_last)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                             num_workers=8, drop_last=drop_last)

    '''
    创建模型
    '''
    if pooling_method in ['MaxPool', 'AvgPool']:
        if pooling_method == 'MaxPool':
            pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
            pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pooling_method == 'AvgPool':
            pool_1 = nn.AvgPool2d(kernel_size=2, stride=2)
            pool_2 = nn.AvgPool2d(kernel_size=2, stride=2)

        if model_name == 'LeNet5':
            if dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
                model = LeNet5(num_classes=10, pool_1=pool_1, pool_2=pool_2, in_channels=1, padding=2)
            elif dataset_name == 'CIFAR10':
                model = LeNet5(num_classes=10, pool_1=pool_1, pool_2=pool_2, in_channels=3, padding=0)
            elif dataset_name == 'CIFAR100':
                model = LeNet5(num_classes=100, pool_1=pool_1, pool_2=pool_2, in_channels=3, padding=0)
            else:
                raise ValueError('dataset_name for LeNet could only be ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]')

        elif model_name == 'VGG11':

            if pooling_method == 'MaxPool':
                pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
                pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
                pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)
            elif pooling_method == 'AvgPool':
                pool_3 = nn.AvgPool2d(kernel_size=2, stride=2)
                pool_4 = nn.AvgPool2d(kernel_size=2, stride=2)
                pool_5 = nn.AvgPool2d(kernel_size=2, stride=2)
            if dataset_name == 'CIFAR10':
                model = VGG(pool_list=[pool_1, pool_2, pool_3, pool_4, pool_5], num_classes=10, initial_image_size=32)
            elif dataset_name == 'CIFAR100':
                model = VGG(pool_list=[pool_1, pool_2, pool_3, pool_4, pool_5], num_classes=100, initial_image_size=32)
            elif dataset_name == 'STL10':
                model = VGG(pool_list=[pool_1, pool_2, pool_3, pool_4, pool_5], num_classes=10, initial_image_size=96)
            elif dataset_name == 'OxfordIIITPet':
                model = VGG(pool_list=[pool_1, pool_2, pool_3, pool_4, pool_5], num_classes=2, initial_image_size=224)
            else:
                raise ValueError('dataset_name for VGG could only be ["CIFAR10", "CIFAR", "STL10", "OxfordIIITPet"]')

    elif pooling_method == 'S3Pool':

        if model_name == 'LeNet5':
            if dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
                G1 = 2
                G2 = 2
                seed = para_json["random_seed"]
                pool_1 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G1, random_seed=seed)
                pool_2 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G2, random_seed=seed)
                model = LeNet5(num_classes=10, pool_1=pool_1, pool_2=pool_2, in_channels=1, padding=2)
            elif dataset_name == 'CIFAR10':
                G1 = 2
                G2 = 2
                seed = para_json["random_seed"]
                pool_1 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G1, random_seed=seed)
                pool_2 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G2, random_seed=seed)
                model = LeNet5(num_classes=10, pool_1=pool_1, pool_2=pool_2, in_channels=3, padding=0)
            elif dataset_name == 'CIFAR100':
                G1 = 2
                G2 = 2
                seed = para_json["random_seed"]
                pool_1 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G1, random_seed=seed)
                pool_2 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G2, random_seed=seed)
                model = LeNet5(num_classes=100, pool_1=pool_1, pool_2=pool_2, in_channels=3, padding=0)
            else:
                raise ValueError('dataset_name for LeNet could only be ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]')

        elif model_name == 'VGG11':
            if dataset_name == 'CIFAR10':
                G1 = 16
                G2 = 8
                G3 = 4
                G4 = 2
                G5 = 2
                seed = para_json["random_seed"]
                pool_1 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G1, random_seed=seed)
                pool_2 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G2, random_seed=seed)
                pool_3 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G3, random_seed=seed)
                pool_4 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G4, random_seed=seed)
                pool_5 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G5, random_seed=seed)
                model = VGG(pool_list=[pool_1, pool_2, pool_3, pool_4, pool_5], num_classes=10, initial_image_size=32)
            elif dataset_name == 'CIFAR100':
                G1 = 16
                G2 = 8
                G3 = 4
                G4 = 2
                G5 = 2
                seed = para_json["random_seed"]
                pool_1 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G1, random_seed=seed)
                pool_2 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G2, random_seed=seed)
                pool_3 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G3, random_seed=seed)
                pool_4 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G4, random_seed=seed)
                pool_5 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G5, random_seed=seed)
                model = VGG(pool_list=[pool_1, pool_2, pool_3, pool_4, pool_5], num_classes=100, initial_image_size=32)
            elif dataset_name == 'STL10':
                G1 = 32
                G2 = 16
                G3 = 8
                G4 = 4
                G5 = 2
                seed = para_json["random_seed"]
                pool_1 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G1, random_seed=seed)
                pool_2 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G2, random_seed=seed)
                pool_3 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G3, random_seed=seed)
                pool_4 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G4, random_seed=seed)
                pool_5 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G5, random_seed=seed)
                model = VGG(pool_list=[pool_1, pool_2, pool_3, pool_4, pool_5], num_classes=10, initial_image_size=96)
            elif dataset_name == 'OxfordIIITPet':
                G1 = 56
                G2 = 28
                G3 = 14
                G4 = 7
                G5 = 7
                seed = para_json["random_seed"]
                pool_1 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G1, random_seed=seed)
                pool_2 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G2, random_seed=seed)
                pool_3 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G3, random_seed=seed)
                pool_4 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G4, random_seed=seed)
                pool_5 = S3Pool2d(pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=G5, random_seed=seed)
                model = VGG(pool_list=[pool_1, pool_2, pool_3, pool_4, pool_5], num_classes=2, initial_image_size=224)
            else:
                raise ValueError('dataset_name for VGG could only be ["CIFAR10", "CIFAR", "STL10", "OxfordIIITPet"]')
               
    elif pooling_method == 'BDSLocPool2d':
        if model_name == 'LeNet5':
            if dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
                pool_1 = make_bdslpool_layer(param_dict=para_json, input_size=28)
                pool_2 = make_bdslpool_layer(param_dict=para_json, input_size=10)
                model = LeNet5(num_classes=10, pool_1=pool_1, pool_2=pool_2, in_channels=1, padding=2)
            elif dataset_name == 'CIFAR10':
                pool_1 = make_bdslpool_layer(param_dict=para_json, input_size=28)
                pool_2 = make_bdslpool_layer(param_dict=para_json, input_size=10)
                model = LeNet5(num_classes=10, pool_1=pool_1, pool_2=pool_2, in_channels=3, padding=0)
            elif dataset_name == 'CIFAR100':
                pool_1 = make_bdslpool_layer(param_dict=para_json, input_size=28)
                pool_2 = make_bdslpool_layer(param_dict=para_json, input_size=10)
                model = LeNet5(num_classes=100, pool_1=pool_1, pool_2=pool_2, in_channels=3, padding=0)
            else:
                raise ValueError('dataset_name for LeNet could only be ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]')

        elif model_name == 'VGG11':
            if dataset_name == 'CIFAR10':
                pool_1 = make_bdslpool_layer(param_dict=para_json, input_size=32)
                pool_2 = make_bdslpool_layer(param_dict=para_json, input_size=16)
                pool_3 = make_bdslpool_layer(param_dict=para_json, input_size=8)
                pool_4 = make_bdslpool_layer(param_dict=para_json, input_size=4)
                pool_5 = make_bdslpool_layer(param_dict=para_json, input_size=2)
                model = VGG(pool_list=[pool_1, pool_2, pool_3, pool_4, pool_5], num_classes=10, initial_image_size=32)
            elif dataset_name == 'CIFAR100':
                pool_1 = make_bdslpool_layer(param_dict=para_json, input_size=32)
                pool_2 = make_bdslpool_layer(param_dict=para_json, input_size=16)
                pool_3 = make_bdslpool_layer(param_dict=para_json, input_size=8)
                pool_4 = make_bdslpool_layer(param_dict=para_json, input_size=4)
                pool_5 = make_bdslpool_layer(param_dict=para_json, input_size=2)
                model = VGG(pool_list=[pool_1, pool_2, pool_3, pool_4, pool_5], num_classes=100, initial_image_size=32)
            elif dataset_name == 'STL10':
                pool_1 = make_bdslpool_layer(param_dict=para_json, input_size=96)
                pool_2 = make_bdslpool_layer(param_dict=para_json, input_size=48)
                pool_3 = make_bdslpool_layer(param_dict=para_json, input_size=24)
                pool_4 = make_bdslpool_layer(param_dict=para_json, input_size=12)
                pool_5 = make_bdslpool_layer(param_dict=para_json, input_size=6)
                model = VGG(pool_list=[pool_1, pool_2, pool_3, pool_4, pool_5], num_classes=10, initial_image_size=96)
            elif dataset_name == 'OxfordIIITPet':
                pool_1 = make_bdslpool_layer(param_dict=para_json, input_size=224)
                pool_2 = make_bdslpool_layer(param_dict=para_json, input_size=112)
                pool_3 = make_bdslpool_layer(param_dict=para_json, input_size=56)
                pool_4 = make_bdslpool_layer(param_dict=para_json, input_size=28)
                pool_5 = make_bdslpool_layer(param_dict=para_json, input_size=14)
                model = VGG(pool_list=[pool_1, pool_2, pool_3, pool_4, pool_5], num_classes=2, initial_image_size=224)
            else:
                raise ValueError('dataset_name for VGG could only be ["CIFAR10", "CIFAR", "STL10", "OxfordIIITPet"]')
        
    # 设置设备并将模型移至设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 准备模型训练
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    accuracy = Accuracy(task='multiclass', num_classes=model.num_classes).to(device)

    # 创建method_key
    method_key = f"{pooling_method}_{dim_reduction}" if pooling_method == 'BDSLocPool2d' else pooling_method
    
    # 初始化数据结构
    key = para_json['model_dataset']
    if key not in train_times:
        train_times[key] = {}
    if key not in inference_times:
        inference_times[key] = {}
    if key not in train_batches:
        train_batches[key] = {}
    if key not in inference_batches:
        inference_batches[key] = {}
    
    train_times[key][method_key] = []
    inference_times[key][method_key] = []
    train_batches[key][method_key] = 0  # 记录训练集的batch数量
    inference_batches[key][method_key] = 0  # 记录验证集的batch数量

    # ==================== 测量训练时间 ====================
    print(f"\n开始测量训练时间...")

    # 设置总epoch数为7，第一个为预热
    EPOCHS = 7
    
    # 获取训练集的batch数量
    num_train_batches = len(train_loader)
    train_batches[key][method_key] = num_train_batches
    print(f"训练集batch数量: {num_train_batches}")
    
    for epoch in range(EPOCHS):
        model.train()  # 设置为训练模式
        epoch_start_time = time.time()
        
        # 训练一个epoch
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            # 前向传播
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_time = time.time() - epoch_start_time
        
        # 跳过第一个epoch（预热）
        if epoch > 0:
            train_times[key][method_key].append(epoch_time)
            print(f"Epoch {epoch} 训练时间: {epoch_time:.4f}s")
    
    # ==================== 测量推理时间 ====================
    print(f"\n开始测量推理时间...")
    
    # 获取验证集的batch数量
    num_inference_batches = len(test_loader)
    inference_batches[key][method_key] = num_inference_batches
    print(f"验证集batch数量: {num_inference_batches}")
    
    # 进行6次完整的推理过程
    for i in range(6):
        model.eval()  # 设置为评估模式
        inference_start_time = time.time()
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                
                # 模型推理
                _ = model(X)
        
        inference_time = time.time() - inference_start_time
        inference_times[key][method_key].append(inference_time)
        print(f"推理 {i+1} 时间: {inference_time:.4f}s")

# 创建和保存训练时间数据框
train_df = pd.DataFrame()
train_avg_df = pd.DataFrame()  # 新增：平均每个epoch的训练时间

for model_dataset, methods in train_times.items():
    for method_key, times in methods.items():
        if model_dataset not in train_df:
            train_df[model_dataset] = pd.Series(dtype=float)
            train_avg_df[model_dataset] = pd.Series(dtype=float)
        
        # 原始时间数据
        train_df.loc[method_key, model_dataset] = ", ".join([f"{t:.6f}" for t in times])
        
        # 平均每个epoch的训练时间（秒）
        avg_epoch_time = np.mean(times)
        train_avg_df.loc[method_key, model_dataset] = f"{avg_epoch_time:.6f}"

# 创建和保存推理时间数据框
inference_df = pd.DataFrame()
inference_avg_df = pd.DataFrame()  # 新增：平均每个batch的推理时间

for model_dataset, methods in inference_times.items():
    for method_key, times in methods.items():
        if model_dataset not in inference_df:
            inference_df[model_dataset] = pd.Series(dtype=float)
            inference_avg_df[model_dataset] = pd.Series(dtype=float)
        
        # 原始时间数据
        inference_df.loc[method_key, model_dataset] = ", ".join([f"{t:.6f}" for t in times])
        
        # 平均每个batch的推理时间（毫秒）
        num_batches = inference_batches[model_dataset][method_key]
        avg_batch_time = (np.mean(times) / num_batches) * 1000  # 转换为毫秒
        inference_avg_df.loc[method_key, model_dataset] = f"{avg_batch_time:.6f}"

# 保存结果到CSV文件
train_csv_path = os.path.join(output_path, "train_times.csv")
inference_csv_path = os.path.join(output_path, "inference_times.csv")
train_avg_csv_path = os.path.join(output_path, "train_times_avg_epoch.csv")
inference_avg_csv_path = os.path.join(output_path, "inference_times_avg_batch_ms.csv")

train_df.to_csv(train_csv_path)
inference_df.to_csv(inference_csv_path)
train_avg_df.to_csv(train_avg_csv_path)
inference_avg_df.to_csv(inference_avg_csv_path)

print(f"\n原始训练时间已保存至 {train_csv_path}")
print(f"原始推理时间已保存至 {inference_csv_path}")
print(f"平均每个epoch训练时间（秒）已保存至 {train_avg_csv_path}")
print(f"平均每个batch推理时间（毫秒）已保存至 {inference_avg_csv_path}")
