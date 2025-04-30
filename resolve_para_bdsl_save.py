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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import torch.optim as optim
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import argparse

config_file = 'para_json/para_bdsl_20250426.json'
exp_type = 'BDSL/'
model_save_dir = 'models_0426/'
runs_dir = 'runs_0426/'

with open(config_file, 'r') as f:
    para_json_list = json.load(f)

print('Number of para_json:', len(para_json_list))

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
        trans_num = param_dict['trans_num'],
        k = param_dict['k'] if 'k' in param_dict else 2
    )

# 创建用于保存模型的目录
os.makedirs(model_save_dir, exist_ok=True)

# para_json = para_json_list[100]
for para_json in para_json_list:

    para_json_idx = para_json['index']

    if para_json_idx < 72:
        continue

    print(para_json_idx, para_json)

    model_name, dataset_name = para_json['model_dataset'].split('_')
    pooling_method = para_json['pooling_method']

    '''
    create dataloader
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)


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
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # print(summary(model, input_size=INPUT_SIZE, mode='train',
    #                 col_names=['input_size', 'output_size', 'num_params', 'trainable'], row_settings=['var_names'], verbose=0
    # ))

    EPOCH_NUM = 150 if dataset_name == 'OxfordIIITPet' else 100

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    accuracy = Accuracy(task='multiclass', num_classes=model.num_classes)
    accuracy = accuracy.to(device)

    # timestamp = datetime.now().strftime("%Y-%m-%d")
    experiment_name = exp_type + str(para_json_idx).rjust(4, '0')
    log_dir = os.path.join(runs_dir, experiment_name)
    writer = SummaryWriter(log_dir)
    
    # 创建模型保存目录
    save_dir = os.path.join(model_save_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化用于跟踪最佳验证准确率的变量
    best_val_acc = 0.0

    # 设置torch的随机数种子
    torch.manual_seed(para_json["random_seed"])

    for epoch in tqdm(range(EPOCH_NUM)):

        epoch_start_time = time.time()

        # Training loop
        train_loss, train_acc = 0.0, 0.0
        all_train_preds = []
        all_train_labels = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            model.train()
            
            y_pred = model(X)
            
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            
            acc = accuracy(y_pred, y)
            train_acc += acc
            
            optimizer.zero_grad()
            start_time = time.time()
            loss.backward()
            optimizer.step()

            all_train_preds.extend(torch.argmax(y_pred, dim=1).cpu().numpy())
            all_train_labels.extend(y.cpu().numpy())
                
        scheduler.step()
            
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
            
        # Validation loop
        val_loss, val_acc = 0.0, 0.0
        all_val_preds = []
        all_val_labels = []
        model.eval()
        with torch.inference_mode():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                
                y_pred = model(X)
                
                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                
                acc = accuracy(y_pred, y)
                val_acc += acc

                all_val_preds.extend(torch.argmax(y_pred, dim=1).cpu().numpy())
                all_val_labels.extend(y.cpu().numpy())
                
            val_loss /= len(test_loader)
            val_acc /= len(test_loader)

        epoch_time = time.time() - epoch_start_time

        train_precision = precision_score(all_train_labels, all_train_preds, average='macro')
        train_recall = recall_score(all_train_labels, all_train_preds, average='macro')
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
        val_precision = precision_score(all_val_labels, all_val_preds, average='macro')
        val_recall = recall_score(all_val_labels, all_val_preds, average='macro')
        val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')

        writer.add_scalars(main_tag="loss", tag_scalar_dict={"train": train_loss, "val": val_loss}, global_step=epoch)
        writer.add_scalars(main_tag="accuracy", tag_scalar_dict={"train": train_acc, "val": val_acc}, global_step=epoch)
        writer.add_scalars(main_tag="precision", tag_scalar_dict={"train": train_precision, "val": val_precision}, global_step=epoch)
        writer.add_scalars(main_tag="recall", tag_scalar_dict={"train": train_recall, "val": val_recall}, global_step=epoch)
        writer.add_scalars(main_tag="f1", tag_scalar_dict={"train": train_f1, "val": val_f1}, global_step=epoch)
        writer.add_scalar(tag="epoch_time", scalar_value=epoch_time, global_step=epoch)        

        print(f"Epoch: {epoch}| Epoch Time: {epoch_time:.2f}s\n"
            f"Train loss: {train_loss: .6f}| Train acc: {train_acc: .6f}| Val loss: {val_loss: .6f}| Val acc: {val_acc: .6f}|\n"
            f"Train Precision: {train_precision:.6f}| Train Recall: {train_recall:.6f}| Train F1: {train_f1:.6f}|\n"
            f"Val Precision: {val_precision:.6f}| Val Recall: {val_recall:.6f}| Val F1: {val_f1:.6f}|")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存模型状态字典
            model_save_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'para_json': para_json,  # 保存配置信息
            }, model_save_path)
            print(f"保存最佳模型至 {model_save_path}，验证准确率: {val_acc:.6f}")
            
            # 保存模型配置信息
            config_save_path = os.path.join(save_dir, "model_config.json")
            config_data = {
                'epoch': epoch,
                'val_acc': float(val_acc),
                'train_acc': float(train_acc),
                'best_epoch': epoch,
            }
            # 将para_json中的所有内容添加到config_data中
            for key, value in para_json.items():
                # 确保PyTorch张量和其他特殊对象被转换为可序列化格式
                if isinstance(value, torch.Tensor):
                    config_data[key] = value.item() if value.numel() == 1 else value.tolist()
                else:
                    config_data[key] = value
                    
            with open(config_save_path, 'w') as f:
                json.dump(config_data, f, indent=4)
                
            writer.add_text("best_model", f"Epoch {epoch}: 保存最佳模型，验证准确率: {val_acc:.6f}", global_step=epoch)
