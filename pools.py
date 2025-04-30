import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('gilbert')
from gilbert2d import gilbert2d
import numpy as np
from sklearn.decomposition import PCA


class BDSLocPool2d(nn.Module):
    def __init__(self, input_size=224, pool_size=2, dim_reduction='hilbert', val_pool='SameAsTrain', random_seed=7, trans_num='e', k=2):
        super(BDSLocPool2d, self).__init__()
        self.pool_size = pool_size
        self.dim_reduction = dim_reduction
        self.val_pool = val_pool
        self.trans_num = trans_num
        self.rng = torch.Generator().manual_seed(random_seed)
        self.k = k
        # 预先生成BDS序列
        self.bds_seq = self._generate_bds_sequence(input_size * input_size)
        
        # 预先生成点
        if dim_reduction == 'hilbert':
            self.points = np.array(list(gilbert2d(input_size, input_size)))
        elif dim_reduction == 'pca':
            y, x = np.mgrid[0:input_size, 0:input_size]
            points = np.column_stack((x.flatten(), y.flatten()))
            pca = PCA(n_components=1)
            points_1d = pca.fit_transform(points)
            sorted_indices = np.argsort(points_1d.flatten())
            self.points = points[sorted_indices]
        elif dim_reduction == 'C':
            # 按行优先顺序排列所有点
            y, x = np.mgrid[0:input_size, 0:input_size]
            points = np.column_stack((x.flatten(), y.flatten()))
            self.points = points
        else:
            raise ValueError(f"Unsupported dim_reduction: {self.dim_reduction}")
            
        # 转换为tensor，并移动到GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.points = torch.from_numpy(self.points).long().to(device)
        self.bds_seq = torch.from_numpy(self.bds_seq).long().to(device)
        self.linear_indices = self.points[:, 1] * input_size + self.points[:, 0]

        # 预计算验证掩码
        self.val_mask = None
        if val_pool == 'SameAsTrain':
            self.val_mask = torch.zeros(input_size * input_size, dtype=torch.bool, device='cuda')
            num_samples = int(input_size//self.pool_size * input_size//self.pool_size * self.k)
            sampled_indices = self.bds_seq[0:num_samples]
            self.val_mask.scatter_(0, self.linear_indices[sampled_indices], True)
            self.val_mask = self.val_mask.view(input_size, input_size)
        
    def _generate_bds_sequence(self, n):
        if self.trans_num == 'e':
            a = np.e * np.arange(1, n+1) % 1
        elif self.trans_num == 'pi':
            a = np.pi * np.arange(1, n+1) % 1
        elif self.trans_num == 'golden':
            a = (5**0.5 - 1) / 2 * np.arange(1, n+1) % 1
        else:
            a = np.e * np.arange(1, n+1) % 1
        temp = np.argsort(a)
        r = np.argsort(temp)
        return r
        
    def forward(self, x):

        if (not self.training) and (self.val_pool != 'SameAsTrain'):
            if self.val_pool == 'AvgPool':
                x = F.avg_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)
            elif self.val_pool == 'MaxPool':
                x = F.max_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)
            else:
                raise ValueError("val_pool for BDSSpatialPool2d must be ['AvgPool', 'MaxPool', 'SameAsTrain']")
            return x

        # 第1步：填充和最大池化
        x = F.pad(x, (0, self.pool_size-1, 0, self.pool_size-1), mode='constant', value=0)

        x = F.max_pool2d(x, kernel_size=self.pool_size, stride=1) #stride=1 for maxpooling

        # 第2步：计算采样大小和选择BDS序列起始点
        N, C, H, W = x.size()
        new_H, new_W = H//self.pool_size, W//self.pool_size
        num_samples = int(new_H * new_W * self.k)
        
        # 随机选择BDS序列的起始位置
        start_idx = torch.randint(0, H*W - num_samples + 1, (1,), generator=self.rng).item()
        sampled_indices = self.bds_seq[start_idx:start_idx + num_samples]
        
        # # 第3步：获取采样点坐标
        # sampled_points = self.points[sampled_indices]  # (num_samples, 2)
        
        # 第4步：创建掩码
        if self.training:
            mask = torch.zeros(H * W, dtype=torch.bool, device=x.device)
            mask.scatter_(0, self.linear_indices[sampled_indices], True)
            mask = mask.view(H, W)
        else:
            mask = self.val_mask
        
        # 第5步：扩展掩码
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        mask = mask.expand(N, C, -1, -1)  # (N, C, H, W)
        
        # 第6步：应用掩码
        x_masked = x.clone()
        x_masked[~mask] = 0
        
        # 第7步：最终池化
        x_masked = F.max_pool2d(x_masked, kernel_size=self.pool_size, stride=self.pool_size)
        
        return x_masked


class S3Pool2d(nn.Module):
    def __init__(self, pool_size=2, maxpool=True, val_pool='MaxPool', grid_size=None, random_seed=7):
        super(S3Pool2d, self).__init__()
        self.pool_size = pool_size
        self.maxpool = maxpool
        self.val_pool = val_pool
        self.grid_size = grid_size if grid_size else pool_size
        self.rng = torch.Generator().manual_seed(random_seed)

    def forward(self, input):

        if not self.training:
            if self.val_pool == 'AvgPool':
                input = F.avg_pool2d(input, kernel_size=self.pool_size, stride=self.pool_size)
            elif self.val_pool == 'MaxPool':
                input = F.max_pool2d(input, kernel_size=self.pool_size, stride=self.pool_size)
            else:
                raise ValueError("val_pool for S3Pool2d must be ['AvgPool', 'MaxPool']")
            return input


        if self.maxpool:
            input = F.pad(input, (0, self.pool_size-1, 0, self.pool_size-1), mode='constant', value=0)
            input = F.max_pool2d(input, kernel_size=self.pool_size, stride=1) #stride=1 for maxpooling

        b, c, h, w = input.shape

        n_w, n_h = h // self.grid_size, w // self.grid_size
        n_sample_per_grid = self.grid_size // self.pool_size

        k_to_add_w = w // self.pool_size - n_w * n_sample_per_grid
        k_to_add_h = w // self.pool_size - n_h * n_sample_per_grid

        idx_w = []
        idx_h = []

        for i in range(n_w):
            offset = self.grid_size * i
            this_n = self.grid_size if i < n_w - 1 else h - offset
            if i < k_to_add_w:
                this_idx = torch.sort(torch.randperm(this_n, generator=self.rng)[:n_sample_per_grid+1])[0]
            else:
                this_idx = torch.sort(torch.randperm(this_n, generator=self.rng)[:n_sample_per_grid])[0]
            idx_w.append(offset + this_idx)

        for i in range(n_h):
            offset = self.grid_size * i
            this_n = self.grid_size if i < n_h - 1 else w - offset
            if i < k_to_add_h:
                this_idx = torch.sort(torch.randperm(this_n, generator=self.rng)[:n_sample_per_grid+1])[0]
            else:
                this_idx = torch.sort(torch.randperm(this_n, generator=self.rng)[:n_sample_per_grid])[0]
            idx_h.append(offset + this_idx)

        idx_w = torch.cat(idx_w)
        idx_h = torch.cat(idx_h)

        # print(idx_h)
        # print(idx_w)

        output = input[:,:,idx_w][:,:,:,idx_h]

        return output

# batch_size = 2
# channels = 3
# height = 10
# width =10
# pool_size = 2
# grid_size = 5
# maxpool = True

# layer = S3Pool2d(pool_size=pool_size, maxpool=maxpool, grid_size=grid_size)
# input_tensor = torch.randn(batch_size, channels, height, width)
# output_tensor = layer(input_tensor)

# expected_shape = (batch_size, channels, height // pool_size, width // pool_size)
# print(f"Expected shape: {expected_shape}")
# print(f"Output shape: {output_tensor.shape}")