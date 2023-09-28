import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import pickle
import scipy.sparse as sparse
import time
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from metrics.cluster.accuracy import clustering_accuracy
import argparse
import random
from tqdm import tqdm
import os
import csv
import torch.nn.functional as F


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(mainall)
#         self.max_pool = nn.AdaptiveMaxPool2d(mainall)
#
#         # 利用1x1卷积代替全连接
#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, mainall, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, mainall, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else mainall
#         self.conv1 = nn.Conv2d(2, mainall, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=mainall, keepdim=True)
#         max_out, _ = torch.max(x, dim=mainall, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=mainall)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
#
# # CBAM注意力机制
# class 消融没有CBAM(nn.Module):
#     def __init__(self, channel, ratio, kernel_size=7):  # MNIST:4;FashionMNIST:8;CIFAR10.txt:8
#         super(消融没有CBAM, self).__init__()
#         self.channelattention = ChannelAttention(channel, ratio)
#         self.spatialattention = SpatialAttention(kernel_size=kernel_size)
#
#     def forward(self, x):
#         out = x.unsqueeze(2).unsqueeze(3)
#         out = out * self.channelattention(out)
#         out = out * self.spatialattention(out)
#         out = torch.squeeze(out)
#         return out

# ------------------------#
# CBAM模块的Pytorch实现
# ------------------------#

# 通道注意力模块
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio):  # 16
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // ratio
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        avgout = self.avg_pool(x).view(x.size(0), x.size(1)).unsqueeze(2).unsqueeze(3)
        maxout = self.max_pool(x).view(x.size(0), x.size(1)).unsqueeze(2).unsqueeze(3)
        avgout = torch.squeeze(avgout)
        maxout = torch.squeeze(maxout)
        avgout = self.shared_MLP(avgout)
        maxout = self.shared_MLP(maxout)
        out = self.sigmoid(avgout + maxout)
        return out


# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = out.unsqueeze(2).unsqueeze(3)
        out = self.sigmoid(self.conv2d(out))
        out = out.squeeze(2).squeeze(2)
        return out

# CBAM模块
class CBAM(nn.Module):
    def __init__(self, channel, ratio, mlp_dim):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel, ratio)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
#
# class MLPBlock(nn.Module):
#     def __init__(self, mlp_dim: int, hidden_dim: int, dropout=0.):
#         super(MLPBlock, self).__init__()
#         self.mlp_dim = mlp_dim
#         self.hidden_dim = hidden_dim
#         self.dropout = dropout
#         self.Linear1 = nn.Linear(mlp_dim, hidden_dim)
#         self.gelu = nn.GELU()
#         self.dropout = nn.Dropout(dropout)
#         self.Linear2 = nn.Linear(hidden_dim, mlp_dim)
#
#     def forward(self, x):
#         x = self.Linear1(x)
#         x = self.gelu(x)
#         # x = self.dropout(x)
#         x = self.Linear2(x)
#         x = torch.tanh_(x)
#         # x = self.dropout(x)
#         return x

# gMLPBlock
class SpatialGatingUnit(nn.Module):  # [-mainall,256,256]
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)  # [-mainall,256,256]->[-mainall,256,512]
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)  # [-mainall,256,512]->[-mainall,256,512]
        nn.init.constant_(self.spatial_proj.bias, 1.0)  # 偏差

    def forward(self, x):
        # chunk(arr, size)接收两个参数，一个是原数组，一个是分块的大小size，默认值为1，
        # 原数组中的元素会按照size的大小从头开始分块，每一块组成一个新数组，如果最后元素个数不足size的大小，那么它们会组成一个快。
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = v.unsqueeze(2)
        v = self.spatial_proj(v)
        v = torch.squeeze(v)
        out = u + v
        return out
class MLPBlock(nn.Module):
    def __init__(self, mlp_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(mlp_dim)
        self.channel_proj1 = nn.Linear(mlp_dim, hidden_dim * 2)  # (256, d_ffn * 2=1024)  [-mainall,256,1024]
        self.sgu = SpatialGatingUnit(hidden_dim, hidden_dim)  #
        self.channel_proj2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)  # [-mainall,256,256]
        x = F.gelu(self.channel_proj1(x))  # GELU激活函数 [-mainall,256,256]
        x = self.sgu(x)  # [-mainall,256,256]
        x = self.channel_proj2(x)
        out = x + residual
        return out

class ST(nn.Module):
    def __init__(self):
        super(ST, self).__init__()

    def forward(self, x):
        x = torch.squeeze(x)
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        sub = x_abs
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


# 软阈值
class AdaptiveSoftThreshold(nn.Module):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.register_parameter("bias", nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float()))  # b

    # 计算|t| − b
    def forward(self, c):
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)  # torch.abs：计算 input 中每个元素的绝对值


class SENet(nn.Module):

    def __init__(self, input_dims, hid_dims, out_dims, channel, ratio, kaiming_init=True):
        super(SENet, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        self.channel = channel
        self.ratio = ratio
        self.kaiming_init = kaiming_init
        self.shrink = 1.0 / out_dims
        # self.net_q = MLPBlock(mlp_dim=self.input_dims,
        #                       hidden_dim=self.hid_dims,
        #                       dropout=0.)
        #
        # self.net_k = MLPBlock(mlp_dim=self.input_dims,
        #                       hidden_dim=self.hid_dims,
        #                       dropout=0.)
        self.net_q = MLPBlock(mlp_dim=self.input_dims,
                              hidden_dim=self.hid_dims,)

        self.net_k = MLPBlock(mlp_dim=self.input_dims,
                              hidden_dim=self.hid_dims,)

        self.CBAM = CBAM(channel=self.channel, ratio=self.ratio, mlp_dim=self.input_dims)
        self.ST = ST()
        self.thres = AdaptiveSoftThreshold(1)

    def query_embedding(self, queries):
        q_emb = self.CBAM(queries)
        q_emb = self.net_q(q_emb)
        q_emb = self.ST(q_emb)
        return q_emb

    def key_embedding(self, keys):
        k_emb = self.CBAM(keys)
        k_emb = self.net_k(k_emb)
        k_emb = self.ST(k_emb)
        return k_emb

    def get_coeff(self, q_emb, k_emb):
        c = self.thres(q_emb.mm(k_emb.t()))  # .t():转置； .mm():q_emb * k_emb转置（内积）
        return self.shrink * c

    def forward(self, queries, keys):
        q = self.query_embedding(queries)
        k = self.key_embedding(keys)
        out = self.get_coeff(q_emb=q, k_emb=k)
        return out


# 得到C稀疏矩阵：
def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()  # 正则化参数


def get_sparse_rep(senet, data, batch_size=10, chunk_size=100, non_zeros=1000):  # 获得稀疏表示
    N, D = data.shape
    non_zeros = min(N, non_zeros)
    C = torch.empty([batch_size, N])  # C中系数置零
    if (N % batch_size != 0):
        raise Exception("batch_size should be a factor of dataset size.")
    if (N % chunk_size != 0):
        raise Exception("chunk_size should be a factor of dataset size.")

    val = []
    indicies = []
    with torch.no_grad():
        senet.eval()
        for i in range(data.shape[0] // batch_size):
            chunk = data[i * batch_size:(i + 1) * batch_size].cuda()
            q = senet.query_embedding(chunk)
            for j in range(data.shape[0] // chunk_size):
                chunk_samples = data[j * chunk_size: (j + 1) * chunk_size].cuda()
                k = senet.key_embedding(chunk_samples)
                temp = senet.get_coeff(q, k)  # 内积
                C[:, j * chunk_size:(j + 1) * chunk_size] = temp.cpu()

            rows = list(range(batch_size))
            cols = [j + i * batch_size for j in rows]
            C[rows, cols] = 0.0  # 对角线置零

            _, index = torch.topk(torch.abs(C), dim=1, k=non_zeros)  # 软阈值Tb

            val.append(C.gather(1, index).reshape([-1]).cpu().data.numpy())
            index = index.reshape([-1]).cpu().data.numpy()
            indicies.append(index)

    val = np.concatenate(val, axis=0)
    indicies = np.concatenate(indicies, axis=0)
    indptr = [non_zeros * i for i in range(N + 1)]

    C_sparse = sparse.csr_matrix((val, indicies, indptr), shape=[N, N])
    return C_sparse


# 系数矩阵C的列构造3-最近邻图（对于MNIST、Fashion MNIST和EMNIST）
def get_knn_Aff(C_sparse_normalized, k=3, mode='symmetric'):
    C_knn = kneighbors_graph(C_sparse_normalized, k, mode='connectivity', include_self=False, n_jobs=10)
    if mode == 'symmetric':
        Aff_knn = 0.5 * (C_knn + C_knn.T)
    elif mode == 'reciprocal':
        Aff_knn = C_knn.multiply(C_knn.T)
    else:
        raise Exception("Mode must be 'symmetric' or 'reciprocal'")
    return Aff_knn


# 评价
def evaluate(senet, data, labels, num_subspaces, spectral_dim, non_zeros=1000, n_neighbors=3,
             batch_size=10000, chunk_size=10000, affinity='nearest_neighbor', knn_mode='symmetric'):
    C_sparse = get_sparse_rep(senet=senet, data=data, batch_size=batch_size,
                              chunk_size=chunk_size, non_zeros=non_zeros)
    C_sparse_normalized = normalize(C_sparse).astype(np.float32)
    if affinity == 'symmetric':
        Aff = 0.5 * (np.abs(C_sparse_normalized) + np.abs(C_sparse_normalized).T)  # 亲和矩阵|C|+|C^T|
    elif affinity == 'nearest_neighbor':
        Aff = get_knn_Aff(C_sparse_normalized, k=n_neighbors, mode=knn_mode)  # 系数矩阵C的列构造3-最近邻图
    else:
        raise Exception("affinity should be 'symmetric' or 'nearest_neighbor'")
    preds = utils.spectral_clustering(Aff, num_subspaces, spectral_dim)
    acc = clustering_accuracy(labels, preds)
    nmi = normalized_mutual_info_score(labels, preds, average_method='geometric')
    ari = adjusted_rand_score(labels, preds)
    return acc, nmi, ari


# 设置固定随机种子
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    random.seed(seed)
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--num_subspaces', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=200.0)
    parser.add_argument('--lmbd', type=float, default=0.9)
    parser.add_argument('--hid_dims', type=int, default=1024)  # MNIST:896;FashionMNIST:1024;CIFAR10.txt:786
    parser.add_argument('--out_dims', type=int, default=1024)  # MNIST:896;FashionMNIST:1024;CIFAR10.txt:786
    parser.add_argument('--channel', type=int, default=500)
    parser.add_argument('--ratio', type=int, default=4)  # MNIST:4;FashionMNIST:8;CIFAR10.txt:8
    parser.add_argument('--total_iters', type=int, default=100000)
    parser.add_argument('--save_iters', type=int, default=200000)
    parser.add_argument('--eval_iters', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_min', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--chunk_size', type=int, default=10000)
    parser.add_argument('--non_zeros', type=int, default=1000)
    parser.add_argument('--n_neighbors', type=int, default=3)
    parser.add_argument('--spectral_dim', type=int, default=15)
    parser.add_argument('--affinity', type=str, default="nearest_neighbor")
    parser.add_argument('--mean_subtract', dest='mean_subtraction', action='store_true')
    parser.set_defaults(mean_subtraction=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.dataset == 'MNIST':
        args.__setattr__('gamma', 200.0)
        args.__setattr__('spectral_dim', 15)  # 15
        args.__setattr__('hid_dims', 1024)  # MNIST:1024
        args.__setattr__('out_dims', 1024)  # MNIST:1024
        args.__setattr__('ratio', 8)
        args.__setattr__('mean_subtract', False)
        args.__setattr__('lr_min', 0.0)
    elif args.dataset == 'FashionMNIST':
        args.__setattr__('gamma', 200.0)
        args.__setattr__('spectral_dim', 15)  # 15
        args.__setattr__('hid_dims', 786)  # MNIST:896;FashionMNIST:786
        args.__setattr__('out_dims', 786)  # MNIST:896;FashionMNIST:786
        args.__setattr__('ratio', 4)
        args.__setattr__('mean_subtract', False)
        args.__setattr__('lr_min', 0.0)
    elif args.dataset == 'EMNIST':
        args.__setattr__('gamma', 150.0)
        args.__setattr__('num_subspaces', 26)
        args.__setattr__('spectral_dim', 26)
        args.__setattr__('ratio', 4)
        args.__setattr__('mean_subtract', True)
        args.__setattr__('chunk_size', 10611)
        args.__setattr__('lr_min', 1e-3)
    elif args.dataset == 'CIFAR10':
        args.__setattr__('gamma', 200.0)
        args.__setattr__('num_subspaces', 10)
        args.__setattr__('chunk_size', 10000)
        args.__setattr__('total_iters', 50000)
        args.__setattr__('eval_iters', 100000)
        args.__setattr__('hid_dims', 512)  # 786:服务器; 512:电脑
        args.__setattr__('out_dims', 512)  # 786:服务器; 512:电脑
        args.__setattr__('channel', 128)
        args.__setattr__('lr_min', 0.0)
        args.__setattr__('spectral_dim', 13)  # 10
        args.__setattr__('ratio', 4)
        args.__setattr__('mean_subtract', False)
        args.__setattr__('affinity', 'symmetric')
    else:
        raise Exception("Only MNIST, FashionMNIST, EMNIST and CIFAR10.txt are currently supported.")

    fit_msg = "Experiments on {}, numpy_seed=0, total_iters=100000, lambda=0.9, gamma=200.0".format(args.dataset,
                                                                                                    args.seed)
    print(fit_msg)

    folder = "result/CBAMgmlp/test/修后/{}_result".format(args.dataset)
    if not os.path.exists(folder):
        os.mkdir(folder)

    same_seeds(args.seed)  # 固定种子
    tic = time.time()

    if args.dataset in ["MNIST", "FashionMNIST", "EMNIST"]:
        # with open('datasets/{}/{}_scattering_train_data.pkl'.format(args.dataset, args.dataset), 'rb') as f:
        #     train_samples = pickle.load(f)
        # with open('datasets/{}/{}_scattering_train_label.pkl'.format(args.dataset, args.dataset), 'rb') as f:
        #     train_labels = pickle.load(f)
        with open('datasets/{}/{}_scattering_test_data.pkl'.format(args.dataset, args.dataset), 'rb') as f:
            test_samples = pickle.load(f)
        with open('datasets/{}/{}_scattering_test_label.pkl'.format(args.dataset, args.dataset), 'rb') as f:
            test_labels = pickle.load(f)
        # full_samples = np.concatenate([train_samples, test_samples], axis=0)
        # full_labels = np.concatenate([train_labels, test_labels], axis=0)
        full_samples = test_samples
        full_labels = test_labels
    elif args.dataset in ["CIFAR10"]:
        with open('datasets/CIFAR10-MCR2/cifar10-features.npy', 'rb') as f:
            full_samples = np.load(f)
        with open('datasets/CIFAR10-MCR2/cifar10-labels.npy', 'rb') as f:
            full_labels = np.load(f)
    else:
        raise Exception("Only MNIST, FashionMNIST and EMNIST are currently supported.")

    if args.mean_subtract:
        print("Mean Subtraction")
        full_samples = full_samples - np.mean(full_samples, axis=0, keepdims=True)  # mean subtraction 平均减法

    full_labels = full_labels - np.min(full_labels)  # 计算sre时需要label的范围是 0 ~ num_subspaces - CoordAtt

    result = open('{}/results.csv'.format(folder), 'w')
    writer = csv.writer(result)
    writer.writerow(["N", "ACC", "NMI", "ARI"])

    global_steps = 0
    for N in [200, 500, 1000, 2000, 5000, 10000, 20000]:
        sampled_idx = np.random.choice(full_samples.shape[0], N, replace=False)  # 从数组中随机抽取元素
        samples, labels = full_samples[sampled_idx], full_labels[sampled_idx]
        block_size = min(N, 10000)

        with open('{}/{}_samples_{}.pkl'.format(folder, args.dataset, N), 'wb') as f:
            pickle.dump(samples, f)
        with open('{}/{}_labels_{}.pkl'.format(folder, args.dataset, N), 'wb') as f:
            pickle.dump(labels, f)

        all_samples, ambient_dim = samples.shape[0], samples.shape[1]

        data = torch.from_numpy(samples).float()
        data = utils.p_normalize(data)  # 范数

        n_iter_per_epoch = samples.shape[0] // args.batch_size
        n_step_per_iter = round(all_samples // block_size)
        n_epochs = args.total_iters // n_iter_per_epoch

        senet = SENet(ambient_dim, args.hid_dims, args.out_dims, args.channel, args.ratio, kaiming_init=True).cuda()
        optimizer = optim.Adam(senet.parameters(), lr=args.lr)  # Adam优化器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=args.lr_min)  # 调整学习率

        n_iters = 0
        pbar = tqdm(range(n_epochs), ncols=120)  # 进度条

        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")
            randidx = torch.randperm(data.shape[0])  # 将0~data.shape[0]（包括0和data.shape[0]）随机打乱后获得的数字序列

            for i in range(n_iter_per_epoch):
                senet.train()  # 启用 Batch Normalization 和 Dropout

                batch_idx = randidx[i * args.batch_size: (i + 1) * args.batch_size]
                batch = data[batch_idx].cuda()
                q_batch = senet.query_embedding(batch)
                k_batch = senet.key_embedding(batch)

                rec_batch = torch.zeros_like(batch).cuda()
                reg = torch.zeros([1]).cuda()
                for j in range(n_step_per_iter):
                    block = data[j * block_size: (j + 1) * block_size].cuda()
                    k_block = senet.key_embedding(block)
                    c = senet.get_coeff(q_batch, k_block)
                    rec_batch = rec_batch + c.mm(block)
                    reg = reg + regularizer(c, args.lmbd)

                diag_c = senet.thres((q_batch * k_batch).sum(dim=1, keepdim=True)) * senet.shrink  # f(X, xj; Θ)
                rec_batch = rec_batch - diag_c * batch
                reg = reg - regularizer(diag_c, args.lmbd)

                rec_loss = torch.sum(torch.pow(batch - rec_batch, 2))  # 计算两个张量或者一个张量与一个标量的指数计算结果，返回一个张量
                loss = (0.5 * args.gamma * rec_loss + reg) / args.batch_size  # 损失函数

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(senet.parameters(), 0.001)  # 梯度剪裁 梯度上限：0.001
                optimizer.step()

                global_steps += 1
                n_iters += 1

                if n_iters % args.save_iters == 0:
                    with open('{}/SENet_{}_N{:d}_iter{:d}.pth.tar'.format(folder, args.dataset, N, n_iters), 'wb') as f:
                        torch.save(senet.state_dict(), f)
                    print("Model Saved.")

                if n_iters % args.eval_iters == 0:
                    print("Evaluating on sampled data...")
                    acc, nmi, ari = evaluate(senet, data=data, labels=labels, num_subspaces=args.num_subspaces,
                                             affinity=args.affinity,
                                             spectral_dim=args.spectral_dim, non_zeros=args.non_zeros,
                                             n_neighbors=args.n_neighbors,
                                             batch_size=block_size, chunk_size=block_size,
                                             knn_mode='symmetric')
                    print("ACC-{:.6f}, NMI-{:.6f}, ARI-{:.6f}".format(acc, nmi, ari))

            pbar.set_postfix(loss="{:3.4f}".format(loss.item()),
                             rec_loss="{:3.4f}".format(rec_loss.item() / args.batch_size),
                             reg="{:3.4f}".format(reg.item() / args.batch_size))
            scheduler.step()

        print("Evaluating on {}-full...".format(args.dataset))
        full_data = torch.from_numpy(full_samples).float()
        full_data = utils.p_normalize(full_data)
        acc, nmi, ari = evaluate(senet, data=full_data, labels=full_labels, num_subspaces=args.num_subspaces,
                                 affinity=args.affinity,
                                 spectral_dim=args.spectral_dim, non_zeros=args.non_zeros, n_neighbors=args.n_neighbors,
                                 batch_size=args.chunk_size,
                                 chunk_size=args.chunk_size, knn_mode='symmetric')
        print("N-{:d}: ACC-{:.6f}, NMI-{:.6f}, ARI-{:.6f}".format(N, acc, nmi, ari))
        writer.writerow([N, acc, nmi, ari])
        result.flush()

        with open('{}/SENet_{}_N{:d}.pth.tar'.format(folder, args.dataset, N), 'wb') as f:
            torch.save(senet.state_dict(), f)

        torch.cuda.empty_cache()
    result.close()
