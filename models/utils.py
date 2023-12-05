#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Xiang

from turtle import forward
import torch
from torch import nn, einsum
import os
import sys 

from pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation

class Conv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1,  if_bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2)):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), padding=0, if_bn=True, activation_fn=nn.LeakyReLU(negative_slope=0.2)):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class MLP(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Linear(last_channel, out_channel))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            last_channel = out_channel
        layers.append(nn.Linear(last_channel, layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(nn.LeakyReLU(negative_slope=0.2)(self.conv_1(x))) + shortcut
        return out


def sample_and_group(xyz, points, npoint, nsample, radius, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous() # (B, N, 3)
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint)) # (B, 3, npoint)

    idx = ball_query(radius, nsample, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous()) # (B, npoint, nsample)
    grouped_xyz = grouping_operation(xyz, idx) # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, nsample)

    if points is not None:
        grouped_points = grouping_operation(points, idx) # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    device = xyz.device
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float, device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample, device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


class PointNet_SA_Module(nn.Module):
    def __init__(self, npoint, nsample, radius, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(xyz, points, self.npoint, self.nsample, self.radius, self.use_xyz)

        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        return new_xyz, new_points


class PointNet_FP_Module(nn.Module):
    def __init__(self, in_channel, mlp, use_points1=False, in_channel_points1=None, if_bn=True):
        """
        Args:
            in_channel: int, input channel of points2
            mlp: list of int
            use_points1: boolean, if use points
            in_channel_points1: int, input channel of points1
        """
        super(PointNet_FP_Module, self).__init__()
        self.use_points1 = use_points1

        if use_points1:
            in_channel += in_channel_points1

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: Tensor, (B, 3, N)
            xyz2: Tensor, (B, 3, M)
            points1: Tensor, (B, in_channel, N)
            points2: Tensor, (B, in_channel, M)

        Returns:MLP_CONV
            new_points: Tensor, (B, mlp[-1], N)
        """
        dist, idx = three_nn(xyz1.permute(0, 2, 1).contiguous(), xyz2.permute(0, 2, 1).contiguous())
        dist = torch.clamp_min(dist, 1e-10)  # (B, N, 3)
        recip_dist = 1.0/dist
        norm = torch.sum(recip_dist, 2, keepdim=True).repeat((1, 1, 3))
        weight = recip_dist / norm
        interpolated_points = three_interpolate(points2, idx, weight) # B, in_channel, N

        if self.use_points1:
            new_points = torch.cat([interpolated_points, points1], 1)
        else:
            new_points = interpolated_points

        new_points = self.mlp_conv(new_points)
        return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :, pad: nsample+pad]
    return idx.int()


def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        k(nsample): int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint) --- center points
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample) --- grouped features
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous() # (B, N, 3)
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint)) # (B, 3, npoint)
    if idx is None:
        idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx) # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)

    if points is not None:
        grouped_points = grouping_operation(points, idx) # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1) # (B, f+3, npoint, nsample)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


class PointNet_SA_Module_KNN(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True, if_idx=False):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module_KNN, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points, idx=None):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_knn(xyz, points, self.npoint, self.nsample, self.use_xyz, idx=idx)     
        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        if self.if_idx:
            return new_xyz, new_points, idx
        else:
            return new_xyz, new_points


def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd


class Transformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y+identity

# def up_sample_folding(x, ratio, grid_2d, h, w, scope, bn_decay, weight_decay, is_training):
#     print('shape of x in up_sample_folding: ', x.shape)
#     batch_size = x.shape[0]
#     npoint = x.shape[2]
#     ch = x.shape[1]
#     feature = x.repeat(1,1,ratio) #


# def up_down_up_folding(x, ratio, grid_2d, h, w, scope, bn_decay, weight_decay, is_training):
#     x_up = up_sample_folding(x, ratio, grid_2d, h, w, scope='up_sample_x', bn_decay=bn_decay, weight_decay=weight_decay, is_training=is_training)
#     x_up_down = down_sample(x_up, ratio, scope='down_sample', bn_decay=bn_decay, weight_decay=weight_decay, is_training=is_training)
#     delta = x_up_down - x
#     delta_up = up_sample_folding(delta, ratio, grid_2d, h, w, scope='up_sample_delta', bn_decay=bn_decay, weight_decay=weight_decay, is_training=is_training)
#     gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
#     x_up = x_up + delta_up * gamma
#     return x_up

def create_grid(grid_size, bsize):
        a = torch.linspace(-0.5, 0.5, steps=grid_size, dtype=torch.float).view(1, grid_size).expand(grid_size, grid_size).reshape(1, -1)
        b = torch.linspace(-0.5, 0.5, steps=grid_size, dtype=torch.float).view(grid_size, 1).expand(grid_size, grid_size).reshape(1, -1)
        folding_seed = torch.cat([a, b], dim=0).view(1, 2, grid_size ** 2).cuda() # 1 2 N
        print('folding_seed: ',folding_seed)
        seed = folding_seed.view(1, 2, grid_size**2).expand(bsize, 2, grid_size**2)
        return seed

class Self_Attention(nn.Module):
    def __init__(self, in_dim):
        """
        Args:
        in_dim: channel of input feature
        """
        super(Self_Attention, self).__init__()
        self.in_dim = in_dim
        self.att_q = Conv1d(in_channel=self.in_dim, out_channel=self.in_dim//8, activation_fn=None)
        self.att_k = Conv1d(in_channel=self.in_dim, out_channel=self.in_dim//8, activation_fn=None)
        self.att_v = Conv1d(in_channel=self.in_dim, out_channel=self.in_dim)
        # self.att_f = nn.Sequential(
        #     nn.Conv1d(self.in_dim, self.in_dim//8, 1),
        #     nn.BatchNorm1d(self.in_dim//8)
        # )
        # self.att_g = nn.Sequential(
        #     nn.Conv1d(self.in_dim, self.in_dim//8, 1),
        #     nn.BatchNorm1d(self.in_dim//8)
        # )
        # self.att_h = nn.Sequential(
        #     nn.Conv1d(self.in_dim, self.in_dim, 1),
        #     nn.BatchNorm1d(self.in_dim),
        #     nn.ReLU()
        # )

    def forward(self, x):
        ch = x.shape[1]
        assert ch==self.in_dim
        q = self.att_q(x) # [b_size, ch//8, npoint]
        k = self.att_k(x) # [b_size, ch//8, npoint]
        v = self.att_v(x) # [b_size, ch, npoint]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        o = attn @ v
        o += x
        return o

class Upsample_Folding(nn.Module):
    def __init__(self, ratio, g_size, in_dim, layers):
        """
        Args:
        ratio: point_num_layeri+1/point_num_layeri=(grid[i+1]/grid[i])**2
        g_size: grid size
        in_dim: channel of input feature       
        """
        super(Upsample_Folding, self).__init__()
        self.ratio = ratio
        self.g_size = g_size
        self.in_dim = in_dim
        self.layers = layers
        self.attention = Self_Attention(self.in_dim+3)
        self.mlp_conv = MLP_CONV(in_channel=self.in_dim+2, layer_dims=self.layers, bn=True)
        self.conv1d = Conv1d(in_channel=self.in_dim, out_channel=self.in_dim)

    def forward(self, x):
        b_size = x.shape[0]
        feature = x.repeat(1,1,self.ratio) # [b c 256]
        grid = create_grid(self.g_size, b_size)
        x = torch.cat([feature, grid.to(x.device)], dim=1) # [b c+2 256]
        x = self.mlp_conv(x)
        x = torch.cat([feature, x], dim=1)
        x = self.attention(x)
        x = self.conv1d(x)
        return x    

class Down_Sample(nn.Module):
    def __init__(self, ratio, in_dim):
        super().__init__()
        self.ratio = ratio
        self.in_dim = in_dim
        self.conv1d = Conv1d(in_channel=self.in_dim, out_channel=self.in_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        npoint = x.shape[2]
        x = torch.reshape(x, (batch_size, -1, int(npoint / self.ratio)))
        x = self.conv1d(x)
        return x


class UpDownUp_Folding(nn.Module):
    def __init__(self, ratio, grid_size, in_channel, layers):
        super(UpDownUp_Folding, self).__init__()
        self.ratio = ratio
        self.grid_size = grid_size 
        self.in_channel = in_channel
        self.layers = layers
        self.upsample_fold = Upsample_Folding(ratio=self.ratio, g_size=self.grid_size, in_dim=self.in_channel, layers=self.layers)
        self.conv1 = Conv1d()
        self.downsample = Down_Sample(self.ratio, self.in_channel)

    def forward(self, x):
        x_up = self.upsample_fold(x)
        x_down = self.downsample(x)
        delta = x_down-x
        delta = self.upsample_fold(delta)
        x_up += delta
        return x_up


def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    #init weights/bias
    nn.init.xavier_uniform_(li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li


def get_MLP_layers(dims, doLastRelu=False):
    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i-1], dims[i]))
        if i==len(dims)-1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers

class FC_Enhance(nn.Module):
    def __init__(self, FC_dims, MLP_doLastRelu=False):
        
        super(FC_Enhance, self).__init__()
        layers = list(get_MLP_layers(FC_dims, False))
        # print('type of layres: ' , type(layers))
        self.fc = nn.Sequential(*layers)
    def forward(self, x):
        return self.fc(x)












