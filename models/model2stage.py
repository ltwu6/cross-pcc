import torch
import torch.nn as nn
from models.utils import PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, Transformer,\
    Conv2d, FC_Enhance
from pytorch3d.ops.knn import knn_points

DEVICE = 'cuda:0'

class Encoder_3D(nn.Module):
    def __init__(self, out_dim=512,use_tfm=True):
        super(Encoder_3D, self).__init__()
        self.use_tfm = use_tfm
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        
        self.transformer_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim,out_dim),
            nn.BatchNorm1d(out_dim))
    def forward(self, point_cloud):
        """
        Args:
             point_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = point_cloud
        l0_points = point_cloud
        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)

        if self.use_tfm:
            l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 128)

        if self.use_tfm:
            l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)
        
        return l3_points, l2_points, l1_points

class Encoder_2D(nn.Module):
    def __init__(self, out_dim=512):
        """Encoder that encodes information of RGB image
        """
        super(Encoder_2D, self).__init__()
        self.out_dim = out_dim

        # net for global feature
        self.conv0_1 = Conv2d(3, 16, 3, stride = 1, padding = 1)
        self.conv0_2 = Conv2d(16, 16, 3, stride = 1, padding = 1)
        self.conv1_1 = Conv2d(16, 32, 3, stride = 2, padding = 1) # 224 -> 112
        self.conv2_1 = Conv2d(32, 64, 3, stride = 2, padding = 1) # 112 -> 56
        self.conv3_1 = Conv2d(64, 128, 3, stride = 2, padding = 1) # 56 -> 28
        self.conv4_1 = Conv2d(128, 256, 5, stride = 2, padding = 2) # 28 -> 14
        self.conv5_1 = Conv2d(256, self.out_dim, 5, stride = 2, padding = 2, if_bn=True, activation_fn=None) # 14 -> 7
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.mlp = nn.Sequential(
            nn.Linear(self.out_dim,self.out_dim),
            nn.BatchNorm1d(self.out_dim))

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.conv0_2(x)
        x = self.conv1_1(x)
        r1 = x
        x = self.conv2_1(x) # 56
        r2 = x
        x = self.conv3_1(x) # 28
        r3 = x
        x = self.conv4_1(x) # 14
        r4 = x
        x1 = self.conv5_1(x) # 7
        x = self.avg_pool(x1)
        x = x.squeeze(-1)
        return (x, r1,r2,r3,r4,x1)

class Cross_PCC(nn.Module):
    def __init__(self, encoder_dim=512, global_dim=512, cal_flag=False, branch2d=False, num_coarse=1024, num_out=2048, use_tfm=True):
        super().__init__()
        """
        
        """
        self.cal_flag = cal_flag
        self.branch2d = branch2d
        self.encoder_dim = encoder_dim
        self.global_dim = global_dim
        self.num_coarse = num_coarse
        self.out_num = num_out
        self.use_tfm = use_tfm

        self.encoder3d=Encoder_3D(out_dim=self.encoder_dim,use_tfm=self.use_tfm)
        if self.branch2d:
            self.encoder2d=Encoder_2D(out_dim=self.encoder_dim)

        self.decoder=Decoder(dim_feat=self.global_dim, num_pc=256, num_p0=512, radius=1, up_factors=[4], cal_flag=self.cal_flag)

        self.mlp_conv = MLP_CONV(in_channel=256+self.encoder_dim+self.encoder_dim, layer_dims=[512, self.global_dim], bn=True) # should be fc?
        self.enc_fc = FC_Enhance(FC_dims=[512, 512, 512])
    def forward(self, image, point_cloud, mask, params, bounds, view_id, inv_param):
        """
        image: (n c h w), input rgb image
        point_cloud: (b n c) partial input
        mask: (b h w) 
        params: (b 8 4 4) projection matrix
        bounds: (b 8 240 2) boundary points of all 8 views
        view_id: view id of input rgb image
        inv_param: (b 8 4 4) inverse projection matrix
        """
        part_ori = point_cloud
        point_cloud = point_cloud.permute(0, 2, 1).contiguous() #[b n c]-->[b c n]

        feat_3d, l2, l1 = self.encoder3d(point_cloud) # [b 512 1], [b 256 128]
        if self.branch2d:
            feat_2d = self.encoder2d(image) # b 512 1
            global_feat = torch.cat([l2, feat_3d.expand(-1, -1, l2.shape[-1]), feat_2d[0].expand(-1, -1, l2.shape[-1])], dim=1) # b 1280 128
            global_feat = self.mlp_conv(global_feat)
            global_feat = torch.max(global_feat,dim=2,keepdim=False)[0] # B 512
        else:
            global_feat = feat_3d.squeeze(-1)
        global_feat = self.enc_fc(global_feat)
        pcd_list, proj_list, calib_pc = self.decoder(global_feat.unsqueeze(-1),part_ori, mask, params, bounds, view_id, inv_param) # coarse: [b,n1,6], fine:[b,n2,6]
        return pcd_list, proj_list, calib_pc

class Decoder(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, radius=1, up_factors=[4], cal_flag=True):
        super(Decoder, self).__init__()
        self.cal_flag = cal_flag
        self.num_p0 = num_p0
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_pc)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = [1] + up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(SPD(dim_feat=dim_feat, up_factor=factor, i=i, radius=radius))

        self.projection = Projection()

        self.uppers = nn.ModuleList(uppers)

        self.calib = Calibration()
    def forward(self, feat, partial, mask, params, bounds, view_id, inv_param, return_P0=True):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
            mask: mask of input view image
            params: projection matrix
            bounds: boundary poitns, (n,8,m,2)
            view_id: view id of input view image
        """
        arr_pcd = []
        arr_proj = []
        pcd = self.decoder_coarse(feat).permute(0, 2, 1).contiguous()  # (B, num_pc, 3)
        assert torch.isnan(pcd).sum() == 0 and torch.isinf(pcd).sum() == 0
        arr_pcd.append(pcd)
        proj_P0, proj_P0z = self.projection(pcd, params) # b 8 n 2
        arr_proj.append(proj_P0)
        pcd = fps_subsample(torch.cat([pcd, partial], 1), self.num_p0)
        if return_P0:
            arr_pcd.append(pcd)
        pcd = pcd.permute(0, 2, 1).contiguous() # b 3 n
        for i, upper in enumerate(self.uppers):
            pcd = upper(pcd, feat) # b 3 n
            pcd = pcd.permute(0, 2, 1).contiguous() # b n 3
            arr_pcd.append(pcd) # b n 3
            
            ## calibration begin
            proj_P, proj_Pz = self.projection(pcd, params) # b 8 n 2
            arr_proj.append(proj_P)
            if (i+1) < len(self.uppers):
                pcd = pcd.permute(0, 2, 1).contiguous() # b 3 n
        ## with calibration
        if self.cal_flag:
            pc = self.calib(arr_pcd[-1], mask, bounds, view_id, inv_param, arr_proj[-1], proj_Pz)
        else:
            pc = None
        return arr_pcd, arr_proj, pc  # elem. no.: 4,3,3,1

class SPD(nn.Module):
    def __init__(self, dim_feat=512, up_factor=1, i=0, radius=1):
        """Snowflake Point Deconvolution"""
        super(SPD, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + dim_feat, layer_dims=[512, 256, 64])

        self.ps = nn.ConvTranspose1d(64, 128, up_factor, up_factor, bias=False)   # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=640, hidden_dim=256, out_dim=128)

        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])
        self.tanh = nn.Tanh()

    def forward(self, pcd_prev, feat_global):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, dim_feat, 1)
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_global.repeat(1, 1, feat_1.size(2))], 1)

        Q = self.mlp_2(feat_1) # 64

        feat_child = self.ps(Q)  # (B, 128, N_prev * up_factor)

        K_curr = self.mlp_delta_feature(torch.cat([feat_child, feat_global.repeat(1, 1, feat_child.size(2))], 1)) # 128+512
        
        delta = torch.mul(torch.tanh(self.mlp_delta(torch.relu(K_curr))), 5.7558) # 5.7558=atanh(0.49999*2)
        pcd_child = self.up_sampler(pcd_prev)
        pcd_child = torch.clamp(pcd_child, min=-0.49999, max=0.49999)        
        pcd_child = torch.atanh(pcd_child*2)
        pcd_child = pcd_child + delta
        pcd_child = torch.mul(self.tanh(pcd_child), 0.5)
        return pcd_child

class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )
        self.tanh = nn.Tanh()
    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, 256)
        completion = torch.mul(self.tanh(completion), 0.5)
        return completion

class Refine(nn.Module):
    def __init__(self, args, encoder_dim=512, global_dim=512, init_weight=True,use_tfm=True):
        super(Refine, self).__init__()
        self.category = args.category
        self.args = args
        self.use_tfm = use_tfm
        self.refine_mlp = DGCNN(k=20)
        self.calib = Calibration()
        self.projection = Projection()
        self.stage1 = Cross_PCC(encoder_dim=512, global_dim=512, cal_flag=True, branch2d=True,use_tfm=self.use_tfm)
        self.tanh = nn.Tanh()
        
        if init_weight:
            self.init_stage1()
    def init_stage1(self):
        print('init parameter for stage1')
        ckpt_path = self.args.pthfile
        ckpt_dict = torch.load(ckpt_path)
        self.stage1.load_state_dict(ckpt_dict['model'])
        print(f'load ckpt from {ckpt_path}')

    def forward(self, image, pc_part, mask, params, bounds, view_id, inv_param):
        """
        pc: (b n 3), pc output by model
        proj_fine: projected fine pc
        proj_finez: projected z coord. of fine pc
        """
        pc_list, proj_list, _, pc = self.stage1(image, pc_part, mask, params, bounds, view_id, inv_param)
        for parameter in self.stage1.parameters():
            parameter.requires_grad = False 
        pc_t = pc.permute(0, 2, 1).contiguous() # b 3 n
        offset = self.refine_mlp(pc_t) # b 3 n
        offset = torch.mul(self.tanh(offset), 0.1)
        res = pc_t+offset # B 3 N
        res = res.permute(0,2,1).contiguous() # b n 3
        res_output = res
        proj_res, proj_resz = self.projection(res, params)
        calib_res = self.calib(res, mask, bounds, view_id, inv_param, proj_res, proj_resz)
        return res_output, proj_res, calib_res, pc

class Projection(nn.Module):
    def __init__(self):
        super(Projection, self).__init__()
    def forward(self, sample_pc, trans_mat_right):
        bsz, pn = sample_pc.shape[0], sample_pc.shape[1]
        homo_pc = torch.cat([sample_pc, torch.ones(bsz, pn, 1).to(device=DEVICE)], dim=-1)
        homo_pc = homo_pc.unsqueeze(dim=1).repeat(1,trans_mat_right.shape[1],1,1)
        pc_xyz = torch.matmul(homo_pc, trans_mat_right)
        pylt = torch.lt(pc_xyz[:,:, :, 2].reshape(bsz,pn*trans_mat_right.shape[1]), 0)
        pygtall = torch.gt(torch.sum(pylt, dim=1), 0)
        pc_xyz[:,:, :, [2]]= torch.where(pc_xyz[:, :, :, [2]]!=0, pc_xyz[:,:, :, [2]], torch.tensor([1e-8], dtype=torch.float32, device=DEVICE,requires_grad=True))
        pc_xy = pc_xyz[:,:,:, :2] / (pc_xyz[:,:, :, [2]])
        pc_xy[:,:,:,1] = 224.0-pc_xy[:,:,:,1]
        return pc_xy, pc_xyz[:,:, :, 2]  

class Calibration(nn.Module):
    def __init__(self):
        super(Calibration, self).__init__()

    def out_distance(self, proj_xy, bounds, mask, view_id):
        """compute the distance between outliers and their nearest boundary point
        for all view projections
        projxy: projected 2d points, [b 8 n 2]
        bounds: boundary points, [b 8 m 2]
        mask: silhouette images, [b, 8, 224, 224]
        view_id: view id of input RGB image
        """
        projxy = proj_xy.clone()
        bsz, v_num, p_num = projxy.shape[0], projxy.shape[1], projxy.shape[2]
        projxy[:,:,:,1] = 224-projxy[:,:,:,1]
        projxy = projxy.round().long() # [b 8 n 2]
        # compute outlier index in accordance with projxy
        bidx = torch.arange(bsz).unsqueeze(-1).unsqueeze(-1).repeat(1, v_num,p_num).contiguous().view(bsz*v_num*p_num)
        vidx = torch.arange(v_num).unsqueeze(-1).repeat(1,p_num).contiguous().view(v_num*p_num).repeat(bsz)
        xidx = projxy[:,:,:,1].reshape(bsz*v_num*p_num)
        yidx = projxy[:,:,:,0].reshape(bsz*v_num*p_num)
        # padding and clamp for solving some coordinates of xidx and yidx out of [0,224]
        mask = torch.nn.ZeroPad2d(1)(mask)
        xidx = torch.clamp(xidx+1, min=0, max=225)
        yidx = torch.clamp(yidx+1, min=0, max=225)
        
        mask_res = mask[bidx,vidx,xidx,yidx] # b*8*n
        mask_res = mask_res.reshape(bsz, v_num, p_num)
        
        # compute the distance for each batch and view, b 8 t
        distance_res = torch.zeros((bsz, v_num), device=DEVICE)
        view_outliers = []
        view_outidx = []
        view_bounds = []
        for i in range(bsz):
            for j in range(v_num):
                out_idx = torch.nonzero(mask_res[i, j]==0).squeeze(-1) # (k), k is the num of outliers
                # # get outliers, b 8 t 2
                if len(list(out_idx.shape))!=0:
                    outliers = projxy[i, j, out_idx].type_as(bounds) # k 2
                    distance, nn_idx, n_bound = knn_points(outliers.unsqueeze(0)/224.0, bounds[i,j].unsqueeze(0)/224.0, K=1, return_nn=True)# 1 k 1, 1 k 1, 1,k,2
                else:
                    outliers = None
                    distance=torch.tensor([0.0], device=DEVICE,requires_grad=True)
                    nn_idx, n_bound = None, None
            
                distance_res[i,j] = torch.mean(distance.squeeze())
                if j == view_id:
                    view_outliers.append(outliers) # k 2
                    view_outidx.append(out_idx) # k
                    view_bounds.append(n_bound.squeeze()*224.0 if n_bound!=None else n_bound) # k 2
        # return mean distance value and view_id outliers(b,k) and its corresponding nearest boundary points
        return distance_res, view_outliers, view_outidx, view_bounds 

    def move_out(self, bound_point, outidx, z_coor, inv_param, pred_pc):
        """
        bound_point: list b* (k, 2), nearest boundary points, k is different among batches
        outidx: list b*(k), outlier index
        z_coor: (b,n), z coordinate of viewid points in camera coord. sys.
        inv_param: (b,4,4), inverse param for back-project from image plane to object coord. system
        pred_pc: (b,n,3), coord. of predicted pc before calibration
        """
        bsz, v_num = inv_param.shape[0], inv_param.shape[1]
        pc_copy = pred_pc.clone()
        for i in range(bsz):
            if bound_point[i] != None:
                back_point = bound_point[i]*z_coor[i, outidx[i]].unsqueeze(-1).repeat(1,2) # k 2
                back_point = torch.matmul(torch.cat([back_point, z_coor[i, outidx[i]].unsqueeze(-1), torch.ones((back_point.shape[0], 1), device=DEVICE)], dim=-1), inv_param[i]) # (k,4)*(4,4)
                pc_copy[i, outidx[i]] = back_point[:,:3]
        return pc_copy

    def forward(self, pc, mask, bounds, view_id, inv_param, proj_fine, proj_finez):
        """
        pc: (b n 3), pc output by model
        proj_fine: projected fine pc
        proj_finez: projected z coord. of fine pc
        """
        out_loss, view_outliers, view_outidx, view_bounds = self.out_distance(proj_fine, bounds, mask, view_id)
        pc = self.move_out(view_bounds, view_outidx, proj_finez[:,view_id], inv_param[:, view_id], pc)
        
        return pc


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class DGCNN(nn.Module):
    def __init__(self, k, output_channels=40):
        super(DGCNN, self).__init__()
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)


        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv5 = nn.Sequential(nn.Conv1d(512, 3, kernel_size=1, bias=False),
        #                            self.bn5,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = MLP_CONV(in_channel=512, layer_dims=[256, 64, 3], bn=True)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)#b 3 n
        return x
