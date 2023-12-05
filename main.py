# coding=utf-8
import argparse
import os
from thop import profile
import numpy as np
import csv
from models.model2stage import Cross_PCC,Refine
from torch.utils.data import DataLoader
from utils.dataloader_epn import *
from utils.data_util import remove_store
from utils.io_util import save_mat
import torch
import torch.nn as nn
from utils import meter
import time 
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.cdloss import chamfer_distance, chamfer_distance_sqrt, chamfer_distance_sqrtp

parser = argparse.ArgumentParser()
parser.add_argument('--lr',default=1e-4, type=float, help='learning rate')
parser.add_argument('--gpu',default='0')
parser.add_argument('--bz',default=32, type=int,help='batch size')
parser.add_argument('--epoch',default=201, type=int, help='training epoch of stage 1')
parser.add_argument('--refine_epoch', type=int, default=51, help='training epoch of stage 2')
parser.add_argument('--eval_epoch',default=10, type=int, help='evaluation epoch') # default=3
parser.add_argument('--num_workers',default=3, type=int, help='num_workers of dataloader')
parser.add_argument('--resume',default=False, type=bool, help='restore from saved model')
parser.add_argument('--lamda_part', default=1.0) # weight of partial cd loss
parser.add_argument('--data_list', default='./data/3depn/3depn_list') 
parser.add_argument('--data_path', default='./data/3depn')
parser.add_argument('--root_path', default='.', help='log path')
parser.add_argument('--input_num', default=2048, type=int)
parser.add_argument('--gt_num', default=4096, type=int, help='point num of 2d gt')
parser.add_argument('--category', default='cabinet')
parser.add_argument('--view_num', type=int, default=8)
parser.add_argument('--step_decay', type=int, default=10, help='lr decay step of stage 2')
parser.add_argument('--gamma', type=float, default=0.1, help='step decay rate')

args = parser.parse_args()
cat_map = {
            'plane':'02691156',
            'cabinet':'02933112',
            'car':'02958343',
            'chair':'03001627',
            'lamp':'03636649',
            'couch':'04256520',
            'table':'04379243',
            'watercraft':'04530566'
        }

MODEL = 'cross-pcc'
DEVICE = 'cuda:0'

TIME_FLAG = time.asctime(time.localtime(time.time()))

SAVE_PC_DIR = f'{args.root_path}/log/{args.category}/saved_pc'
SAVE_BESTPC_DIR = f'{args.root_path}/log/{args.category}/best_pc'
SAVE_VIEWCD_DIR = f'{args.root_path}/log/{args.category}/saved_viewcd'
CKPT_RECORD_FOLDER = f'{args.root_path}/log/{args.category}/record'
CKPT_FILE = f'{args.root_path}/log/{args.category}/'
os.makedirs(CKPT_FILE, exist_ok=True)
# os.makedirs(SAVE_BESTPC_DIR, exist_ok=True)
# os.makedirs(SAVE_VIEWCD_DIR, exist_ok=True)
# os.makedirs(CKPT_RECORD_FOLDER, exist_ok=True)
# os.makedirs(SAVE_PC_DIR, exist_ok=True)

losses_all = meter.AverageValueMeter()
losses_cd = meter.AverageValueMeter()

def save_record(epoch, prec1, type, net:nn.Module):
    state_dict = net.state_dict()
    torch.save(state_dict,
               os.path.join(CKPT_RECORD_FOLDER, f'epoch{epoch}_{type}_{prec1:.4f}.pth'))

def save_ckpt(epoch, net, type, optimizer_all):
    ckpt = dict(
        epoch=epoch,
        model=net.state_dict(),
        optimizer_all=optimizer_all.state_dict(),
    )
    torch.save(ckpt,CKPT_FILE+type +str(epoch)+'_ckpt.pth')

def train(epoch,model,g_optimizer,train_loader,board_writer):
    loss = None
    pbar = tqdm(total=len(train_loader))
    for iteration, (views, pc_parts, params, point2d, mask, inv_param, bounds, key) in enumerate(train_loader):

        model.train()
        views = views.to(device=DEVICE)
        view_id = np.random.randint(low=0, high=args.view_num) 
        pc_parts = pc_parts.to(device=DEVICE)
        params = params.to(device=DEVICE)
        point2d = point2d.to(device=DEVICE)
        mask = mask.to(device=DEVICE)
        inv_param = inv_param.to(device=DEVICE)
        bounds = bounds.to(device=DEVICE)
        batch_size = views.size(0)
        g_optimizer.zero_grad()
        pcd_list, proj_list, calres = model(views[:,view_id, :3, :, :],pc_parts, mask, params, bounds, view_id, inv_param) # 
        if args.lamda_part !=0:
            loss_part1,_ = chamfer_distance_sqrtp(pc_parts, pcd_list[0], point_reduction='mean')
            loss_part2,_ = chamfer_distance_sqrtp(pc_parts, pcd_list[2], point_reduction='mean')
            loss_part3,_ = chamfer_distance_sqrtp(pc_parts, pcd_list[3], point_reduction='mean')

        proj_coarse1_re = proj_list[0].reshape(batch_size*args.view_num, 256, 2)
        proj_coarse2_re = proj_list[1].reshape(batch_size*args.view_num, 512, 2)
        proj_fine_re = proj_list[2].reshape(batch_size*args.view_num, 2048, 2)
        point2d_re = point2d.reshape(batch_size*args.view_num, args.gt_num, 2)      
        loss2d1,_ = chamfer_distance(proj_coarse1_re/224.0, point2d_re, point_reduction='mean')
        loss2d2,_ = chamfer_distance(proj_coarse2_re/224.0, point2d_re, point_reduction='mean')
        loss2d3,_ = chamfer_distance(proj_fine_re/224.0, point2d_re, point_reduction='mean')
      
        loss =  loss2d1+loss2d2+loss2d3
        if args.lamda_part!=0:
            loss = loss + args.lamda_part*loss_part3+args.lamda_part*loss_part2 +args.lamda_part*loss_part1
             
        assert torch.isnan(loss).sum() == 0 or torch.isinf(loss).sum() == 0
        loss.backward()
        g_optimizer.step()
        losses_all.add(float(loss))  # batchsize

        board_writer.add_scalar('loss/loss_all', losses_all.value()[0], global_step=iteration+epoch*len(train_loader))
        board_writer.add_scalar('lr',g_optimizer.state_dict()['param_groups'][0]['lr'],global_step=iteration+epoch*len(train_loader))

        pbar.set_postfix(loss=losses_all.value()[0])
        pbar.update(1)
        
    if epoch % args.eval_epoch ==0: 
        save_ckpt(epoch, model, '', g_optimizer)
        save_record(epoch, losses_all.value()[0], '', model)
    pbar.close()
    return loss

def refine(epoch,refiner,g_optimizer,train_loader,board_writer):
    loss = None
    pbar = tqdm(total=len(train_loader))
    for iteration, (views, pc_parts, params, point2d, mask, inv_param, bounds, key) in enumerate(train_loader):
        refiner.train()
        views = views.to(device=DEVICE)
        view_id = np.random.randint(low=0, high=args.view_num) 
        pc_parts = pc_parts.to(device=DEVICE)
        params = params.to(device=DEVICE)
        point2d = point2d.to(device=DEVICE)
        mask = mask.to(device=DEVICE)
        inv_param = inv_param.to(device=DEVICE)
        bounds = bounds.to(device=DEVICE)
        batch_size = views.size(0)
        g_optimizer.zero_grad()
        refine_pc, refine_proj, calib_pc, pc0 = refiner(views[:,view_id, :3, :, :],pc_parts, mask, params, bounds, view_id, inv_param)
        loss_part, _ = chamfer_distance_sqrtp(pc_parts, refine_pc, point_reduction='mean')
        refine_proj_re = refine_proj.reshape(batch_size*args.view_num, 2048, 2)
        point2d_re = point2d.reshape(batch_size*args.view_num, args.gt_num, 2)      
        loss2d, _ = chamfer_distance_sqrt(refine_proj_re/224.0, point2d_re, point_reduction='mean')
        loss = loss_part+loss2d
        assert torch.isnan(loss).sum() == 0 and torch.isinf(loss).sum() == 0
        loss.backward()
        g_optimizer.step()

        losses_all.add(float(loss))
        board_writer.add_scalar('loss/loss_refine', losses_all.value()[0], global_step=iteration+epoch*len(train_loader))
        board_writer.add_scalar('lr',g_optimizer.state_dict()['param_groups'][0]['lr'],global_step=iteration+epoch*len(train_loader))

        pbar.set_postfix(loss=losses_all.value()[0])
        pbar.update(1)
        
    if epoch % args.eval_epoch ==0: 
        save_ckpt(epoch, refiner, 'stage2_', g_optimizer)
        save_record(epoch, losses_all.value()[0], 'stage2_', refiner)
    pbar.close()
    return loss

def model_eval(epoch,model,test_loader,last_loss):
    print('---------Evaluation-----------')
    losses_eval_cd = meter.AverageValueMeter()   
    pbar = tqdm(total=len(test_loader))
    model.eval()
    with torch.no_grad():
        if args.category == 'all':
            min_list = {
                '02691156': [], '02933112': [], '02958343': [], '03001627': [], '03636649': [], '04256520': [], '04379243': [], '04530566': []
            }
            avg_list = {
            '02691156': [], '02933112': [], '02958343': [], '03001627': [], '03636649': [], '04256520': [], '04379243': [], '04530566': []
        }
        else:
            min_list = {cat_map[args.category]: []}
            avg_list = {cat_map[args.category]: []}
        for model_idx, (views, pc_parts, pcs, params, inv_params, mask, bounds, key) in enumerate(test_loader):
            fname = key[0].split('/')[0]
            views = views.to(device=DEVICE)
            pc_parts = pc_parts.to(device=DEVICE)
            pcs = pcs.to(device=DEVICE)
            params = params.to(device=DEVICE)
            inv_params = inv_params.to(device=DEVICE)
            mask = mask.to(device=DEVICE)
            bounds = bounds.to(device=DEVICE)
            views_minloss = []
            assert views.shape[1] == args.view_num

            save_viewcd_path = os.path.join(SAVE_VIEWCD_DIR, key[0].split('/')[0])
            os.makedirs(save_viewcd_path,exist_ok=True)
            csv_file = open(os.path.join(save_viewcd_path, key[0].split('/')[1]+'_'+key[0].split('/')[-1]+'.csv'), 'w')
            writer = csv.writer(csv_file)
            writer.writerow(['view_id', 'cd'])
            for i in range(args.view_num):
                pcd_list, proj_list, _ = model(views[:,i, :3, :, :],pc_parts, mask, params, bounds, i, inv_params) # 
                loss_cd,_ = chamfer_distance(pcd_list[-1], pcs, point_reduction='mean')
                loss_cd = loss_cd*10000
                writer.writerow([i,float(loss_cd)])
                views_minloss.append(float(loss_cd))
                avg_list[fname].append(float(loss_cd))
                losses_eval_cd.add(float(loss_cd))
                if epoch%args.eval_epoch==0:
                    fine_2d = torch.squeeze(proj_list[-1], 0).detach().cpu().numpy()
                    fine_2d = fine_2d.astype(np.int32)
                    pc_result = torch.squeeze(pcd_list[3], 0).detach().cpu().numpy()
                    pc_coarse1 = torch.squeeze(pcd_list[2], 0).detach().cpu().numpy()
                    pc_cat = torch.squeeze(pcd_list[1], 0).detach().cpu().numpy()
                    pc_coarse0 = torch.squeeze(pcd_list[0], 0).detach().cpu().numpy()
                                        
                    os.makedirs(os.path.join(SAVE_PC_DIR,key[0].split('/')[0],key[0].split('/')[1],str(model_idx)),exist_ok=True)
                    save_mat(os.path.join(SAVE_PC_DIR,key[0].split('/')[0],key[0].split('/')[1], str(model_idx),'fine_id_'+'_view_'+str(i)+'.mat'), pc_result)
                    save_mat(os.path.join(SAVE_PC_DIR,key[0].split('/')[0],key[0].split('/')[1], str(model_idx),'coarse1_id_'+'_view_'+str(i)+'.mat'), pc_coarse1)
                    save_mat(os.path.join(SAVE_PC_DIR,key[0].split('/')[0],key[0].split('/')[1], str(model_idx),'cat_id_'+'_view_'+str(i)+'.mat'), pc_cat)
                    save_mat(os.path.join(SAVE_PC_DIR,key[0].split('/')[0],key[0].split('/')[1], str(model_idx),'coarse0_id_'+'_view_'+str(i)+'.mat'), pc_coarse0)                 
            min_list[fname].append(min(views_minloss))
            csv_file.close()
            pbar.set_postfix(loss=losses_eval_cd.value()[0])
            pbar.update(1)
        for synset_id in min_list.keys():
            min_list[synset_id] = np.mean(min_list[synset_id])
            avg_list[synset_id] = np.mean(avg_list[synset_id])
        if np.mean(list(avg_list.values())) < last_loss:
            os.makedirs(SAVE_BESTPC_DIR, exist_ok=True)
            remove_store(SAVE_PC_DIR, SAVE_BESTPC_DIR+os.sep+'pc')
    pbar.close()
    return min_list, avg_list

def refine_eval(epoch,refiner,test_loader,last_loss):
    print('---------Evaluation-----------')
    losses_eval_cd = meter.AverageValueMeter()
    
    pbar = tqdm(total=len(test_loader))
    refiner.eval()
    with torch.no_grad():
        if args.category == 'all':
            min_list = {
                '02691156': [], '02933112': [], '02958343': [], '03001627': [], '03636649': [], '04256520': [], '04379243': [], '04530566': []
            }
            avg_list = {
            '02691156': [], '02933112': [], '02958343': [], '03001627': [], '03636649': [], '04256520': [], '04379243': [], '04530566': []
        }
        else:
            min_list = {cat_map[args.category]: []}
            avg_list = {cat_map[args.category]: []}
        
        for model_idx, (views, pc_part, pcs, params, inv_params, mask, bounds, key) in enumerate(test_loader):
            fname = key[0].split('/')[0]
            views = views.to(device=DEVICE)
            pc_part = pc_part.to(device=DEVICE)
            pcs = pcs.to(device=DEVICE)
            params = params.to(device=DEVICE)
            inv_params = inv_params.to(device=DEVICE)
            mask = mask.to(device=DEVICE)
            bounds = bounds.to(device=DEVICE)
            views_minloss = []
            assert views.shape[1] == args.view_num

            save_viewcd_path = os.path.join(SAVE_VIEWCD_DIR+'_refine', key[0].split('/')[0])
            os.makedirs(save_viewcd_path,exist_ok=True)
            csv_file2 = open(os.path.join(save_viewcd_path, key[0].split('/')[1]+'_stage2.csv'), 'w')
            writer2 = csv.writer(csv_file2)
            writer2.writerow(['view_id', 'cd'])
            for i in range(args.view_num):
                refine_pc, refine_proj, calib_pc, pc0 = refiner(views[:,i, :3, :, :],pc_part, mask, params, bounds, i, inv_params)
                loss_cd,_ = chamfer_distance(calib_pc, pcs, point_reduction='mean')
                loss_cd = loss_cd*10000
                writer2.writerow([i,float(loss_cd)])
                views_minloss.append(float(loss_cd))
                avg_list[fname].append(float(loss_cd))
                losses_eval_cd.add(float(loss_cd))   
                if epoch%args.eval_epoch==0:
                    fine_2d = torch.squeeze(refine_proj, 0).detach().cpu().numpy()
                    fine_2d = fine_2d.astype(np.int32)
                    
                    pc_result = torch.squeeze(calib_pc, 0).detach().cpu().numpy()
                    pc_refine = torch.squeeze(refine_pc, 0).detach().cpu().numpy()
                    pc_0 = torch.squeeze(pc0, 0).detach().cpu().numpy()           
                    os.makedirs((os.path.join(SAVE_PC_DIR+'_refine',key[0].split('/')[0],key[0].split('/')[1])),exist_ok=True)
                    save_mat(os.path.join(SAVE_PC_DIR+'_refine',key[0].split('/')[0],key[0].split('/')[1],'calib_id_'+str(model_idx)+'_view_'+str(i)+'.mat'), pc_result)
                    save_mat(os.path.join(SAVE_PC_DIR+'_refine',key[0].split('/')[0],key[0].split('/')[1],'refine_id_'+str(model_idx)+'_view_'+str(i)+'.mat'), pc_refine)
                    save_mat(os.path.join(SAVE_PC_DIR+'_refine',key[0].split('/')[0],key[0].split('/')[1],'calib0_id_'+str(model_idx)+'_view_'+str(i)+'.mat'), pc_0)
            min_list[fname].append(min(views_minloss))
            csv_file2.close()
            pbar.set_postfix(loss=losses_eval_cd.value()[0])
            pbar.update(1)
        for synset_id in min_list.keys():
            min_list[synset_id] = np.mean(min_list[synset_id])
            avg_list[synset_id] = np.mean(avg_list[synset_id])
        if np.mean(list(avg_list.values())) < last_loss:
            os.makedirs(SAVE_BESTPC_DIR+'_refine',exist_ok=True)
            remove_store(SAVE_PC_DIR+'_refine', SAVE_BESTPC_DIR+'_refine'+os.sep+'pc')
    pbar.close()
    return min_list, avg_list

def stage1():
    torch.backends.cudnn.benchmark = True
    model = Cross_PCC(encoder_dim=512, global_dim=512,cal_flag=False, branch2d=True)
    model.to(device=DEVICE) 
    g_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=0.0001)

    mstone = list(range(10, 140, 10))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(g_optimizer, milestones = mstone, gamma = 0.7, last_epoch=-1)

    TRAIN_DATA = TrainDataLoader(filepath=args.data_list, data_path=args.data_path, pc_input_num=args.input_num,category=args.category)
    TEST_DATA = TestDataLoader(filepath=args.data_list,data_path=args.data_path,status='test', pc_input_num=args.input_num,category=args.category)

    train_loader = DataLoader(TRAIN_DATA,
                              batch_size=args.bz ,
                              num_workers=args.num_workers,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True
                              )
    test_loader = DataLoader(TEST_DATA,
                              batch_size=1 ,
                              num_workers=args.num_workers,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=True
                              )

    resume_epoch = 0 # default 0

    board_writer = SummaryWriter(comment=f'{MODEL}_bs{args.bz}_lr{args.lr}_{args.category}_{TIME_FLAG}')

    if args.resume:
        ckpt_path = CKPT_FILE
        ckpt_dict = torch.load(ckpt_path+os.sep+'.pth')
        model.load_state_dict(ckpt_dict['model'])
        g_optimizer.load_state_dict(ckpt_dict['optimizer_all'])
        resume_epoch = ckpt_dict['epoch']

    
    os.makedirs(os.path.join(CKPT_RECORD_FOLDER),exist_ok=True)

    best_loss = 9999
    best_epoch = 0
    for epoch in range(resume_epoch, args.epoch): 
        losses=train(epoch,model,g_optimizer,train_loader,board_writer)
        if epoch % args.eval_epoch ==0:# and epoch!=0:
            loss_min, loss_avg = model_eval(epoch,model,test_loader, best_loss)
            current_mean = np.mean(list(loss_avg.values()))
            with open(os.path.join(CKPT_FILE, 'result.txt'), 'a') as rf:
                for cid in loss_min.keys():
                    rf.write(f'epoch: {epoch:0>4} | {cid}: min_CD--{loss_min[cid]:.6f} | avg_CD--{loss_avg[cid]:.6f}\n')
                rf.write(f'epoch: {epoch:0>4} | min: {np.mean(list(loss_min.values())):.6f} | avg: {current_mean:.6f}\n')
                rf.write('\n')
            if current_mean < best_loss:
                best_loss = current_mean
                best_epoch = epoch
        scheduler.step()
        
        print('best epoch:',best_epoch,' | ', 'best avg loss',best_loss)
        print('****************************') 

    print('Train Finished!')
    return best_epoch

def stage2(b_epoch):
    torch.backends.cudnn.benchmark = True
    args.pthfile = os.path.join(CKPT_FILE, str(b_epoch)+'_ckpt.pth')
    refiner = Refine(args=args)

    refiner.to(device=DEVICE)
    g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, refiner.parameters()), lr=args.lr,weight_decay=0.0001)

    scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size = args.step_decay, gamma = args.gamma, last_epoch=-1)
    
    TRAIN_DATA = TrainDataLoader(filepath=args.data_list, data_path=args.data_path, pc_input_num=args.input_num,category=args.category)
    TEST_DATA = TestDataLoader(filepath=args.data_list,data_path=args.data_path,status='test', pc_input_num=args.input_num,category=args.category)

    train_loader = DataLoader(TRAIN_DATA,
                              batch_size=args.bz ,
                              num_workers=args.num_workers,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True
                              )
    test_loader = DataLoader(TEST_DATA,
                              batch_size=1 ,
                              num_workers=args.num_workers,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=True
                              )

    resume_epoch = 0 # default 0

    board_writer = SummaryWriter(comment=f'{MODEL}_bs{args.bz}_lr{args.lr}_{args.category}_{TIME_FLAG}')

    if args.resume:
        ckpt_path = os.path.join(CKPT_FILE, '.pth')
        ckpt_dict = torch.load(ckpt_path)
        refiner.load_state_dict(ckpt_dict['model'])
        g_optimizer.load_state_dict(ckpt_dict['optimizer_all'])
        # g_optimizer.param_groups[0]['lr']= 5e-6
        resume_epoch = ckpt_dict['epoch']

    
    os.makedirs(os.path.join(CKPT_RECORD_FOLDER),exist_ok=True)
    
    best_loss = 9999
    best_epoch = 0
    for epoch in range(args.refine_epoch): 
        losses=refine(epoch, refiner,g_optimizer,train_loader,board_writer)
        if epoch % args.eval_epoch ==0 and epoch!=0:
            loss_min, loss_avg = refine_eval(epoch,refiner,test_loader, best_loss)
            current_mean = np.mean(list(loss_avg.values()))
            with open(os.path.join(CKPT_FILE, 'result_refine.txt'), 'a') as rf:
                for cid in loss_min.keys():
                    rf.write(f'epoch: {epoch:0>4} | {cid}: min_CD--{loss_min[cid]:.6f} | avg_CD--{loss_avg[cid]:.6f}\n')
                rf.write(f'epoch: {epoch:0>4} | min: {np.mean(list(loss_min.values())):.6f} | avg: {current_mean:.6f}\n')
                rf.write('\n')
            if current_mean < best_loss:
                best_loss = current_mean
                best_epoch = epoch
        scheduler.step()
        print('best epoch:',best_epoch,' | ', 'best avg loss',best_loss)
        print('****************************') 

    print('Train Finished!')

if __name__ == '__main__':
    best_epoch = stage1()
    best_epoch = 40
    stage2(best_epoch)


    