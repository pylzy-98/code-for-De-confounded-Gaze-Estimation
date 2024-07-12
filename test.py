import torch
import torch.nn as nn
import os, sys
from train_utils import *
import numpy as np
import math
import time
import random
import DataProcessFuncs as funcs
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
import palettable
import cv2
from tqdm import tqdm
import hydra
from torch import autocast
from torch.cuda.amp import GradScaler
from omegaconf import OmegaConf


class ContiGaze():
    def __init__(self, config) -> None:
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device('cuda', config.train_setting.device)
        else:
            self.device = torch.device('cpu')
        
        self.model = GazeCausalModel(config).to(self.device)
        self.prev_p = torch.zeros(size=(1,)).to(self.device)
        self.prev_z = torch.zeros(size=(1, config.decoder.d_model)).to(self.device)
        print(self.model)
        self.gaze_loss = nn.L1Loss().to(self.device)
        self.detr_cls_loss = nn.CrossEntropyLoss().to(self.device)
        self.detr_hd_loss = nn.L1Loss().to(self.device)
        
        
        causal_model_train_name = ['causal_intervention']
        causal_model_optim_params = []
        normal_model_optim_params = []
        for name, para in self.model.named_parameters():
            if name.split('.')[0] in causal_model_train_name:
                causal_model_optim_params.append({'params': para, 'lr': config.train_setting.causal_model_lr})
            else:
                normal_model_optim_params.append({'params': para, 'lr': config.train_setting.normal_model_lr})
        self.normal_optim = torch.optim.Adam(normal_model_optim_params, lr=config.train_setting.normal_model_lr, betas=(0.5, 0.95))
        self.causal_model_optim = torch.optim.Adam(causal_model_optim_params, lr=config.train_setting.causal_model_lr, betas=(0.5, 0.95))

        
        self.start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()).replace(':', '-').replace(' ', '-')
        self.fig_path = os.path.join(config.log.save_path, self.start_time, 'fig')
        self.txt_path = os.path.join(config.log.save_path, self.start_time, 'txt')
        self.model_path = os.path.join(config.log.save_path, self.start_time, 'model')
        for path in [self.fig_path, self.txt_path, self.model_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        
       
        config_data = OmegaConf.create(config)
        OmegaConf.save(config_data, os.path.join(self.txt_path, 'config.yaml'))

            
    def train(self, train_dataloader, now_epoch):
        print(f'[Meta Training][Epoch {now_epoch:^3}/{self.config.train_setting.epoch:^3}] starting...', end='')
        self.model.train()
        self.prev_p = torch.zeros(size=(1,)).to(self.device)
        total_loss = {'gaze_loss': 0., 'detr_loss': 0.}
        with tqdm(train_dataloader, unit='batch') as loader:
            total_batch_num = len(train_dataloader)
            loader.set_description(f'Training: {now_epoch:^3}/{self.config.train_setting.epoch:^3}')
            for ind, (face, gaze, head_pose, tag) in enumerate(loader):
                img = face['face'].to(self.device)
                gaze = gaze.to(self.device)
                head_pose = head_pose.to(self.device)
                detr_cls_labels = self.detr_cls_label(num=img.shape[0], is_onehot=False)
                losses = {}
                with autocast(device_type='cuda', enabled=True):
                    detr_cls, detr_hd_coord, detr_gz, pred_gaze, Ez, _ = self.model(img, prev_p=self.prev_p, prev_z=self.prev_z)
                    losses['cls_loss'] = self.detr_cls_loss(detr_cls, detr_cls_labels) * self.config.train_setting.cls_loss_coff
                    losses['detr_hd_loss'] = self.detr_hd_loss(detr_hd_coord, head_pose) * self.config.train_setting.hd_loss_coff
                    if self.config.train_setting.using_detr_gz_loss and self.config.train_setting.version == 'train_1':
                        losses['detr_gz_loss'] = self.gaze_loss(detr_gz, gaze) * self.config.train_setting.detr_gz_coff
                    
                    losses['gaze_loss'] = self.gaze_loss(pred_gaze, gaze) * self.config.train_setting.gaze_loss_coff
                    loss = sum(l for l in losses.values() if l is not None) / self.config.train_setting.accumulation_steps
                    
                    
                loss.backward()

                if (ind + 1) % self.config.train_setting.accumulation_steps == 0:
                    if self.config.train_setting.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    self.config.train_setting.grad_clip_norm)
                    self.normal_optim.step()
                    self.causal_model_optim.step()
                    
                    self.normal_optim.zero_grad()
                    self.causal_model_optim.zero_grad()

                self.prev_z = Ez[0, ...].reshape(1, -1)
                self.prev_p += Ez.shape[0]
                
                
                gaze_loss = losses['gaze_loss'].detach().cpu().numpy() / self.config.train_setting.gaze_loss_coff
                cls_loss = losses['cls_loss'].detach().cpu().numpy() / self.config.train_setting.cls_loss_coff
                detr_hd_loss = losses['detr_hd_loss'].detach().cpu().numpy() / self.config.train_setting.hd_loss_coff
                loader.set_postfix(gaze_loss=f'{gaze_loss:.4f}')
                total_loss['gaze_loss'] += gaze_loss
                total_loss['detr_loss'] += (cls_loss + detr_hd_loss)
         
        total_loss['gaze_loss'] = total_loss['gaze_loss'] / total_batch_num
        total_loss['detr_loss'] = total_loss['detr_loss'] / total_batch_num
        
        return total_loss  
    
    def test(self, test_dataloader, prev_p, prev_z, style='test', model_state_dict=None, max_iter=88,):
        print(f'[Testing] {style} starting...')
        if model_state_dict is not None:
            state_dict = torch.load(model_state_dict)['model_state_dict']
            self.model.load_state_dict(state_dict)
        self.model.eval()
        
        
            
        count = 0
        angular_error = 0
        iter_num = 0

        with torch.no_grad():
            for _, (face, gaze, head_pose, tag) in enumerate(test_dataloader):
                img = face['face'].to(self.device)  
                
                detr_cls, detr_hd_coord, detr_gz, pred_gaze, _, (token_features, gaze_fea) = self.model(img, prev_p=prev_p,
                                                                    prev_z=prev_z)
                
                angular_error += ContiGaze.Gen_test_log_batch(pred_gaze, gaze)
                count += img.shape[0]
                
                iter_num += 1
                    
                print(f'\r[{iter_num}/{max_iter}][angular_error={angular_error/count * 180/np.pi}]', end='')
        
        error = {'angular_error': angular_error/count * 180/np.pi}
        
        return error
    
    
    def detr_cls_label(self, num, is_onehot=False):
        hd_labels = torch.zeros(size=(num, ), dtype=torch.long)
        gaze_labels = torch.ones(size=(num, ), dtype=torch.long)
        conf_labels = torch.ones(size=(num, ), dtype=torch.long) + 1
        
        labels = torch.cat([hd_labels, gaze_labels, conf_labels])
        if is_onehot:
            labels = nn.functional.one_hot(labels, num_classes=3)
        
        return labels.to(self.device)
    
    @staticmethod
    def Gen_test_log_batch(gazes, labels):
        gazes = gazes.cpu().detach().numpy()
        errors = funcs.angular_batch(gazes, labels)
        return np.sum(errors)
                
        

        

def trained_model_cross_domain(config):
    GazeModel = ContiGaze(config)
    prev_p = torch.zeros(size=(1,)).to(GazeModel.device)
    prev_z = torch.zeros(size=(1, config.decoder.d_model)).to(GazeModel.device)
    test_loader = get_dataloader(config.data, config.data.testset_setting)
    
    model_state_dict = f'model_pth/resnet18_ETH.pth'
    error = GazeModel.test(test_loader, model_state_dict=model_state_dict, prev_p=prev_p, prev_z=prev_z)
    print(error)

        
@hydra.main(config_path='./config', config_name='causality', version_base=None)
def do_running(config):
    seed_everything(config.train_setting.random_seed)
    sys.stdout.flush()
    
    trained_model_cross_domain(config)
    
    
if __name__ == '__main__':
    do_running()