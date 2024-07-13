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
        print(self.model)
        
        self.start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()).replace(':', '-').replace(' ', '-')
        self.fig_path = os.path.join(config.log.save_path, self.start_time, 'fig')
        self.txt_path = os.path.join(config.log.save_path, self.start_time, 'txt')
        self.model_path = os.path.join(config.log.save_path, self.start_time, 'model')
        for path in [self.fig_path, self.txt_path, self.model_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        
       
        config_data = OmegaConf.create(config)
        OmegaConf.save(config_data, os.path.join(self.txt_path, 'config.yaml'))

    
    
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
                                                                    prev_z=prev_z, mode='test')
                
                angular_error += ContiGaze.Gen_test_log_batch(pred_gaze, gaze)
                count += img.shape[0]
                
                iter_num += 1
                    
                print(f'\r[{iter_num}/{max_iter}][angular_error={angular_error/count * 180/np.pi}]', end='')
        
        error = {'angular_error': angular_error/count * 180/np.pi}
        
        return error
    
    @staticmethod
    def Gen_test_log_batch(gazes, labels):
        gazes = gazes.cpu().detach().numpy()
        errors = funcs.angular_batch(gazes, labels)
        return np.sum(errors)
                
        

        

def trained_model_cross_domain(config):
    GazeModel = ContiGaze(config)
    prev_p = torch.zeros(size=(1,)).to(GazeModel.device) # input prev_p
    prev_z = torch.zeros(size=(1, config.decoder.d_model)).to(GazeModel.device) # input prev_z
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
