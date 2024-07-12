import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, sys
import image_encoder, decoder
import numpy as np
import math
import time
import random
import json
import random
import torch.nn.functional as F
import cv2
import einops
from timm.scheduler import CosineLRScheduler
import torchvision.transforms as T
from sklearn.cluster import KMeans
import pickle
from typing import List, Optional, Tuple, Union

TAG_SAVEPATH = {'MPII': './MPII_tag.pkl',
  'EyeDiap': './EyeDiap_tag.pkl'}

class MLP(nn.Module):
    def __init__(
        self, 
        n_layers: int, 
        d_in: int, d_out: Optional[int] = None,
        use_bn=False, dropout=0.0, dropout_last_layer=False, act=nn.ReLU(),
        d_hidden_factor: int = 4, d_hidden: Optional[int] = None):

        super(MLP, self).__init__()
            
        if d_out is None:
            d_out = d_in
        if d_hidden is None:
            d_hidden = d_hidden_factor * d_in
        assert n_layers >= 0
        if n_layers == 0:
            assert d_in == d_out, f'If n_layers == 0, then d_in == d_out, but got {d_in} != {d_out}'
            self.layers = nn.Identity()
        else:
            current_dim_in = d_in
            layers = []
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(current_dim_in, d_hidden, bias=use_bn))
                if use_bn:
                    layers.append(nn.BatchNorm1d(d_hidden))
                if act is not None:
                    layers.append(act)
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                current_dim_in = d_hidden
            layers.append(nn.Linear(current_dim_in, d_out, bias=use_bn))
            if dropout_last_layer and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *dims, d = x.shape
        if len(dims) > 1:
            x = x.reshape(-1, d)

        x = self.layers(x)
        return x.view(*dims, -1)
                

class CausalIntervention(nn.Module):
    def __init__(self, causal_model_config, version) -> None:
        super().__init__()
        self.version = version
        self.fusion_mode = causal_model_config.fusion_mode
        self.hd_layers = MLP(n_layers=causal_model_config.hd_layers.n_layers, d_in=causal_model_config.hd_layers.d_in,
                             d_out=causal_model_config.hd_layers.d_out, d_hidden_factor=causal_model_config.hd_layers.d_hidden_factor)
        self.Ez_layers = MLP(n_layers=causal_model_config.Ez_layers.n_layers, d_in=causal_model_config.Ez_layers.d_in,
                             d_out=causal_model_config.Ez_layers.d_out, d_hidden_factor=causal_model_config.Ez_layers.d_hidden_factor)
        d_in=causal_model_config.gaze_fc.d_in
        if self.fusion_mode == 'concat':
            d_in = int(d_in * 2)
        self.gaze_fc = MLP(n_layers=causal_model_config.gaze_fc.n_layers, d_in=d_in,
                             d_out=causal_model_config.gaze_fc.d_out, d_hidden_factor=causal_model_config.gaze_fc.d_hidden_factor,
                             dropout=causal_model_config.gaze_fc.dropout, dropout_last_layer=causal_model_config.gaze_fc.dropout_last_layer)
        
    def forward(self, X, W, EZ):
        fx = self.hd_layers(X)
        fez = self.Ez_layers(EZ)
        if self.fusion_mode == 'add':
            gaze_fea = 0.5*W + 0.5*(fx + fez)
        if self.fusion_mode == 'concat':
            gaze_fea = torch.cat((W, fx + fez), dim=1) 

        
        gaze = self.gaze_fc(gaze_fea)
        
        return gaze, gaze_fea
        

class GazeCausalModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.img_encoder = GazeCausalModel.get_model(model_tye='image_encoder', model_config=config.image_encoder)
        self.decoder = GazeCausalModel.get_model(model_tye='decoder', model_config=config.decoder)
        
        self.causal_intervention = CausalIntervention(config.causal_model, version=config.train_setting.version)
        
        self.detr_classifier = nn.Linear(config.decoder.d_model, 3, bias=False)

        self.detr_hd = MLP(n_layers=config.feature_model.detr_hd.n_layers, d_in=config.feature_model.detr_hd.d_in,
                             d_out=config.feature_model.detr_hd.d_out, d_hidden_factor=config.feature_model.detr_hd.d_hidden_factor)
        self.w_gz = MLP(n_layers=config.feature_model.detr_gz.n_layers, d_in=config.feature_model.detr_gz.d_in,
                             d_out=config.feature_model.detr_gz.d_out, d_hidden_factor=config.feature_model.detr_gz.d_hidden_factor)
        self.causal_tokens = nn.Parameter(torch.randn(3, config.decoder.d_model) / math.sqrt(config.decoder.d_model))
    
    def forward(self, face_img, prev_p: torch.FloatTensor, prev_z: torch.FloatTensor):
        img_features, pos_embeddings = self.img_encoder(face_img)
        
        if self.causal_tokens.ndim == 2:
            causal_tokens = self.causal_tokens.repeat(img_features.shape[0], 1, 1)
        
        token_features, assigment_probs, intermediate_features, cross_attentions = self.decoder(
            token_features=causal_tokens,
            region_features=img_features, 
            token_mask=None, 
            region_mask=None, 
            region_pos_embeddings=pos_embeddings,
            return_intermediate=self.config.decoder.return_intermediate,
            return_intermediate_attentions=self.config.decoder.return_intermediate_attentions
        )
        
        x = token_features[:, 0, :]  
        w = token_features[:, 1, :]  
        z = token_features[:, 2, :]  

        
        detr_hd = self.detr_classifier(x)
        detr_gaze = self.detr_classifier(w)
        detr_conf = self.detr_classifier(z)
        
        detr_hd_coord = self.detr_hd(x)

        detr_gz = self.w_gz(w)
        
        Ez = (prev_p * prev_z + x.shape[0] * torch.mean(z, dim=0, keepdim=True)) / (prev_p + x.shape[0]) 
        Ez = Ez.repeat(x.shape[0], 1)
        
        gaze, gaze_fea = self.causal_intervention(x, w, Ez)
        
        return torch.cat([detr_hd, detr_gaze, detr_conf], dim=0), detr_hd_coord, detr_gz, gaze, Ez.detach(), (token_features.detach(), gaze_fea.detach())
    

    
    
    @staticmethod
    def get_model(model_tye: str, model_config):
        if model_tye == 'image_encoder':
            model = image_encoder.image_encoder(extractor_name=model_config.name,
                                                pretrained=model_config.pretrained,
                                                pos_dim=model_config.output_dim)
        elif model_tye == 'decoder':
            model = decoder.TransformerTokenDecoder(
                d_model=model_config.d_model, nhead=model_config.nhead, 
                n_joint_encoder_layers=model_config.n_joint_encoder_layers,
                n_decoder_layers=model_config.n_decoder_layers,
                n_output_encoder_layers=model_config.n_output_encoder_layers,
                act=model_config.act, dropout=model_config.dropout, attention_dropout=model_config.attention_dropout,
                droppath_prob=model_config.droppath_prob,
                layer_scale=model_config.layer_scale, layer_scale_init=model_config.layer_scale_init,
                enc_dec_droppath=model_config.enc_dec_droppath, 
                decoder_sa=model_config.decoder_sa, decoder_ff=model_config.decoder_ff,
                shortcut_tokens=model_config.shortcut_tokens, shortcut_pos_embeddings=model_config.shortcut_pos_embeddings
            )
        
        return model


class loader(Dataset):
    def __init__(self, dataset_name, dataset_paths, dataset_type, image_size):
        self.lines = []
        self.labels = {}
        self.dataset_path = dataset_paths[dataset_name]
        self.type = dataset_type
        self.dataset_name = dataset_name
        self.image_size = image_size

        self.hd_num = 9

        subjects = os.listdir(f'{self.dataset_path}/Image')
        subjects.sort()
        assert dataset_type in ['train', 'test', 'full']
        if dataset_name == 'Gaze360':
            if dataset_type == 'train' or dataset_type == 'test':
                subjects = [dataset_type]
            else:
                subjects = ['test']
        elif dataset_name == 'ETH':
            if self.type == 'train':
                subjects = subjects[:75]
            elif self.type == 'test' or self.type == 'full':
                subjects = subjects[75:]

        elif dataset_name == 'MPII':
            if self.type == 'train':
                subjects = subjects[:-1]
            elif self.type == 'test':
                subjects = subjects[-1:]
        elif dataset_name == 'EyeDiap':
            assert dataset_type == 'full'
        elif dataset_name == 'EyeDiapAll':
            assert dataset_type == 'full'
        elif dataset_name == 'ETHM':
            assert dataset_type == 'full'
            subjects = subjects[75:]
        elif dataset_name == 'Gaze360M':
            assert dataset_type == 'full'

            subjects = ['test']
        else:
            print(f'Reader of dataset [{dataset_name}] not implemented!')
            raise ('Unknown Dataset ERROR')

        self.subjects = subjects
        
        
        for subject in subjects:
            with open(os.path.join(self.dataset_path, 'Label', subject + '.label'), 'r') as label_file:
                label_content = label_file.readlines()
                label_content.pop(0)  
                for label_single_line in label_content:
                    label_single_line = label_single_line.split(' ')
                    if self.dataset_name == 'EyeDiap':
                        label_single_line = [label_single_line[ind] for ind in range(len(label_single_line)) if ind in [0, 6, 7]]
                    elif self.dataset_name == 'Gaze360':
                        label_single_line = [label_single_line[ind] for ind in range(len(label_single_line)) if ind in [0, 5, 6]]
            
                    for i in list(range(1, 3)):
                        label_single_line[i] = list(map(float, label_single_line[i].split(',')))
                    if dataset_name == 'ETHM' or dataset_name == 'Gaze360M':
                        if not (40.5/180*np.pi>abs(label_single_line[1][0]) and -23/180*np.pi<label_single_line[1][1]<5.5/180*np.pi):
                            continue
                    if self.dataset_name == 'Gaze360':
                        if label_single_line[2] == [100., 100.]:
                            continue
                    self.lines.append(label_single_line)
        
        tag_savepath = TAG_SAVEPATH[dataset_name]
        if not os.path.exists(tag_savepath):        
            self.hd_tag = self.head_pose_cluster()
            self.skin_tag = self.skin_color_sample()
            self.id_tag = self.id_sample()
            self.hue_tag = self.hue()
            with open(tag_savepath, 'wb') as file:
                pickle.dump({'hd': self.hd_tag, 'skin': self.skin_tag, 'id': self.id_tag, 'hue': self.hue_tag}, file)
        else:
            with open(tag_savepath, 'rb') as file:
                loaded_dict = pickle.load(file)
            self.hd_tag = loaded_dict['hd']
            self.skin_tag = loaded_dict['skin']
            self.id_tag = loaded_dict['id']
            self.hue_tag = loaded_dict['hue'] 
            

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):

        line = self.lines[idx]
        face_img = cv2.imread(os.path.join(self.dataset_path, 'Image', line[0]))
        if face_img.shape[0] != self.image_size:
            face_img = cv2.resize(face_img, (self.image_size, self.image_size))
        label = line[1]
        label = torch.tensor(label).type(torch.FloatTensor)
        
        face_img = face_img / 255.0
        face_img = face_img.transpose(2, 0, 1)
        face_img = torch.from_numpy(face_img).type(torch.FloatTensor)
        
        img = {"face": face_img,
               "name": line[0]}
        head_pose = torch.tensor(line[2]).type(torch.FloatTensor)
        if self.dataset_name == 'ETH':
            head_pose = head_pose[[1, 0]] # yaw, pitch
        elif self.dataset_name == 'MPII':
            head_pose = head_pose[[0, 1]]
        elif self.dataset_name == 'Gaze360':
            head_pose = torch.arcsin(torch.sin(head_pose))
        elif self.dataset_name == 'EyeDiap':
            head_pose = torch.arcsin(torch.sin(head_pose))
            head_pose[0], head_pose[1] = head_pose[0], -head_pose[1]
       
        tag = self.hd_tag[line[0]] + '_' + self.id_tag[line[0]] + '_' + self.skin_tag[line[0]] + '_' + str(self.hue_tag[line[0]])

        return img, label, head_pose, tag
    
    def hue(self):
        subjects = self.subjects
        
        img_path = []
        for subject in subjects:
            with open(os.path.join(self.dataset_path, 'Label', subject + '.label'), 'r') as label_file:
                label_content = label_file.readlines()
                label_content.pop(0)  
                for label_single_line in label_content:
                    label_single_line = label_single_line.split(' ')
                    img_path.append(label_single_line[0])
        
        hue_value = []
        for img in img_path:
            face_img = cv2.imread(os.path.join(self.dataset_path, 'Image', img))
            v_image = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)[:,:,2]
            hue_value.append(np.mean(v_image))
        
        return dict(zip(img_path, hue_value))

    def head_pose_cluster(self):
        # return {imgpath: label}
        subjects = self.subjects
        
        head_pose_coord = []
        head_pose_path = []
        for subject in subjects:
            with open(os.path.join(self.dataset_path, 'Label', subject + '.label'), 'r') as label_file:
                label_content = label_file.readlines()
                label_content.pop(0) 
                for label_single_line in label_content:
                    label_single_line = label_single_line.split(' ')
                    head_pose_path.append(label_single_line[0])
                    if self.dataset_name == 'EyeDiap':
                        head_pose_coord.append(list(map(float, label_single_line[7].split(','))))
                    elif self.dataset_name == 'Gaze360':
                        head_pose_coord.append(list(map(float, label_single_line[6].split(','))))
                    else:
                        head_pose_coord.append(list(map(float, label_single_line[2].split(','))))
        
        head_pose_coord = np.array(head_pose_coord)
        
        if self.dataset_name == 'ETH':
            head_pose_coord[:, [0, 1]] = head_pose_coord[:, [1, 0]] 
        elif self.dataset_name == 'MPII':
            head_pose_coord = head_pose_coord
        elif self.dataset_name == 'Gaze360':
            head_pose_coord = np.arcsin(np.sin(head_pose_coord))
        elif self.dataset_name == 'EyeDiap':
            head_pose_coord = np.arcsin(np.sin(head_pose_coord))
            head_pose_coord[:, 1] = -head_pose_coord[:, 1]
        
        labels = KMeans(n_clusters=self.hd_num).fit_predict(head_pose_coord)
        hd_labels = np.char.add('hd', labels.astype(int).astype(str)).tolist()       
        head_pose_tag = dict(zip(head_pose_path, hd_labels))
        print("cluster:{}".format([np.sum(np.where(np.array(labels)==x, np.ones_like(labels), np.zeros_like(labels))) for x in np.unique(labels)]))
        
        return head_pose_tag
        
    
    def skin_color_sample(self):
        subjects = self.subjects
        
        skin_info = {}
        for subject in subjects:
            with open(os.path.join(self.dataset_path, 'Image', subject, 'basic_info.txt.txt'), 'r') as label_file:
                label_content = label_file.readline()
                skin_info[subject] = label_content
        
        img_path = []
        for subject in subjects:
            with open(os.path.join(self.dataset_path, 'Label', subject + '.label'), 'r') as label_file:
                label_content = label_file.readlines()
                label_content.pop(0)  
                for label_single_line in label_content:
                    label_single_line = label_single_line.split(' ')
                    img_path.append(label_single_line[0])
                    
        skin = []
        for path in img_path:
            if self.dataset_name == 'ETH':
                if skin_info[path[:11]] == 'white':
                    skin.append('white')
                if skin_info[path[:11]] == 'yellow':
                    skin.append('yellow')
                if skin_info[path[:11]] == 'black':
                    skin.append('black')
            elif self.dataset_name == 'MPII':
                if skin_info[path[:3]] == 'white':
                    skin.append('white')
                if skin_info[path[:3]] == 'yellow':
                    skin.append('yellow')
                if skin_info[path[:3]] == 'black':
                    skin.append('black')
            elif self.dataset_name == 'EyeDiap':
                if skin_info[path.split('/')[0]] == 'white':
                    skin.append('white')
                if skin_info[path.split('/')[0]] == 'yellow':
                    skin.append('yellow')
                if skin_info[path.split('/')[0]] == 'black':
                    skin.append('black')
            elif self.dataset_name == 'Gaze360':
                skin.append('none')

        skin_tag = dict(zip(img_path, skin))
        
        return skin_tag
            
    def id_sample(self):
        subjects = self.subjects
        
        img_path = []
        for subject in subjects:
            with open(os.path.join(self.dataset_path, 'Label', subject + '.label'), 'r') as label_file:
                label_content = label_file.readlines()
                label_content.pop(0)  
                for label_single_line in label_content:
                    label_single_line = label_single_line.split(' ')
                    img_path.append(label_single_line[0])        
        id_ = []
        ind = {}
        for i, sub in enumerate(subjects):
            ind[sub] = 'id' + str(int(i))
        
        for path in img_path:
            if self.dataset_name == 'ETH':
                id_.append(ind[path[:11]])
            elif self.dataset_name == 'MPII':
                id_.append(ind[path[:3]])
            elif self.dataset_name == 'EyeDiap':
                id_.append(ind[path.split('/')[0]])
        
        id_tag = dict(zip(img_path, id_))
        
        return id_tag



def get_dataloader(main_dataconfig, special_dataconfig):
    dataset_name = special_dataconfig.dataset_name
    dataset_path = main_dataconfig.datapath
    dataset_type = special_dataconfig.type
    image_size = main_dataconfig.image_size
    batch_size = special_dataconfig.batch_size
    shuffle= special_dataconfig.shuffle
    num_workers = special_dataconfig.num_workers
    drop_last = special_dataconfig.drop_last
    print(f"[{dataset_name} Dataset] {dataset_type} Set Loading......")
    print(f"[[{dataset_name} Dataset] Path: [{dataset_path[dataset_name]}]")
    dataset = loader(dataset_name, dataset_path, dataset_type, image_size)
    print(f"[{dataset_name} Dataset] Data Loaded! [Image Num: {len(dataset)}][Batch: {batch_size}][Size:{image_size}][Shuffle:{shuffle}]")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    return load



def set_seed(seed):
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed,details are ", e)
        pass
    np.random.seed(seed)
    random.seed(seed)
    # cuda env
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def gazeto3d_batch(gaze):
    gaze_gt = torch.zeros((gaze.shape[0], 3)).to(gaze.device)
    gaze_gt[:, 0] = -torch.cos(gaze[:, 1]) * torch.sin(gaze[:, 0])
    gaze_gt[:, 1] = -torch.sin(gaze[:, 1])
    gaze_gt[:, 2] = -torch.cos(gaze[:, 1]) * torch.cos(gaze[:, 0])
    return gaze_gt


def gazeto3d(gaze):
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt

def gazeto3d_head(gaze):
    gaze_gt = torch.zeros([3,])
    gaze_gt[0] = -torch.cos(gaze[1]) * torch.sin(gaze[0])
    gaze_gt[1] = -torch.sin(gaze[1])
    gaze_gt[2] = -torch.cos(gaze[1]) * torch.cos(gaze[0])
    return gaze_gt

def gazeto2d_head(gaze):
    gaze_gt = torch.zeros((2,))
    gaze_gt[1] = torch.arcsin(-gaze[1])
    gaze_gt[0] = torch.arcsin(-gaze[0] / torch.cos(gaze_gt[1]))
    return gaze_gt

def angular(gaze, label):
    total = np.sum(gaze * label)
    return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi


def dis(p1, p2):
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def build_scheduler(optimizer, config):
    num_steps = int(config.train_setting.max_steps)
    warmup_steps = int(config.train_setting.warmup_steps)

    return CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=config.train_setting.min_lr,
        warmup_lr_init=config.train_setting.warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )




