import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
import cv2
import math
from sklearn import manifold
import sklearn
import time
import torch
from itertools import chain
import pickle
import json

ETH_path = ".eth"
Gaze360_path = "./Gaze360/"
MPII_path = "./MPIIFaceGaze/"
EyeDiap_path = "./EyeDiap/"
dataset_paths = {'ETH': ETH_path,
                 'Gaze360': Gaze360_path,
                 'MPII': MPII_path,
                 'EyeDiap': EyeDiap_path}


def gazeTo2d(gaze):
    try:
        yaw = np.arctan2(-gaze[0], -gaze[2])
    except:
        print(gaze)
        exit()

    pitch = np.arcsin(np.clip(-gaze[1], -1, 1))
    return np.array([yaw, pitch])


def gazeto3d(gaze):
    assert len(gaze)==2
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt


def angular(gaze, label):
    if len(gaze)==2:
        gaze = gazeto3d(gaze)

    if len(label)==2:
        label = gazeto3d(label)
    total = np.sum(gaze * label)

    return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

def gazeTo3d_array(gaze):
    assert len(gaze.shape) == 2
    assert gaze.shape[1] == 2
    gaze_gt = np.zeros((gaze.shape[0], 3))
    gaze_gt[:, 0] = -np.cos(gaze[:, 1]) * np.sin(gaze[:, 0])
    gaze_gt[:, 1] = -np.sin(gaze[:, 1])
    gaze_gt[:, 2] = -np.cos(gaze[:, 1]) * np.cos(gaze[:, 0])
    return gaze_gt

def gazeTo2d_array(gaze):
    assert len(gaze.shape) == 2
    assert gaze.shape[1] == 3
    yaw = np.arctan2(-gaze[:, 0], -gaze[:, 2]).reshape((gaze.shape[0], 1))


    pitch = np.arcsin(-gaze[:, 1]).reshape((gaze.shape[0], 1))
    return np.hstack((yaw, pitch))



def angular_batch(gaze, label):
    if gaze.shape[1]==2:
        gaze = gazeTo3d_array(gaze)
    if label.shape[1] == 2:
        label = gazeTo3d_array(label)
    total = np.sum(gaze * label, axis=1)
    return np.arccos(np.clip(total/(np.linalg.norm(gaze, ord=2, axis=1)* np.linalg.norm(label, ord=2, axis=1)), -0.99999999, 0.99999999))

def AngularError(gaze, label):
    total = torch.sum(gaze * label, 1)
    cos_v = total/(torch.linalg.norm(gaze, 2, dim=1)*torch.linalg.norm(label, 2, dim=1))
    cos_v = cos_v - torch.clamp(cos_v - 1 + 1e-7, 0, 1)
    cos_v = cos_v - torch.clamp(cos_v + 1 - 1e-7, -1, 0)
    return torch.mean(torch.acos(cos_v))


def AngularError_array(gaze, label):
    if gaze.shape[1] == 2:
        gaze = gazeto3d_batch(gaze)
    if label.shape[1] == 2:
        label  = gazeto3d_batch(label)

    total = torch.sum(gaze * label, 1)

    cos_v = total/(torch.linalg.norm(gaze, 2, dim=1)*torch.linalg.norm(label, 2, dim=1))
    cos_v = cos_v - torch.clamp(cos_v - 1 + 1e-7, 0, 1)
    cos_v = cos_v - torch.clamp(cos_v + 1 - 1e-7, -1, 0)
    return torch.acos(cos_v)

def gazeto3d_batch(gaze):
    gaze_gt = torch.zeros((gaze.shape[0], 3)).to(gaze.device)
    gaze_gt[:, 0] = -torch.cos(gaze[:, 1]) * torch.sin(gaze[:, 0])
    gaze_gt[:, 1] = -torch.sin(gaze[:, 1])
    gaze_gt[:, 2] = -torch.cos(gaze[:, 1]) * torch.cos(gaze[:, 0])
    return gaze_gt


def ReadLogFile(path, split_icon=',', if3D=False):
    path_split = path.split('/')
    names = []
    gazes = []
    labels = []
    errors = []
    with open(path, 'r') as f:
        lines = f.readlines()
        head = lines[0]
        error = lines[-1]
        lines.pop(0)
        lines.pop(-1)

        for line in lines:
            line = line.strip().split(split_icon)
            names.append(line[0])
            if not if3D:
                gazes.append([float(line[1]), float(line[2])])
                labels.append([float(line[3]), float(line[4])])
                errors.append(float(line[5]))
            else:
                gazes.append(gazeTo2d([float(line[1]), float(line[2]), float(line[3])]))
                labels.append([float(line[4]), float(line[5])])
                errors.append(float(line[6]))
    print(f'Log:[{path_split[-2]}] [{path_split[-1]}] Error:[{np.mean(errors):^6.2f}]')
    return names, np.array(gazes), np.array(labels), np.array(errors)

def ReadLogFile_3D(path, split_icon=',', print_log=True):
    names = []
    gazes = []
    labels = []
    errors = []
    with open(path, 'r') as f:
        lines = f.readlines()
        head = lines[0]
        error = lines[-1]
        lines.pop(0)
        if print_log:
            print(lines[-1], end='')
        lines.pop(-1)

        for line in lines:
            line = line.strip().split(split_icon)
            names.append(line[0])
            gazes.append(gazeTo2d([float(line[1]), float(line[2]), float(line[3])]))
            labels.append([float(line[4]), float(line[5])])
            errors.append(float(line[6]))
    return names, np.array(gazes), np.array(labels), np.array(errors)

def draw_gaze(image_in, pos, yawpitch, length=100.0, thickness=1, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(yawpitch[0]) * np.cos(yawpitch[1])
    dy = -length * np.sin(yawpitch[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out



def HeatError(x, y, error, title='Heat Map'):
    x = x*180/np.pi
    y = y*-180/np.pi
    HEIGHT = math.ceil(max(y))-math.floor(min(y))+1
    WIDTH = math.ceil(max(x))-math.floor(min(x))+1

    heat = np.zeros((HEIGHT, WIDTH))
    count = np.zeros((HEIGHT, WIDTH))
    y_bias = math.floor(min(y))
    x_bias = math.floor(min(x))

    for i in range(len(x)):
        if error[i]<1000:

            heat[math.floor(y[i])-y_bias][math.floor(x[i])-x_bias] += error[i]
            count[math.floor(y[i])-y_bias][math.floor(x[i])-x_bias] += 1
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if heat[i][j]==0:
                heat[i][j] = float('nan')
            else:
                heat[i][j] /= count[i][j]
    plt.figure()

    plt.imshow(heat, interpolation='nearest')#, cmap=plt.cm.hot,
               #norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()
    gap_x = WIDTH//6
    gap_y = HEIGHT//6
    x_stick = list(range(0, WIDTH, gap_x))
    x_label = np.array(x_stick)+x_bias
    y_stick = list(range(0, HEIGHT, gap_y))
    y_label = -(np.array(y_stick)+y_bias)

    plt.xticks(x_stick, x_label)
    plt.yticks(y_stick, y_label)


    plt.title(title)
    plt.show()




def ErrorOneAxis(prediction, label, title='', bound_x=None, bound_y=None):
    prediction = prediction*180/np.pi
    label = label * 180 / np.pi

    plt.figure()
    if not bound_x is None:
        plt.xlim(bound_x)
    if not bound_y is None:
        plt.ylim(bound_y)
    plt.scatter(label, prediction, s=2, c='r', alpha=0.3)
    plt.plot([np.min(label), np.max(label)], [np.min(label), np.max(label)], c='b')
    plt.title(title)
    plt.show()


def PrintErrorAfterMove(model_name, target='MPII', max_epoch=10, prefix='Evaluation'):
    base_dir = 'E:/Experiments/Heatmap/Evaluation/'

    dir = base_dir + model_name

    print(f'[Model Name:{model_name}] [Test Dataset:{target}]')
    for epoch in range(1, max_epoch+1):
        names, gazes, labels, errors = ReadLogFile_3D(
            dir + f'[{prefix}][ETH-{target}][Epoch{str(epoch).zfill(2)}].log', print_log=False)

        bias = np.mean(labels - gazes, 0)
        new_error = []
        for i in range(len(errors)):
            new_error.append(angular(gazeto3d(gazes[i] + bias), gazeto3d(labels[i])))
        print(f'[Epoch:{str(epoch).zfill(2)}] Before:{np.mean(errors):^5.2f}  After:{np.mean(new_error):^5.2f}')


def ErrorOfAxis(x, error, gap=10):

    xmin = min(x)
    xmax = max(x)

    length = (math.ceil(xmax) - math.floor(xmin))//gap + 1
    line_y = np.zeros(length)
    count_y = np.zeros(length)
    for i in range(len(x)):

        line_y[int((x[i]-xmin)//gap)] += abs(error[i])
        count_y[int((x[i]-xmin)//gap)] += 1
    for i in range(len(line_y)):
        if count_y[i] > 0:
            line_y[i] /= count_y[i]
    line_x = list(range(length))


    x_label = np.array(list(range(math.floor(xmin), math.ceil(xmax)+1, gap)))

    indexes = [x for x in range(0, len(x_label), len(x_label)//7)]

    plt.plot(x_label, line_y)

    plt.show()

