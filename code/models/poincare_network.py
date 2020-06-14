import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from matplotlib import pyplot as plt
from torch.nn.modules.loss import HingeEmbeddingLoss
from random import randint
import geoopt.geoopt.manifolds.poincare.math as pm
import geoopt.geoopt.optim.rsgd as rsgd_
import geoopt.geoopt.optim.radam as radam_
from hyrnn.hyrnn.nets import MobiusLinear
from geoopt.geoopt.tensor import ManifoldParameter
from geoopt.geoopt.manifolds.poincare import PoincareBall
from hype.poincare import PoincareManifold as PM
from tqdm import tqdm
import time

sys.path.append('../../poincare-embeddings')
cos = nn.CosineSimilarity(dim=0, eps=1e-6)

# Neural Classifierwork
class Image_Transformer(nn.Module):
    def __init__(self, dimension):
        super(Image_Transformer,self).__init__()
        self.fc1 = MobiusLinear(2048, 1024)
        self.mobius_relu = pm.mobius_fn_apply(F.relu, x)
        self.fc2 = MobiusLinear(1024, dimension)

    def forward(self,x):
        # x = x / (1 + self.para.norm())
        x = self.fc1(x)
        x = self.mobius_relu(x)
        x = self.fc2(x)
        return x

class MyHingeLoss(torch.nn.Module):
    def __init__(self, margin):
        super(MyHingeLoss, self).__init__()
        self.margin = margin

    # TODO the correct implement should set compare_num to a large number
    # def forward(self, output, target):
    #     loss = 0
    #     compare_num = 2
    #     for i in range(len(output)):
    #         v_image = output[i]
    #         t_label = target[i]
    #         temp = []
    #         while( len(temp) < compare_num ):
    #             j = randint(0, len(output)-1)
    #             while j == i:
    #                 j = randint(0, len(output)-1)
    #             temp.append(j)
    #         for j in temp:
    #             t_j = target[j]
    #             # loss += torch.relu( self.margin + PM().distance(t_label, v_image) - PM().distance(t_j, v_image) )
    #             # loss += torch.relu( self.margin - cos(t_label, v_image)*(1-abs(t_label.norm()-v_image.norm())) + cos(t_j, v_image)*(1-abs(t_j.norm()-v_image.norm())) )
    #             loss += torch.relu( self.margin - cos(t_label, v_image) + cos(t_j, v_image) )
    #     return loss / (len(output)*compare_num)
    def forward(self, output, target):
        loss = 0
        for i in range(len(output)):
            v_image = output[i]
            t_label = target[i]
            j = randint(0, len(output)-1)
            while j == i:
                j = randint(0, len(output)-1)
            t_j = target[j]
            loss += torch.relu( self.margin + \
                            PM().distance(t_label, v_image) - PM().distance(t_j, v_image) )
        return loss / len(output)

class MyHingeLoss_cos(torch.nn.Module):

    def __init__(self, margin):
        super(MyHingeLoss_cos, self).__init__()
        self.margin = margin

    # TODO the correct implement should set compare_num to a large number
    # def forward(self, output, target):
    #     loss = 0
    #     compare_num = 2
    #     for i in range(len(output)):
    #         v_image = output[i]
    #         t_label = target[i]
    #         temp = []
    #         while( len(temp) < compare_num ):
    #             j = randint(0, len(output)-1)
    #             while j == i:
    #                 j = randint(0, len(output)-1)
    #             temp.append(j)
    #         for j in temp:
    #             t_j = target[j]
    #             # loss += torch.relu( self.margin + PM().distance(t_label, v_image) - PM().distance(t_j, v_image) )
    #             # loss += torch.relu( self.margin - cos(t_label, v_image)*(1-abs(t_label.norm()-v_image.norm())) + cos(t_j, v_image)*(1-abs(t_j.norm()-v_image.norm())) )
    #             loss += torch.relu( self.margin - cos(t_label, v_image) + cos(t_j, v_image) )
    #     return loss / (len(output)*compare_num)
    def forward(self, output, target):
        loss = 0
        for i in range(len(output)):
            v_image = output[i]
            t_label = target[i]
            j = randint(0, len(output)-1)
            while j == i:
                j = randint(0, len(output)-1)
            t_j = target[j]
            loss += torch.relu( self.margin - cos(t_label, v_image) + cos(t_j, v_image) )
        return loss / len(output)