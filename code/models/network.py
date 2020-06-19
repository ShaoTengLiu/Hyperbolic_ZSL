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
import json

# Neural Classifierwork

class Image_Transformer(nn.Module):
    def __init__(self, dimension):
        super(Image_Transformer,self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        # self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024,dimension)

    def forward(self,x):
        x = self.fc1(x)
        # x = self.relu(x)
        x = self.fc2(x)
        return x

class MyHingeLoss(torch.nn.Module):

    def __init__(self, margin, dimension):
        super(MyHingeLoss, self).__init__()
        self.M = torch.randn((dimension,dimension), requires_grad=True)
        self.margin = margin

    # TODO the correct implement should set compare_num to a large number
    def forward_val(self, output, target):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        loss = 0
        num_compare = 5
        count = 0
        for i in range(len(output)):
            v_image = output[i]
            t_label = target[i]
            for j in range(num_compare):
                if j != i:
                    count += 1
                    t_j = target[j]
                    loss += torch.relu( self.margin - cos(t_label, v_image) + cos(t_j, v_image) )
        return loss / count

    def forward_train(self, output, target):
        for i in range(len(output)):
            v_image = output[i]
            t_label = target[i]
            j = randint(0, len(output)-1)
            while j == i:
                j = randint(0, len(output)-1)
            t_j = target[j]
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            loss = torch.relu( self.margin - cos(t_label, v_image) + cos(t_j, v_image) )
        return loss