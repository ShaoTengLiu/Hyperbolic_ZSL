import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from devise import Image_Transformer
from nltk.corpus import wordnet as wn
from matplotlib import pyplot as plt

# edit path here to load pretrained features
val_img_mat_path = '../checkpoints/Imagenet/val/val_image_mat_resnet.npy'
val_words_embd_path = '../checkpoints/Imagenet/val/val_words_embd_glove_300.npy'
val_classes_embd_path = '../checkpoints/Imagenet/classes/imagenet_classes_embd_glove_300.npy'
val_parent_words_embd_path = '../checkpoints/Imagenet/val/val_parent_words_embd_glove_300.npy'
val_parent_classes_embd_path = '../checkpoints/Imagenet/classes/imagenet_parent_classes_embd_glove_300.npy'

model_path = '../checkpoints/model.pt'

val_img_mat = np.load(val_img_mat_path)
val_words_embd = np.load(val_words_embd_path)
val_classes_embd = np.load(val_classes_embd_path)
val_img_mat = torch.from_numpy(val_img_mat).type(torch.FloatTensor)
val_words_embd = torch.from_numpy(val_words_embd).type(torch.FloatTensor)
val_classes_embd = torch.from_numpy(val_classes_embd).type(torch.FloatTensor)

model = torch.load(model_path).cuda()
val_img_output = model(val_img_mat)
model.eval()

res = np.zeros(len(val_img_output))
cos = nn.CosineSimilarity(dim=0, eps=1e-6)
ranks = []
for i in tqdm( range(len(val_img_output)) ):
	rank = 1
	target = cos(val_img_output[i], val_words_embd[i])
	for j in range(len(val_classes_embd)):
		if( target < cos(val_img_output[i], val_classes_embd[j]) ):
			rank += 1
	ranks.append(rank)
	print("Input: {}, Target_coss: {}, Rank: {}, Mean_Rank(E): {}".format( i, target, rank, sum(ranks)/len(ranks) ) )

mean_rank = sum(ranks)/len(ranks)
hit_1 = ranks.count(1)
hit_2 = ranks.count(1) + ranks.count(2)
hit_5 = sum([ranks.count(i) for i in range(1, 6)])
hit_10 = sum([ranks.count(i) for i in range(1, 11)])
hit_50 = sum([ranks.count(i) for i in range(1, 51)])
hit_100 = sum([ranks.count(i) for i in range(1, 101)])

print("mean_rank(E): {}, hit_1: {}, hit_2: {}, hit_5: {}, hit_10: {}, hit_50: {}, hit_100: {}"\
	.format( mean_rank, hit_1, hit_2, hit_5, hit_10, hit_50, hit_100 ) )

# parent evaluation
print('Parent Evaluation')
val_parent_words_embd = np.load(val_parent_words_embd_path)
val_parent_words_embd = torch.from_numpy(val_parent_words_embd).type(torch.FloatTensor)

val_parent_classes_embed = np.load(val_parent_classes_embd_path)
val_parent_classes_embed = torch.from_numpy(val_parent_classes_embed).type(torch.FloatTensor)

ranks = []
for i in tqdm( range(len(val_img_output)) ):
	rank = 1
	target = cos(val_img_output[i], val_parent_words_embd[i])
	for j in range(len(val_classes_embd)):
		if( target < cos(val_img_output[i], val_classes_embd[j]) ):
			rank += 1
	for j in range(len(val_parent_classes_embed)):
		if( target < cos(val_img_output[i], val_parent_classes_embed[j]) ):
			rank += 1
	ranks.append(rank)
	print("Input: {}, Target_coss: {}, Rank: {}, Mean_Rank(PE): {}".format( i, target, rank, sum(ranks)/len(ranks) ) )

mean_rank = sum(ranks)/len(ranks)
hit_1 = ranks.count(1)
hit_2 = ranks.count(1) + ranks.count(2)
hit_5 = sum([ranks.count(i) for i in range(1, 6)])
hit_10 = sum([ranks.count(i) for i in range(1, 11)])
hit_50 = sum([ranks.count(i) for i in range(1, 51)])
hit_100 = sum([ranks.count(i) for i in range(1, 101)])

print("mean_rank(PE): {}, hit_1: {}, hit_2: {}, hit_5: {}, hit_10: {}, hit_50: {}, hit_100: {}"\
	.format( mean_rank, hit_1, hit_2, hit_5, hit_10, hit_50, hit_100 ) )