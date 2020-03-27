import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from devise_all import Image_Transformer
from nltk.corpus import wordnet as wn
from matplotlib import pyplot as plt

# classes_embd_path = '/home/jjchen/LST/CV/Imagenet/classes_words_embd.npy'
# classes_embd = np.load(classes_embd_path)
# classes_embd = torch.from_numpy(classes_embd).type(torch.FloatTensor)

val_words_embd_path = '/home/jingjing/lst/CV/Imagenet/all/val/val_words_embd_glove_300.npy'
# 200 classes
val_words_embd_200_path = '/home/jingjing/lst/CV/Imagenet/classes/imagenet_classes_embd_glove_300.npy'
val_img_mat_path = '/home/jingjing/lst/CV/Imagenet/all/val/val_image_mat_resnet.npy'
val_words_embd = np.load(val_words_embd_path)
val_words_embd_200 = np.load(val_words_embd_200_path)
val_img_mat = np.load(val_img_mat_path)

val_words_embd = torch.from_numpy(val_words_embd).type(torch.FloatTensor)
val_words_embd_200 = torch.from_numpy(val_words_embd_200).type(torch.FloatTensor)
val_img_mat = torch.from_numpy(val_img_mat).type(torch.FloatTensor)

model = torch.load('./model/model_814.pt').cpu()
val_img_output = model.forward(val_img_mat)
model.eval()
# from sklearn.decomposition import PCA
# pca = PCA(n_components=300)
# val_img_output = pca.fit_transform(val_img_mat)
# val_img_output = torch.from_numpy(val_img_output).type(torch.FloatTensor)

res = np.zeros(len(val_img_output))

cos = nn.CosineSimilarity(dim=0, eps=1e-6)

ranks = []
for i in tqdm( range(len(val_img_output)) ):

	rank = 1
	target = cos(val_img_output[i], val_words_embd[i])
	for j in range(len(val_words_embd_200)):
		if( target < cos(val_img_output[i], val_words_embd_200[j]) ):
			rank += 1
	ranks.append(rank)
	print("Input: {}, Target_coss: {}, Rank: {}, Mean_Rank: {}".format( i, target, rank, sum(ranks)/len(ranks) ) )

# # parent evaluation
# val_parent_words_embd_path = '/home/jjchen/LST/CV/Imagenet/CS231n/tiny-imagenet-200/val/val_parent_words_embd.npy'
# val_parent_words_embd = np.load(val_parent_words_embd_path)
# val_parent_words_embd = torch.from_numpy(val_parent_words_embd).type(torch.FloatTensor)

# img_classes_embed_path = '/home/jjchen/LST/CV/Imagenet/classes_words_embd.npy'
# img_classes_embed = np.load(img_classes_embed_path)
# img_classes_embed = torch.from_numpy(img_classes_embed).type(torch.FloatTensor)

# img_parent_classes_embed_path = '/home/jjchen/LST/CV/Imagenet/classes_parent_words_embd.npy'
# img_parent_classes_embed = np.load(img_parent_classes_embed_path)
# img_parent_classes_embed = torch.from_numpy(img_parent_classes_embed).type(torch.FloatTensor)

# ranks = []
# for i in tqdm( range(len(val_img_output)) ):

# 	rank = 1
# 	target = cos(val_img_output[i], val_parent_words_embd[i])
# 	# for j in range(len(img_classes_embed)):
# 	# 	if( target < cos(val_img_output[i], img_classes_embed[j]) ):
# 	# 		rank += 1
# 	# for j in range(len(img_parent_classes_embed)):
# 	# 	if( target < cos(val_img_output[i], img_parent_classes_embed[j]) ):
# 	# 		rank += 1
# 	ranks.append(rank)
# 	print("Input: {}, Target_coss: {}, Rank: {}, Mean_Rank: {}".format( i, target, rank, sum(ranks)/len(ranks) ) )


mean_rank = sum(ranks)/len(ranks)
hit_1 = ranks.count(1)
hit_2 = ranks.count(1) + ranks.count(2)
hit_5 = sum([ranks.count(i) for i in range(1, 6)])
hit_10 = sum([ranks.count(i) for i in range(1, 11)])
hit_50 = sum([ranks.count(i) for i in range(1, 51)])
hit_100 = sum([ranks.count(i) for i in range(1, 101)])

print("mean_rank: {}, hit_1: {}, hit_2: {}, hit_5: {}, hit_10: {}, hit_50: {}, hit_100: {}"\
	.format( mean_rank, hit_1, hit_2, hit_5, hit_10, hit_50, hit_100 ) )

# # draw the picture of rank
# rank_img_path = './pictures/rank_81.jpg'
# counter = [i for i in range(len(val_img_output))]
# plt.title('Ranks')
# plt.xlabel('Pictures')
# plt.ylabel('Ranks')

# plt.plot(counter, ranks,'r',label='rank')
 
# plt.grid()
# plt.savefig(rank_img_path)