import sys
import os
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
import argparse

from models.poincare_network import Image_Transformer, MyHingeLoss

sys.path.append('../poincare-embeddings')

def parse_option(): 

	parser = argparse.ArgumentParser('argument for training')

	parser.add_argument('--model_folder', type=str, default='./results/model/hype/')
	parser.add_argument('--loss_path', type=str, default='./results/loss/loss_model.jpg')
	parser.add_argument('--train_img_mat_path', type=str, default='../data/train/img/train_image_mat_resnet50.npy')
	parser.add_argument('--val_img_mat_path', type=str, default='../data/val/img/val_image_mat_resnet50.npy')
	parser.add_argument('--word_model', type=str, default='poincare')
	
	parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
	parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
	parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
	parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
	parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
	parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
	parser.add_argument('--mode', type=str, default='normal')
	parser.add_argument('--dimension', type=int, default=100)
	opt = parser.parse_args()

	if opt.mode == 'tiny_test':
		opt.model_folder = './results/tiny/model/hype/'
		opt.loss_path = './results/tiny/loss/loss_model.jpg'
	opt.model_folder = opt.model_folder + opt.word_model
	opt.model_folder = os.path.join(opt.model_folder, str(opt.dimension))
	if not os.path.isdir(opt.model_folder):
		os.makedirs(opt.model_folder)
	# data
	if opt.word_model == 'poincare': # TODO 1.get more pretrained embedding; 2.this code should be designed as multi-functional(pe, pg, pepg)
	   opt.train_words_embd_path = '../data/train/label/Hype_version/train_words_embd_pe_300.npy'
	   opt.val_words_embd_path   = '../data/val/label/Hype_version/val_words_embd_pe_300.npy'
	else: 
		opt.train_words_embd_path = ''
		opt.val_words_embd_path   = ''
	
	if opt.mode == 'tiny_test':
	   opt.train_img_mat_path    = '../data/train/img/tiny-imagenet-200/train_embed/train_image_mat_resnet.npy'
	   opt.train_words_embd_path = '../data/train/label/tiny-imagenet-200/train_embed/Hype_version/train_words_pe_dim.npy'
	   opt.val_img_mat_path      = '../data/val/img/tiny-imagenet-200/val_embed/val_image_mat_resnet.npy'
	   opt.val_words_embd_path   = '../data/val/label/tiny-imagenet-200/val_embed/Hype_version/val_words_pe_dim.npy'
	   opt.train_words_embd_path = opt.train_words_embd_path.replace('dim', str(opt.dimension))
	   opt.val_words_embd_path = opt.val_words_embd_path.replace('dim', str(opt.dimension))
	
	opt.loss_path = opt.loss_path.replace('model', opt.word_model+str(opt.dimension))
	return opt

def get_train_val_data(args): 
	# train
	train_words_embd = np.load(args.train_words_embd_path)
	train_img_mat    = np.load(args.train_img_mat_path)
	#val
	val_words_embd = np.load(args.val_words_embd_path)
	val_img_mat    = np.load(args.val_img_mat_path)

	val_words_embd = torch.from_numpy(val_words_embd).type(torch.FloatTensor)
	val_img_mat    = torch.from_numpy(val_img_mat).type(torch.FloatTensor)

	train_words_embd, train_img_mat = shuffle(train_words_embd, train_img_mat, random_state=233)
	return train_words_embd, train_img_mat, val_words_embd, val_img_mat

def set_model(args): 
	model     = Image_Transformer(args.dimension)
	criterion = MyHingeLoss(2)
	return model, criterion

def set_optimizer(args,model): 
	# optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
	optimizer = radam_.RiemannianAdam(model.parameters(), lr=0.01, stabilize=10)
	return optimizer

def main():
	args = parse_option()

	train_words_embd, train_img_mat, val_words_embd, val_img_mat = get_train_val_data(args)
	print('Get Data!')
	# Train
	model, criterion = set_model(args)
	optimizer  = set_optimizer(args, model)

	model          = model.cuda()
	criterion      = criterion.cuda()
	val_words_embd = val_words_embd.cuda()
	val_img_mat    = val_img_mat.cuda()

	total_step = 0

	train_losses = []
	train_f1s    = []
	test_losses  = []
	test_f1s     = []
	counter      = []

	for epoch in range(args.epochs): 
		model.train()
		losses   = []
		f1s      = []
		avg_loss = 0
		f1       = 0

		# # tensor shuffle
		# shuffle_index       = torch.randperm(len(train_img_mat))
		# train_img_mat_sf    = train_img_mat
		# train_words_embd_sf = train_words_embd
		# for i in range(len(train_img_mat)): 
		# train_img_mat_sf[i]    = train_img_mat[shuffle_index[i]]-1
		# train_words_embd_sf[i] = train_words_embd[shuffle_index[i]]-1

		for j in range(0, len(train_img_mat), args.batch_size): 
			total_step        += 1
			batch_train_img    = train_img_mat[j: j+args.batch_size, :]
			batch_train_words  = train_words_embd[j: j+args.batch_size, :]
			batch_train_img    = torch.from_numpy(batch_train_img).type(torch.FloatTensor).cuda()
			batch_train_words  = torch.from_numpy(batch_train_words).type(torch.FloatTensor).cuda()

			batch_train_img = pm.expmap0(batch_train_img, c=1)
			img_tranf = model( batch_train_img )
			# y_pred    = y_pred.squeeze(1)

			loss = criterion(img_tranf, batch_train_words)
			losses.append(loss.item())
			avg_loss = np.average(losses) # running average
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			f1 = f1_score(batch_train_words.cpu().detach().numpy(), img_tranf.cpu().detach().numpy().round(), average='macro')
			f1s.append(f1)
			avg_f1 = np.average(f1s) # running average

			if total_step % args.print_freq == 0:
				print("Epoch {}, Step {}, Loss {}".format(epoch, j, avg_loss))
				print("Epoch {}, Step {}, Loss {}, f1_score {}".format(i, j, avg_loss, avg_f1))

		print('############# Evaluation ##############')
		model.eval()
		train_f1s.append(avg_f1)
		train_losses.append(avg_loss)

		val_img_mat = pm.expmap0(val_img_mat, c=1)
		img_tranf = model(val_img_mat)
		loss      = criterion(img_tranf, val_words_embd)

		pre_recall_f1 = precision_recall_fscore_support(\
		    val_words_embd.cpu().detach().numpy(), img_tranf.cpu().detach().numpy().round(), average='macro')
		
		if(test_losses != [] and loss.item() < min(test_losses)): 
			torch.save(model, args.model_path)

		torch.cuda.empty_cache()

		test_losses.append(loss.item())
		counter.append(total_step)

		print("Loss {}".format(loss.item()) )

		print("Loss {}, pre {}, recall {}, f1_score {}".format(loss.item(), pre_recall_f1[0], pre_recall_f1[1], pre_recall_f1[2]))

		if pre_recall_f1[2] > best_f1: 
			best_f1        = pre_recall_f1[2]
			best_recall    = pre_recall_f1[1]
			best_precision = pre_recall_f1[0]

		print('#######################################')

		if epoch % args.save_freq == 0:
			print('==> Saving...')
			state = {
				'opt'      : args,
				'model'    : model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'epoch'    : epoch,
			}
			save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
			torch.save(state, save_file)
			# help release GPU memory
			del state

		if epoch % 20 == 0:
			plt.title('Cos_Loss')
			plt.xlabel('number of training examples seen')
			plt.ylabel('Cos_Loss')

			plt.plot(counter, train_losses,'b',label='train')
			plt.plot(counter, test_losses,'r', label='val')
			 
			plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
			plt.grid()
			plt.savefig(args.loss_path)

			plt.cla()
			plt.title('f1_score')
			plt.xlabel('number of training batches seenVal')
			plt.ylabel('f1_score')

			plt.plot(counter, train_f1s,'b',label='train')
			plt.plot(counter, test_f1s,'r', label='test')

			plt.legend(['Train f1_score', 'Test f1_score'], loc='upper right')
			plt.grid()
			plt.savefig('./pictures/f1.jpg')

	print("pre {}, recall {}, f1_score {}".format(best_precision, best_recall, best_f1))

	plt.title('Cos_Loss')
	plt.xlabel('number of training examples seen')
	plt.ylabel('Cos_Loss')

	plt.plot(counter, train_losses,'b',label='train')
	plt.plot(counter, test_losses,'r', label='val')
	 
	plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
	plt.grid()
	plt.savefig(args.loss_path)

	plt.cla()
	plt.title('f1_score')
	plt.xlabel('number of training batches seenVal')
	plt.ylabel('f1_score')

	plt.plot(counter, train_f1s,'b',label='train')
	plt.plot(counter, test_f1s,'r', label='test')
	 
	plt.legend(['Train f1_score', 'Test f1_score'], loc='upper right')
	plt.grid()
	plt.savefig('./pictures/f1.jpg')

if __name__ == '__main__':
	main()