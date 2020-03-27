import os
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
import argparse

from models.network import Image_Transformer, MyHingeLoss

def parse_option(): 

	parser = argparse.ArgumentParser('argument for training')

	parser.add_argument('--model_folder', type=str, default='./results/model/devise')
	parser.add_argument('--loss_path', type=str, default='./results/loss/loss_test.jpg')
	parser.add_argument('--train_img_mat_path', type=str, default='../data/CV/Imagenet/all/train/train_image_mat_resnet50.npy')
	parser.add_argument('--val_img_mat_path', type=str, default='../data/CV/Imagenet/all/val/val_image_mat_resnet50.npy')
	parser.add_argument('--word_model', type=str, default='glove')
	
	parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
	parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
	parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
	parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
	parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
	parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
	parser.add_argument('--mode', type=str, default='normal')
	parser.add_argument('--dimension', type=int, default='dimension')
	opt = parser.parse_args()

	# data
	if opt.word_model == 'glove':
		opt.train_words_embd_path = '../data/CV/Imagenet/all/train/Eu_version/train_words_embd_glove_300.npy'
		opt.val_words_embd_path   = '../data/CV/Imagenet/all/val/Eu_version/val_words_embd_glove_300.npy'
	elif opt.word_model == 'hier':
		opt.train_words_embd_path = '../data/CV/Imagenet/all/train/Eu_version/train_words_embd_hier_1000.npy'
		opt.val_words_embd_path   = '../data/CV/Imagenet/all/val/Eu_version/val_words_embd_hier_1000.npy'
	elif opt.word_model == 'gh':
		opt.train_words_embd_path = '../data/CV/Imagenet/all/train/Eu_version/train_words_embd_gh_300+1000.npy'
		opt.val_words_embd_path   = '../data/CV/Imagenet/all/val/Eu_version/val_words_embd_gh_300+1000.npy'	
	elif opt.word_model == 'ehier':
		opt.train_words_embd_path = '../data/CV/Imagenet/all/train/Eu_version/train_words_embd_ehier_300.npy'
		opt.val_words_embd_path   = '../data/CV/Imagenet/all/val/Eu_version/val_words_embd_ehier_300.npy'
	elif opt.word_model == 'geh':
		opt.train_words_embd_path = '../data/CV/Imagenet/all/train/Eu_version/train_words_embd_geh_300+300.npy'
		opt.val_words_embd_path   = '../data/CV/Imagenet/all/val/Eu_version/val_words_embd_geh_300+300.npy'
	else: 
		opt.train_words_embd_path = '../data/CV/Imagenet/all/train/Hype_version/train_words_embd_poincare_300.npy'
		opt.val_words_embd_path   = '../data/CV/Imagenet/all/val/Hype_version/val_words_embd_poincare_300.npy'

	if not os.path.isdir(opt.model_folder): 
		os.makedirs(opt.model_folder)
	
	if opt.mode == 'tiny_test':
	   opt.train_img_mat_path    = '../data/CV/Imagenet/tiny-imagenet-200/train_image_mat_resnet.npy'
	   opt.train_words_embd_path = '../data/CV/Imagenet/tiny-imagenet-200/train_words_embd.npy'
	   opt.val_img_mat_path      = '../data/CV/Imagenet/tiny-imagenet-200/val/val_image_mat_resnet.npy'
	   opt.val_words_embd_path   = '../data/CV/Imagenet/tiny-imagenet-200/val/val_words_embd.npy'

	print('model:%s mode:%s folder:%s loss:%s dimension:%s' %(opt.word_model, opt.mode, opt.model_folder, opt.loss_path, str(opt.dimension)))
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
	#                X,              y = shuffle(X, y, random_state=233)
	return train_words_embd, train_img_mat, val_words_embd, val_img_mat

def set_model(args): 
	model     = Image_Transformer(args.dimension)
	criterion = MyHingeLoss(0.4, args.dimension)
	return model, criterion

def set_optimizer(args,model): 
	optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
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

		for j in range(0, len(train_img_mat), args.batch_size): 
			total_step        += 1
			batch_train_img    = train_img_mat[j: j+args.batch_size, :]
			batch_train_words  = train_words_embd[j: j+args.batch_size, :]
			batch_train_img    = torch.from_numpy(batch_train_img).type(torch.FloatTensor).cuda()
			batch_train_words  = torch.from_numpy(batch_train_words).type(torch.FloatTensor).cuda()

			img_tranf = model( batch_train_img )
			# y_pred    = y_pred.squeeze(1)

			loss = criterion.forward_train(img_tranf, batch_train_words)
			losses.append(loss.item())
			avg_loss = np.average(losses) # running average
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# TODO code for f1 score is temporarily commented
			# f1 = f1_score(batch_train_words.cpu().detach().numpy(), img_tranf.cpu().detach().numpy().round(), average='macro')
			# f1s.append(f1)
			# avg_f1 = np.average(f1s) # running average

			if total_step % args.print_freq == 0:
				print("Epoch {}, Step {}, Loss {}".format(epoch, j, avg_loss))
				# print("Epoch {}, Step {}, Loss {}, f1_score {}".format(i, j, avg_loss, avg_f1))

		print('############# Evaluation ##############')
		model.eval()
		# train_f1s.append(avg_f1)
		train_losses.append(avg_loss)

		img_tranf = model(val_img_mat)
		loss      = criterion.forward_val(img_tranf, val_words_embd)

		# pre_recall_f1 = precision_recall_fscore_support(\
		#     val_words_embd.cpu().detach().numpy(), img_tranf.cpu().detach().numpy().round(), average='macro')
		
		# if(test_losses != [] and loss.item() < min(test_losses)): 
		# 	torch.save(model, args.model_path)

		torch.cuda.empty_cache()

		test_losses.append(loss.item())
		counter.append(total_step)

		print("Loss {}".format(loss.item()) )

		# print("Loss {}, pre {}, recall {}, f1_score {}".format(loss.item(), pre_recall_f1[0], pre_recall_f1[1], pre_recall_f1[2]))

		# if pre_recall_f1[2] > best_f1: 
		# best_f1        = pre_recall_f1[2]
		# best_recall    = pre_recall_f1[1]
		# best_precision = pre_recall_f1[0]

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

			# plt.cla()
			# plt.title('f1_score')
			# plt.xlabel('number of training batches seenVal')
			# plt.ylabel('f1_score')

			# plt.plot(counter, train_f1s,'b',label='train')
			# plt.plot(counter, test_f1s,'r', label='test')

			# plt.legend(['Train f1_score', 'Test f1_score'], loc='upper right')
			# plt.grid()
			# plt.savefig('./pictures/f1.jpg')

	# print("pre {}, recall {}, f1_score {}".format(best_precision, best_recall, best_f1))

	plt.title('Cos_Loss')
	plt.xlabel('number of training examples seen')
	plt.ylabel('Cos_Loss')

	plt.plot(counter, train_losses,'b',label='train')
	plt.plot(counter, test_losses,'r', label='val')
	 
	plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
	plt.grid()
	plt.savefig(args.loss_path)

	# plt.cla()
	# plt.title('f1_score')
	# plt.xlabel('number of training batches seenVal')
	# plt.ylabel('f1_score')

	# plt.plot(counter, train_f1s,'b',label='train')
	# plt.plot(counter, test_f1s,'r', label='test')
	 
	# plt.legend(['Train f1_score', 'Test f1_score'], loc='upper right')
	# plt.grid()
	# plt.savefig('./pictures/f1.jpg')

if __name__ == '__main__':
	main()