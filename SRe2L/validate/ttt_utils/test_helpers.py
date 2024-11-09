import numpy as np
import torch
import torch.nn as nn
import torchvision
from ttt_utils.misc import *
from ttt_utils.rotation import rotate_batch

def build_model(args):
	# from ttt_models.ResNet import ResNetCifar as ResNet
	from ttt_models.SSHead import ExtractorHead
	print('Building model...')
	print('Training on ImageNet-1K')
	
	classes = 1000

	if args.group_norm == 0:
		norm_layer = nn.BatchNorm2d
	else:
		def gn_helper(planes):
			return nn.GroupNorm(args.group_norm, planes)
		norm_layer = gn_helper

	# net = ResNet(args.depth, args.width, channels=3, classes=classes, norm_layer=norm_layer).to(args.device)
	net = torchvision.models.__dict__[args.model](pretrained=False)
	if args.shared == 'none':
		args.shared = None

	if args.shared == 'layer4' or args.shared is None:
		from ttt_models.SSHead import extractor_from_layer4
		ext = extractor_from_layer4(net)
		head = nn.Linear(512 * args.width, 4)
	elif args.shared == 'layer3':
		from ttt_models.SSHead import extractor_from_layer3, head_on_layer3
		ext = extractor_from_layer3(net)
		head = head_on_layer3(net, args.width, 4)
	elif args.shared == 'layer2':
		from ttt_models.SSHead import extractor_from_layer2, head_on_layer2
		ext = extractor_from_layer2(net)
		head = head_on_layer2(net, args.width, 4)
	ssh = ExtractorHead(ext, head).to(args.device)

	if hasattr(args, 'parallel') and args.parallel:
		net = torch.nn.DataParallel(net)
		ssh = torch.nn.DataParallel(ssh)
	return net, ext, head, ssh
	

def test(dataloader, model, sslabel=None):
	criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
	model.eval()
	correct = []
	losses = []
	for batch_idx, (inputs, labels) in enumerate(dataloader):
		if sslabel is not None:
			inputs, labels = rotate_batch(inputs, sslabel)
		inputs, labels = inputs.to(args.device), labels.to(args.device)
		with torch.no_grad():
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			losses.append(loss.cpu())
			_, predicted = outputs.max(1)
			correct.append(predicted.eq(labels).cpu())
	correct = torch.cat(correct).numpy()
	losses = torch.cat(losses).numpy()
	model.train()
	return 1-correct.mean(), correct, losses

def test_grad_corr(dataloader, net, ssh, ext):
	criterion = nn.CrossEntropyLoss().to(args.device)
	net.eval()
	ssh.eval()
	corr = []
	for batch_idx, (inputs, labels) in enumerate(dataloader):
		net.zero_grad()
		ssh.zero_grad()
		inputs_cls, labels_cls = inputs.to(args.device), labels.to(args.device)
		outputs_cls = net(inputs_cls)
		loss_cls = criterion(outputs_cls, labels_cls)
		grad_cls = torch.autograd.grad(loss_cls, ext.parameters())
		grad_cls = flat_grad(grad_cls)

		ext.zero_grad()
		inputs, labels = rotate_batch(inputs, 'expand')
		inputs_ssh, labels_ssh = inputs.to(args.device), labels.to(args.device)
		outputs_ssh = ssh(inputs_ssh)
		loss_ssh = criterion(outputs_ssh, labels_ssh)
		grad_ssh = torch.autograd.grad(loss_ssh, ext.parameters())
		grad_ssh = flat_grad(grad_ssh)

		corr.append(torch.dot(grad_cls, grad_ssh).item())
	net.train()
	ssh.train()
	return corr


def pair_buckets(o1, o2):
	crr = np.logical_and( o1, o2 )
	crw = np.logical_and( o1, np.logical_not(o2) )
	cwr = np.logical_and( np.logical_not(o1), o2 )
	cww = np.logical_and( np.logical_not(o1), np.logical_not(o2) )
	return crr, crw, cwr, cww
def count_each(tuple):
	return [item.sum() for item in tuple]


def plot_epochs(all_err_cls, all_err_ssh, fname, use_agg=True):
	import matplotlib.pyplot as plt
	if use_agg:
		plt.switch_backend('agg')

	plt.plot(np.asarray(all_err_cls)*100, color='r', label='classifier')
	plt.plot(np.asarray(all_err_ssh)*100, color='b', label='self-supervised')
	plt.xlabel('epoch')
	plt.ylabel('test error (%)')
	plt.legend()
	plt.savefig(fname)
	plt.close()
