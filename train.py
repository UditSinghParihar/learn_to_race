from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch import nn
from torch import optim
import torch
import argparse
from tqdm import tqdm
import numpy as np


class L2RDataset(Dataset):
	def __init__(self):
		"""
		One to one correspondence between 
		self.imgs, self.steer and self.throttle
		"""
		
		self.imgs = []		
		self.steer = []
		self.throttle = []

		self.load_dataset()

	def __len__(self):
		return len(self.imgs)

	def load_dataset(self):
		"""
		Fill self.imgs, self.steer and self.throttle
		"""
		dummyImage = np.zeros((3, 224, 224))
		dummySteer = np.array([0.2])
		dummyThrottle = np.array([0.8])

		self.imgs.append(dummyImage)
		self.steer.append(dummySteer)
		self.throttle.append(dummyThrottle)
		
		pass

	def __getitem__(self, index):
		"""
		Return Numpy array of image, steer and throttle
		"""

		data = dict()

		data['img'] = torch.from_numpy(self.imgs[index])
		data['steer'] = torch.from_numpy(self.steer[index])
		data['throttle'] = torch.from_numpy(self.throttle[index])

		return data


def get_model(args):
	model = models.resnet18(pretrained=True)

	model.fc = nn.Sequential(
		nn.Linear(512, 256), 
		nn.ReLU(),
		nn.Dropout(0.2),
		nn.Linear(256, args.num_actions),
		nn.Tanh()
	)

	return model


class RegressionNetwork(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.features = models.resnet18(pretrained=True)

		self.regressor = nn.Sequential(
			nn.Linear(512, 256), 
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(256, args.num_actions),
			nn.Tanh()
		)
	
	def normalize_imagenet(self, x):
		""" Normalize input images according to ImageNet standards.
		Args:
			x (tensor): input images
		"""
		x = x.clone()
		x[:, 0] = (x[:, 0] - 0.485) / 0.229
		x[:, 1] = (x[:, 1] - 0.456) / 0.224
		x[:, 2] = (x[:, 2] - 0.406) / 0.225
		
		return x


	def forward(self, x):
		x = self.normalize_imagenet(x)
		
		print("X:", x.shape)
		# Resnet Layers
		x = self.features.conv1(x)
		x = self.features.bn1(x)
		x = self.features.relu(x)
		x = self.features.maxpool(x)

		x = self.features.layer1(x)
		x = self.features.layer2(x)
		x = self.features.layer3(x)
		x = self.features.layer4(x)

		x = self.features.avgpool(x)
		
		print("After avgpool: ", x.shape)
		x = self.regressor(x)

		return x


def train_epoch(args, model, dataloader, optimizer):
	model.train()

	# Losses per epoch
	lossEpoch = 0.
	lossEpochSteer = 0.
	lossEpochThottle = 0.
	
	for batch_num, data in enumerate(tqdm(dataloader), 0):
		# efficiently zero gradients
		for p in model.parameters():
			p.grad = None

		img = data['img'].to(args.device, dtype=torch.float32)
		steer = data['steer'].to(args.device, dtype=torch.float32)
		throttle = data['throttle'].to(args.device, dtype=torch.float32)

		# print("Image:", img.shape)

		commands = model.forward(img)

		# print("Commands", commands.shape, commands[0, 0], commands[0, 1])
		# exit(1)

		# Losses per batch
		loss = 0.
		lossSteer = 0.
		lossThrottle = 0.

		loss += F.l1_loss(commands[0, 0], steer).mean()
		lossSteer += F.l1_loss(commands[0, 0], steer).mean()

		loss += F.l1_loss(commands[0, 1], throttle).mean()
		lossThrottle += F.l1_loss(commands[0, 1], throttle).mean()

		loss.backward()
		optimizer.step()

		lossEpoch += float(loss.item())
		lossEpochSteer += float(lossSteer.item())
		lossEpochThottle += float(lossThrottle.item())
		
	lossEpoch = lossEpoch / (batch_num+1)
	lossEpochSteer = lossEpochSteer / (batch_num+1)
	lossEpochThottle = lossEpochThottle / (batch_num+1)

	return lossEpoch, lossEpochSteer, lossEpochThottle


def validate_epoch(args, model, dataloader):
	pass


def evaluate(img, model):
	pass


def main():
	parser = argparse.ArgumentParser(description="Infer the tensorrt engine file.")
	parser.add_argument('--device', type=str, default='cuda', help='Device to use for gpu')
	parser.add_argument('--num_actions', type=int, default=2, help='Number of actions output')
	parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
	parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('--workers', type=int, default=8, help='Dataloading workers per GPU')

	args = parser.parse_args()

	model = get_model(args).to(args.device)
	# model = RegressionNetwork(args).to(args.device)
	# print(model)
	# exit(1)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	trainDataset = L2RDataset()

	dataloaderTrain = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

	for epoch in range(args.epochs):
		lossEpoch, lossEpochSteer, lossEpochThottle = train_epoch(args, model, dataloaderTrain, optimizer)

		# validate_epoch()

		print("Loss for epoch {} is {:.2f}".format(epoch, lossEpoch))

if __name__ == '__main__':
	main()