from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import argparse
from tqdm import tqdm


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
		pass

	def __getitem__(self, index):
		"""
		Return Numpy array of image, steer and throttle
		"""

		data = dict()

		data['img'] = self.imgs[index]
		data['steer'] = self.steer[index]
		data['throttle'] = self.throttle[index]

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

	model.to(args.device)

	return model


def train_epoch(args, model, dataloader):
	model.train()

	# Losses per epoch
	lossEpoch = 0.
	lossEpochSteer = 0.
	lossEpochThottle = 0.
	
	for batch_num, data in enumerate(tqdm(dataloader), 0):
		# efficiently zero gradients
		for p in model.parameters():
			p.grad = None

		img = torch.from_numpy(data['img']).to(args.device, dtype=torch.float32))
		steer = torch.from_numpy(data['steer']).to(args.device, dtype=torch.float32))
		throttle = torch.from_numpy(data['throttle']).to(args.device, dtype=torch.float32))

		commands = model.forward(img)

		# Losses per batch
		loss = 0.
		lossSteer = 0.
		lossThrottle = 0.

		loss += F.l1_loss(commands[0], steer).mean()
		lossSteer += F.l1_loss(commands[0], steer).mean()

		loss += F.l1_loss(commands[1], throttle).mean()
		lossThrottle += F.l1_loss(commands[1], throttle).mean()

		loss.backward()
		optimizer.step()

		lossEpoch += float(loss.item())
		lossEpochSteer += float(lossSteer.item())
		lossEpochThottle += float(lossThrottle.item())
		
	lossEpoch = lossEpoch / batch_num
	lossEpochSteer = lossEpochSteer / batch_num
	lossEpochThottle = lossEpochThottle / batch_num

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

	model = get_model(args)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	trainDataset = L2RDataset()

	dataloaderTrain = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

	for epoch in range(args.epochs):
		lossEpoch, lossEpochSteer, lossEpochThottle = train_epoch(args, model, dataloaderTrain)

		# validate_epoch()

		print("Loss for epoch {} is {:.2f}".format(epoch, lossEpoch))

if __name__ == '__main__':
	main()