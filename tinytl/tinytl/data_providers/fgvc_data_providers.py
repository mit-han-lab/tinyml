import os
import torchvision
from ofa.imagenet_classification.data_providers import ImagenetDataProvider

__all__ = [
	'FGVCDataProvider',
	'AircraftDataProvider', 'CarDataProvider', 'Flowers102DataProvider', 'CUB200DataProvider', 'PetsDataProvider',
	'Food101DataProvider', 'CIFAR10DataProvider', 'CIFAR100DataProvider',
]


class FGVCDataProvider(ImagenetDataProvider):

	@staticmethod
	def name():
		raise not NotImplementedError

	@property
	def n_classes(self):
		raise not NotImplementedError

	@property
	def save_path(self):
		raise not NotImplementedError


class AircraftDataProvider(FGVCDataProvider):

	@staticmethod
	def name():
		return 'aircraft'

	@property
	def n_classes(self):
		return 100

	@property
	def save_path(self):
		return os.path.expanduser('~/dataset/aircraft')


class CarDataProvider(FGVCDataProvider):

	@staticmethod
	def name():
		return 'car'

	@property
	def n_classes(self):
		return 196

	@property
	def save_path(self):
		return os.path.expanduser('~/dataset/stanford_car')


class Flowers102DataProvider(FGVCDataProvider):

	@staticmethod
	def name():
		return 'flowers102'

	@property
	def n_classes(self):
		return 102

	@property
	def save_path(self):
		return os.path.expanduser('~/dataset/flowers102')


class Food101DataProvider(FGVCDataProvider):

	@staticmethod
	def name():
		return 'food101'

	@property
	def n_classes(self):
		return 101

	@property
	def save_path(self):
		return os.path.expanduser('~/dataset/food101')


class CUB200DataProvider(FGVCDataProvider):

	@staticmethod
	def name():
		return 'cub200'

	@property
	def n_classes(self):
		return 200

	@property
	def save_path(self):
		return os.path.expanduser('~/dataset/cub200')


class PetsDataProvider(FGVCDataProvider):

	@staticmethod
	def name():
		return 'pets'

	@property
	def n_classes(self):
		return 37

	@property
	def save_path(self):
		return os.path.expanduser('~/dataset/pets')


class CIFAR10DataProvider(FGVCDataProvider):

	@staticmethod
	def name():
		return 'cifar10'

	@property
	def n_classes(self):
		return 10

	@property
	def save_path(self):
		return os.path.expanduser('~/dataset/cifar10')

	def train_dataset(self, _transforms):
		dataset = torchvision.datasets.CIFAR10(self.save_path, train=True, transform=_transforms, download=True)
		return dataset

	def test_dataset(self, _transforms):
		dataset = torchvision.datasets.CIFAR10(self.save_path, train=False, transform=_transforms, download=True)
		return dataset


class CIFAR100DataProvider(CIFAR10DataProvider):

	@staticmethod
	def name():
		return 'cifar100'

	@property
	def n_classes(self):
		return 100

	@property
	def save_path(self):
		return os.path.expanduser('~/dataset/cifar100')

	def train_dataset(self, _transforms):
		dataset = torchvision.datasets.CIFAR100(self.save_path, train=True, transform=_transforms, download=True)
		return dataset

	def test_dataset(self, _transforms):
		dataset = torchvision.datasets.CIFAR100(self.save_path, train=False, transform=_transforms, download=True)
		return dataset
