from ofa.imagenet_classification.run_manager import ImagenetRunConfig

from .fgvc_data_providers import AircraftDataProvider, Flowers102DataProvider, CarDataProvider
from .fgvc_data_providers import Food101DataProvider, CUB200DataProvider, PetsDataProvider
from .fgvc_data_providers import CIFAR10DataProvider, CIFAR100DataProvider

__all__ = ['FGVCRunConfig']


class FGVCRunConfig(ImagenetRunConfig):

    def __init__(self, n_epochs=50, init_lr=0.01, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='flowers102', train_batch_size=256, test_batch_size=500, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0, no_decay_keys=None,
                 mixup_alpha=None, model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color='tf', image_size=224, fast_evaluation=True, **kwargs):
        super(FGVCRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys, mixup_alpha,
            model_init, validation_frequency, print_frequency,
            n_worker, resize_scale, distort_color, image_size, **kwargs,
        )
        self.fast_evaluation = fast_evaluation

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == AircraftDataProvider.name():
                DataProviderClass = AircraftDataProvider
            elif self.dataset == Flowers102DataProvider.name():
                DataProviderClass = Flowers102DataProvider
            elif self.dataset == CarDataProvider.name():
                DataProviderClass = CarDataProvider
            elif self.dataset == Food101DataProvider.name():
                DataProviderClass = Food101DataProvider
            elif self.dataset == CUB200DataProvider.name():
                DataProviderClass = CUB200DataProvider
            elif self.dataset == PetsDataProvider.name():
                DataProviderClass = PetsDataProvider
            elif self.dataset == CIFAR10DataProvider.name():
                DataProviderClass = CIFAR10DataProvider
            elif self.dataset == CIFAR100DataProvider.name():
                DataProviderClass = CIFAR100DataProvider
            else:
                raise ValueError('Do not support: %s' % self.dataset)
            self.__dict__['_data_provider'] = DataProviderClass(
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
            )
        return self.__dict__['_data_provider']

    @property
    def valid_loader(self):
        if not self.fast_evaluation:
            return self.data_provider.valid

        if self.valid_size is None:
            return self.test_loader
        if self.__dict__.get('_in_memory_valid%d' % self.data_provider.active_img_size, None) is None:
            self.__dict__['_in_memory_valid%d' % self.data_provider.active_img_size] = []
            for images, labels in self.data_provider.valid:
                self.__dict__['_in_memory_valid%d' % self.data_provider.active_img_size].append((images, labels))
        return self.__dict__['_in_memory_valid%d' % self.data_provider.active_img_size]

    @property
    def test_loader(self):
        if not self.fast_evaluation:
            return self.data_provider.test

        if self.__dict__.get('_in_memory_test%d' % self.data_provider.active_img_size, None) is None:
            self.__dict__['_in_memory_test%d' % self.data_provider.active_img_size] = []
            for images, labels in self.data_provider.test:
                self.__dict__['_in_memory_test%d' % self.data_provider.active_img_size].append((images, labels))
        return self.__dict__['_in_memory_test%d' % self.data_provider.active_img_size]

