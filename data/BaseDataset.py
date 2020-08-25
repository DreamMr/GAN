import torch.utils.data as data
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):
    def __init__(self,opt):
        self.opt = opt
        self.root = opt.dataroot

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass

    def get_transform(self,params,grayscale= False,convert=True):
        transform_list = []
        if params.__contains__('Scale'):
            transform_list += [transforms.Scale(params['Scale'])]
        if convert:
            transform_list += [transforms.ToTensor()]
            if grayscale:
                transform_list += [ transforms.Normalize((0.5,),(0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)
