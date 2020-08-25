from .BaseDataset import BaseDataset
import numpy as np
import os
import gzip
from PIL import Image

class FashionMNISTDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self,opt)
        self.img_dir = os.path.join(opt.dataroot,'train-images-idx3-ubyte.gz')
        self.label_dir = os.path.join(opt.dataroot,'train-labels-idx1-ubyte.gz')
        self.images, self.labels = self.read_mnist(self.img_dir,self.label_dir)
        self.label_size = len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        index = index % self.label_size
        img = self.images[index].reshape(28,28)
        label = self.labels[index]

        img = Image.fromarray(img, mode='L')
        img_transform = self.get_transform(self.get_params(),grayscale=True)
        A_img = img_transform(img)

        dic = {'image':A_img,'label':label,'index':index}
        return dic

    def get_params(self):
        if self.opt.netG == 'MLP_G':
            params = {}
        else:
            params = {'Scale': 32}
        return params

    def read_mnist(self,image_path,label_path):
        with gzip.open(label_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

        with gzip.open(image_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

        return images,labels