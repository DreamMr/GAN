import torch
import numpy as np
from collections import OrderedDict
from util import util
import os

class BaseModel:
    def __init__(self,opt):
        self.opt = opt
        self.model_names = []
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu')

        model_path = os.path.join(util.CHECKPOINT_PATH,self.opt.name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        file_name = os.path.join(model_path,'train_opt.txt')
        with open(file_name,'w') as file:
            file.write(str(self.opt))

    def set_input(self,input):
        self.input = input

    def forward(self):
        pass

    def is_train(self):
        self.train = True
        return True

    def test(self):
        with torch.no_grad():
            self.forward()

    def optimize_parameters(self):
        pass

    def set_requires_grad(self, net, requires_grad=False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad  # to avoid computation

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_models(self,epoch):

        for name in self.model_names:
            if isinstance(name,str):
                save_filename = '%s_net_%s.pth' %(epoch,name)
                net = getattr(self,'net' + name)
                save_filename = os.path.join(self.opt.name,save_filename)
                util.save_model(net,save_filename)


    def load_models(self,epoch):
        for name in self.model_names:
            if isinstance(name,str):
                load_filename = '%s_net_%s.pth' % (epoch,name)
                net = getattr(self,'net' + name)
                load_filename = os.path.join(self.opt.name,load_filename)
                net = util.load_model(net,load_filename)
