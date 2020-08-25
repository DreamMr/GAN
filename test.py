# TODO 测试代码的编写
from models import GAN,DCGAN
from options import TestOptions
from util import util
import os
import torch

TESTOUT_PATH = './test_out'

if __name__ == '__main__':
    epochs = [90,95]
    opt = TestOptions.TestOptions().gather_options()
    root_path = os.path.join(TESTOUT_PATH,opt.name)
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    for epoch in epochs:
        model = GAN.GAN(opt)
        model.load_models(epoch)
        cur_path = os.path.join(root_path,str(epoch))
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)

        for i in range(opt.out_number):
            z = torch.randn(1,opt.nz,1,1)
            real_B = torch.randn(1, 1, 28, 28)
            model.set_input({'z': z, 'real_B': real_B})
            model.test()
            visuals = model.get_current_visuals()
            fake = visuals['fake_B']
            file_name = os.path.join(cur_path,str(i)+'.png')
            util.save_img(fake,file_name,(28,28))






