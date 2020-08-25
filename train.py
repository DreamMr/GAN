import torch
from tensorboardX import SummaryWriter
import time
from data import FashionMNIST_dataset
from torch.utils.data import Dataset,DataLoader
from models import DCGAN,GAN
from options import BaseOptions,TrainOptions
import json
from util import util
import os
from evaluation import fid_score

TRAINOUTPUT = './train_out'

def main():
    opt = TrainOptions.TrainOptions().gather_options()
    fashion_dataset = FashionMNIST_dataset.FashionMNISTDataset(opt)
    train_data = DataLoader(fashion_dataset,opt.batch_size,shuffle=True,num_workers=4)

    model = GAN.GAN(opt)
    print(model.netD,model.netG)
    writer = SummaryWriter(comment=opt.name)
    print(len(fashion_dataset))
    total_iters = 0
    for epoch in range(opt.niter):
        start_time = time.time()
        iter = 0
        for i,data in enumerate(train_data):
            z = torch.randn(opt.batch_size,opt.nz,1,1)
            input = {'real_B':data['image'],'z':z}
            model.set_input(input)
            model.optimize_parameters()
            # TODO 测试代码
            losses = model.get_current_losses()

            if total_iters % opt.print_loss == 0:
                loss_json = json.dumps(losses)
                print('epoch: ', epoch,' iter:',iter,' loss:',loss_json)

            if total_iters % opt.save_loss == 0:
                loss_key = list(losses.keys())
                for key in loss_key:
                    writer.add_scalar(key,losses[key],total_iters)

            if total_iters % opt.save_img == 0:
                visuals = model.get_current_visuals()
                visuals_key = list(visuals.keys())
                for key in visuals_key:
                    writer.add_image(key,util.tensor2img(visuals[key]),total_iters,dataformats='HWC')

            total_iters += opt.batch_size
            iter += opt.batch_size


        print('epoch: ',epoch,' end !',' cost time: ',time.time() - start_time)
        if epoch % opt.save_model == 0:
            model.save_models(epoch)

        # if total_iters % opt.eva == 0:
        #     num = len(fashion_dataset)
        #     dir_name = os.path.join(TRAINOUTPUT,opt.name,str(epoch))
        #     if not os.path.isdir(dir_name):
        #         os.mkdir(dir_name)
        #     for i in range(num):
        #         path = os.path.join(dir_name,str(epoch)+'_'+str(i)+'.png')
        #         z = torch.randn(1,opt.nz,1,1)
        #         real_B = torch.randn(1,1,32,32)
        #         model.set_input({'z':z,'real_B':real_B})
        #         model.test()
        #         visuals = model.get_current_visuals()
        #         fake = visuals['fake_B']
        #         util.save_img(fake,path)
            # paths = (os.path.join(opt.dataroot,'train'),os.path.join(TRAINOUTPUT,opt.name))
            # fid_value = fid_score.main('0',paths,batch_size=100,dims=192)
            # print(fid_value)

    writer.close()

if __name__ == '__main__':
    main()