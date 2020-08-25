from .network import *
import numpy as np
import torch
from .BaseModel import BaseModel

class GAN(BaseModel):
    def __init__(self,opt):
        BaseModel.__init__(self,opt)
        self.loss_names = ['G_GAN','D_real','D_fake','D_total']
        self.visual_names = ['real_B','fake_B']
        self.model_names = ['G','D']
        self.nz = self.opt.nz

        self.netG = define_G(self.opt.netG,self.opt.nz,28*28,512,4).to(self.device)
        self.netD = define_D(self.opt.netD,28*28,512,3).to(self.device)

        if self.opt.is_train:
            self.d_optimizer = torch.optim.Adam(self.netD.parameters(),lr = self.opt.lr,betas=(0.5, 0.999))
            self.g_optimizer = torch.optim.Adam(self.netG.parameters(),lr = self.opt.lr,betas=(0.5, 0.999))

            self.criterionGAN = torch.nn.BCELoss().to(self.device)

    def set_input(self, input):
        self.z = input['z'].to(self.device)
        self.real_B = input['real_B'].to(self.device)

    def forward(self):
        self.fake_B = self.netG(self.z).view(self.opt.batch_size,1,28,28)
        self.real_labels = torch.ones(self.opt.batch_size).to(self.device)
        self.fake_labels = torch.zeros(self.opt.batch_size).to(self.device)


    def backward_D(self):
        # real
        real_pred = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(real_pred,self.real_labels)

        # fake
        fake_pred = self.netD(self.fake_B.detach())
        self.loss_D_fake = self.criterionGAN(fake_pred,self.fake_labels)

        self.loss_D_total = (self.loss_D_real + self.loss_D_fake) * 0.5
        self.loss_D_total.backward()

    def backward_G(self):
        fake_pred = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(fake_pred,self.real_labels)
        self.loss_G_GAN.backward()

    def optimize_parameters(self):
        self.forward()

        # Optimize discriminator
        self.set_requires_grad(self.netD,True)
        self.d_optimizer.zero_grad()
        self.backward_D()
        self.d_optimizer.step()

        # Optimize
        self.set_requires_grad(self.netD,False)
        self.g_optimizer.zero_grad()
        self.backward_G()
        self.g_optimizer.step()




