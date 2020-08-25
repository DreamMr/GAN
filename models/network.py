import torch
import numpy as np


def define_G(netG,dim,output_nc,ndf=512,n_layers = 3,norm_layer = torch.nn.BatchNorm2d):
    model = None
    if netG == 'MLP_G':
        model = MLP_G(dim,output_nc,ndf,n_layers)
    elif netG == 'Conv_G':
        model = Conv_G(dim,output_nc,ndf,n_layers,norm_layer)
    else:
        print('Not Found Model G !')
    return model

def define_D(netD,input_nc,ndf=512,n_layers = 3,norm_layer = torch.nn.BatchNorm2d):
    if netD == 'MLP_D':
        model = MLP_D(input_nc,ndf,n_layers)
    elif netD == 'Conv_D':
        model = Conv_D(input_nc,ndf,n_layers)
    else:
        print('Not Found Model D !')
    return model

class MLP_G(torch.nn.Module):
    def __init__(self,dim,output_nc=28*28,ndf=512,n_layers = 3):
        '''

        :param dim:
        :param output_nc:
        :param ndf:
        :param n_layers:
        '''

        super(MLP_G,self).__init__()
        self.ndf = ndf
        cur_ndf = ndf

        module_list = [torch.nn.Linear(dim,cur_ndf),
                            torch.nn.LeakyReLU(0.2)
                       ]
        for i in range(n_layers):
            module_list += [torch.nn.Linear(cur_ndf,cur_ndf),
                            torch.nn.LeakyReLU(0.2)
                            ]
        module_list += [torch.nn.Linear(cur_ndf,output_nc),
                        torch.nn.Tanh()
                        ]

        self.model = torch.nn.Sequential(*module_list)

    def forward(self, input):
        input = input.view(input.size(0),-1)
        output = self.model(input)
        return output

class MLP_D(torch.nn.Module):
    def __init__(self,input_nc,ndf=512,n_layers = 3):
        '''

        :param dim:
        :param input_nc:
        :param ndf:
        :param n_layers:
        '''
        super(MLP_D, self).__init__()
        self.ndf = ndf
        cur_ndf = ndf
        module_list = [torch.nn.Linear(input_nc,cur_ndf),
                       torch.nn.LeakyReLU(0.2)
                       ]

        for i in range(n_layers):
            module_list += [torch.nn.Linear(cur_ndf,cur_ndf),
                           torch.nn.LeakyReLU(0.2)
                           ]
        module_list += [torch.nn.Linear(cur_ndf,1),
                        torch.nn.Sigmoid()
                        ]
        self.model = torch.nn.Sequential(*module_list)

    def forward(self, input):
        input = input.view(input.size(0),-1)
        output = self.model(input)
        return output



class Conv_G(torch.nn.Module):
    def __init__(self,dim,output_nc=3,ndf=1024,n_layers = 2,norm_layer = torch.nn.BatchNorm2d):
        '''
        DCGAN G network
        :param dim:
        :param output_nc:
        :param ndf:
        :param n_layers:
        :param norm_layer:
        '''
        super(Conv_G, self).__init__()
        self.ndf = ndf
        cur_ndf = ndf

        model = [torch.nn.ConvTranspose2d(dim,cur_ndf,kernel_size=4,stride=1,padding=0),
                      torch.nn.BatchNorm2d(cur_ndf),
                      torch.nn.ReLU(True)
                 ]
        for i in range(n_layers):
            model += [torch.nn.ConvTranspose2d(cur_ndf,cur_ndf >> 1,kernel_size=4,stride=2,padding=1),
                      torch.nn.BatchNorm2d(cur_ndf >> 1),
                      torch.nn.ReLU(True)
                      ]
            cur_ndf = cur_ndf >> 1
        model += [torch.nn.ConvTranspose2d(cur_ndf,output_nc,kernel_size=4,stride=2,padding=1),
                  torch.nn.Tanh()]
        self.model = torch.nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        return output


class Conv_D(torch.nn.Module):
    def __init__(self,input_nc = 3,ndf = 256,n_layers = 2):
        '''

        :param input_nc:
        :param ndf:
        '''
        super(Conv_D, self).__init__()
        model = []
        self.ndf = ndf
        self.input_nc = input_nc
        cur_ndf = ndf

        model += [torch.nn.Conv2d(input_nc,cur_ndf,kernel_size=4,stride=2,padding=1),
                  torch.nn.LeakyReLU(0.2,True)]

        for i in range(n_layers):
            model += [torch.nn.Conv2d(cur_ndf,cur_ndf << 1,kernel_size=4,stride=2,padding=1),
                      torch.nn.BatchNorm2d(cur_ndf << 1),
                      torch.nn.LeakyReLU(0.2,True)]
            cur_ndf = cur_ndf << 1

        model += [torch.nn.Conv2d(cur_ndf,1,kernel_size=4,stride=1,padding=0),
                  torch.nn.Sigmoid()]
        self.model = torch.nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        return output


# if __name__ == '__main__':
#     # netD = MLP_G(input_nc=28*28)
#     # input = torch.randn(1,1,28,32)
#     # out = netD(input)
#     # print(out.size())
#
#     netG = MLP_G(dim = 100,output_nc=28*28)
#     input = torch.randn(100,100,1,1)
#     out_G = netG(input)
#     print(out_G.size())





