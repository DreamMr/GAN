# TODO 保存图片、 模型、集成FID

import torch
import numpy as np
from PIL import Image
import os

CHECKPOINT_PATH = './checkpoints/'

def save_model(net,name):
    path = os.path.join(CHECKPOINT_PATH,name)
    model_name = path
    torch.save(net.state_dict(),model_name)


def load_model(model,name):
    model_name = os.path.join(CHECKPOINT_PATH,name)
    model.load_state_dict(torch.load(model_name))
    return model


def tensor2img(tensor, imtype = np.uint8):
    tmp = tensor[0].data.cpu().numpy()
    image_numpy = (np.transpose(tmp,(1,2,0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def save_img(tensor,path,sz):
    tmp = tensor2img(tensor).reshape(28,28)
    img = Image.fromarray(tmp)
    img.save(path,dpi=(300.0,300.0))




