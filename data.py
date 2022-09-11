from PIL import img
import numpy as np
import torch

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_img(path, img_transform, size = (300, 300)):
    img = img.open(path)
    img = img.resize(size, img.LANCZOS)

    if img.mode == 'RGBA':
        tmp = img.new("RGB", img.size, (255, 255, 255))
        tmp.paste(img, mask = img.split()[3])
        img = tmp

    img = img_transform(img).unsqueeze(0)
    return img.to(device)

def get_gram(m):
    _, c, h, w = m.size()
    m =  m.view(c, h*w)
    m = torch.mm(m, m.t())
    return m

def denormalize_img(inp):
    inp = inp.numpy().transpose((1,2,0))
    inp = inp * std + mean
    inp = np.clip(inp, 0, 1)
    return inp
