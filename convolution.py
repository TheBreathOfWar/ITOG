from video.utils_track import get_conv
import numpy as np
from torchvision import transforms # его используем
import cv2
import torch

def get_convol(path, counter):
    image= cv2.imread(path)
    if image is not None:
        counter+=1
        convert_tensor = transforms.ToTensor()
        convert_resize = transforms.Resize((64, 64))
        image = convert_resize(convert_tensor(image))
        image = image.view(64, 64, 3)
        conv = torch.nn.Conv2d(64, 1, kernel_size=5, bias=False, padding=2)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor([[0.,0.,1.,0.,0.],
                                            [0.,1.,1.,1.,0.],
                                            [1.,1.,1.,1.,1.],
                                            [0.,1.,1.,1.,0.],
                                            [0.,0.,1.,0.,0.],]))
        out = conv(image)
        out = out.view(192, 1)
        return out.detach().numpy().reshape(192,), counter
    else:
        return np.zeros(192), counter


human_conv=get_conv('crops/1/0.jpg', 5)
counter=0

for id in [1, 2, 3, 5, 6, 8, 11, 16, 17, 18, 20, 22, 23, 26, 27, 28, 31, 37, 38, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]:
    crops_path=f'crops/{id}/'

    for i in range(1000):
        path=crops_path+f'{i}.jpg'
        human, counter=get_convol(path, counter)
        human_conv+=human

    print(id, counter)


human_conv=human_conv/(counter+1)

with open('convolution.txt', "w") as file:
    file.write(' '.join([str(x) for x in human_conv]))


