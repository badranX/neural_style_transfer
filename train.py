from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from data import mean, std, device, get_gram, get_image, denormalize_img
from model import FeatureExtractor

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


img_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

style_img = get_image('style.jpeg', img_transform)
content_img = get_image('content.png', img_transform)

#generated_img = nn.Parameter(torch.FloatTensor(content_img.size()))
generated_img = content_img.clone()
generated_img.requires_grad = True


#train
optimizer = torch.optim.Adam([generated_img], lr=0.003, betas=[0.5, 0.999])
encoder = FeatureExtractor().to(device)
encoder.eval()

content_weight = 1
style_weight = 100

for epoch in range(1):
    content_features = encoder(content_img)
    style_features = encoder(style_img)
    generated_features = encoder(generated_img)

    loss = nn.MSELoss()
    #content_loss = torch.mean((content_features[-1] - generated_features[-1])**2)
    content_loss = loss(content_features[-1], generated_features[-1])

    style_loss = 0

    for style_feature, generated_feature in zip(style_features, generated_features):
        gram_style = get_gram(style_feature)
        gram_generated = get_gram(generated_feature)

        style_loss += loss(gram_style, gram_generated)

    total_loss = style_weight * style_loss + content_weight * content_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("Epoch {} \t content_loss: {:.5f}\t style_loss: {:0.5f}".format(epoch, content_loss.item(), style_loss.item()))


result = generated_img.squeeze().detach().cpu()
result = denormalize_img(result)
print(result.shape)
img = result * 255
img = img.astype(np.uint8)
img = Image.fromarray(img)
img.save('result.jpg')
plt.imshow(result)
plt.show()
