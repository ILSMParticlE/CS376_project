import torch
from PIL import Image
from torchvision import transforms as T

PATH = './imgs/train_all/img_5.jpg'

to_image = T.ToPILImage()
center_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), T.Grayscale(num_output_channels=1)])
gray_transform = T.Compose([T.Resize(400), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), T.Grayscale(num_output_channels=1)])

def head_tensor(img):
    img_tensor = gray_transform(img)
    h_start = 20
    w_start = 80
    img_tensor = img_tensor[:, h_start:224+h_start, w_start:224+w_start]
    return img_tensor

def hand_tensor(img):
    img_tensor = gray_transform(img)
    img_tensor = img_tensor[:, -224:, -224:]
    return img_tensor

def jacobkie_input(path):
    img = Image.open(path).convert('RGB')
    result = torch.stack([center_transform(img).squeeze(), head_tensor(img).squeeze(), hand_tensor(img).squeeze()], dim=0)
    return result

img = Image.open(PATH).convert('RGB')
print(img)

img_tensor = jacobkie_input(PATH)
overall = center_transform(img)
hand = hand_tensor(img)
head = head_tensor(img)
new_img = to_image(overall)

print(img_tensor.shape)
new_img.show()