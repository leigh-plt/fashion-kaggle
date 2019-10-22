from torchvision import transforms as T
from PIL import Image


# Transform function
train_transform = T.Compose([
    Image.fromarray,
    T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0., hue=0.),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) ])

infer_transform = T.Compose([
    Image.fromarray,
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) ])