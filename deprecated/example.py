from constants import NYUv2_ROOT
from dataset.nyuv2.nyuv2_dataset import NYUv2Dataset
from torchvision import transforms
import os

print(os.getcwd())
t = transforms.Compose([transforms.RandomCrop(400), transforms.ToTensor()])
NYUv2Dataset(
    root=os.path.join(NYUv2_ROOT),
    download=True,
    train=True,
    rgb_transform=t,
    seg_transform=t,
    sn_transform=None,
    depth_transform=t,
    instance_transform=t,
)
