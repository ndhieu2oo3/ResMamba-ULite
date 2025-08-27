import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

data = np.load("./data/ISIC2018_train_192_256.npz")
x, y = data["image"], data["mask"]
test_size = int((20/100)*x.shape[0])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=257)


class RandomCrop(transforms.RandomResizedCrop):
    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        for imgCount in range(len(imgs)):
            imgs[imgCount] = transforms.functional.resized_crop(imgs[imgCount], i, j, h, w, self.size, self.interpolation)
        return imgs
class ISICLoader(Dataset):
    def __init__(self, images, masks,
                 transform=True, typeData = "train"):
        self.transform = transform if typeData == "train" else False  # augment data bool
        self.typeData = typeData
        self.images = images
        self.masks = masks
        # self.edge_extractor = EdgeExtractor()  # Initialize edge extractor

    def __len__(self):
        return len(self.images)

    def rotate(self, image, mask, degrees=(-15,15), p=0.5):
        if torch.rand(1) < p:
            degree = np.random.uniform(*degrees)
            image = image.rotate(degree, Image.NEAREST)
            mask = mask.rotate(degree, Image.NEAREST)
        return image, mask
    def horizontal_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask
    def vertical_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask
    def random_resized_crop(self, image, mask, p=0.1):
        if torch.rand(1) < p:
            image, mask = RandomCrop((192, 256), scale=(0.8, 0.95))([image, mask])
        return image, mask

    def augment(self, image, mask):
        image, mask = self.random_resized_crop(image, mask)
        image, mask = self.rotate(image, mask)
        image, mask = self.horizontal_flip(image, mask)
        image, mask = self.vertical_flip(image, mask)
        return image, mask

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        mask = Image.fromarray(self.masks[idx])
    ####################### augmentation data ##############################
        if self.transform:
            image, mask = self.augment(image, mask)
        image = transforms.ToTensor()(image)
        mask = np.asarray(mask, np.int64)
        mask = torch.from_numpy(mask[np.newaxis])

        # # Extract edge map
        # edge_map = self.edge_extractor(np.array(image.permute(1, 2, 0)))  # Convert to numpy and apply edge extractor

        # return image, mask, edge_map
        return image, mask
    
train_dataset = DataLoader(ISICLoader(x_train, y_train), batch_size=4, pin_memory=True, shuffle=True, num_workers=2, drop_last=True, prefetch_factor = 8)
test_dataset = DataLoader(ISICLoader(x_test, y_test, typeData="test"), batch_size=1, num_workers=2, prefetch_factor=16)
