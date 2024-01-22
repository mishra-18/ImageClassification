import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import os
import scipy.io
import albumentations 
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def transform():

        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def loaddata():

        flowerdata = os.listdir("flowers/jpg")       # have images
        file_paths = list(map(lambda item: "/flowers/jpg/" + item, flowerdata)) 
        imagelabel = scipy.io.loadmat("/flowers/102flowers/imagelabels.mat") # an array of lenght same as flowerdata but with label/class
        image_label = dict(enumerate((imagelabel["labels"][0]).tolist(), 1)) 
        setid = scipy.io.loadmat("/flowers/102flowers/setid.mat")  # contains three array's with indexes for trn, val, tst
        train = setid["trnid"][0]
        valid = setid["valid"][0]
        test = setid["tstid"][0]


        train_path = []
        val_path = []
        test_path = []
        working_dir = "flowers/jpg/"
        for trn_id in train:
            train_path.append(working_dir + f"image_{trn_id:05d}.jpg")
        for val_id in valid:
            val_path.append(working_dir + f"image_{val_id:05d}.jpg")
        for tst_id in test:
            test_path.append(working_dir + f"image_{tst_id:05d}.jpg")


        class CustomDataset(Dataset):    
            def __init__(self, trn, data_path, file_paths):
                self.data_path = data_path
                self.train = trn
                self.file_paths = file_paths
            def __len__(self):
                return len(self.data_path)
            def __getitem__(self, idx):
                path = self.data_path[idx]
                current_idx = int(self.data_path[idx].split("/")[4].split(".")[0].split("_")[1])

                flower = Image.open(path)
                label = image_label[current_idx]

                if self.train:
                    flower = transform.transform(flower)
                else:
                    flower = transform.transform_val(flower)
                return flower, label - 1
            

        train_data = CustomDataset(True, train_path, file_paths)
        valid_data = CustomDataset(False, val_path, file_paths)    

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True)

        return train_loader, valid_loader