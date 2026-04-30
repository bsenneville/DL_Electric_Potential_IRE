from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import glob
import os
import torch
import SimpleITK as sitk
from monai.data.meta_tensor import MetaTensor
import numpy as np

def readFolder(data_folder):
    paths = []
    for file in glob.glob(os.path.join(data_folder, 'f_*')):
        paths.append([file,file.replace('f_', 'g_'), file.replace('f_', 'u_')])
    return paths


def splitting_all(data_folder, transform, fold, seeding):
    number_fold = 5
    paths = readFolder(data_folder=data_folder)
    paths = np.array(paths)
    
    # Train+Val vs Test
    kf = KFold(n_splits=2)
    gen = kf.split(paths)
    for i in range(fold):
        next(gen)
    train_val_index, test_index = next(gen)
    paths_train_val = paths[train_val_index]
    paths_test = paths[test_index]

    # Train vs Val
    kf = KFold(n_splits=number_fold, shuffle=True, random_state=seeding)
    train_index, val_index = next(kf.split(paths_train_val))
    paths_train = paths_train_val[train_index]
    paths_val = paths_train_val[val_index]

    train_set = DatasetCarto(paths=paths_train, transform=transform)
    val_set = DatasetCarto(paths=paths_val, transform=None)
    test_set = DatasetCarto(paths=paths_test, transform=None)

    return train_set, val_set, test_set

class DatasetCarto(Dataset):
    def __init__(self, paths, transform) -> None:
        super().__init__()
        self.paths = paths
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        f_image = sitk.ReadImage(path[0])
        g_image = sitk.ReadImage(path[1])
        u_image = sitk.ReadImage(path[2])
        f_image = sitk.Cast(f_image, sitk.sitkFloat32)
        g_image = sitk.Cast(g_image, sitk.sitkFloat32)

        u_image = sitk.Cast(u_image, sitk.sitkFloat32)
        f_array = sitk.GetArrayFromImage(f_image)
        g_array = sitk.GetArrayFromImage(g_image)
        u_array = sitk.GetArrayFromImage(u_image)

        f_array *= -1
        f_array[g_array==1] = 1
        
        u_array = u_array[None, :]
        f_array = f_array[None, :]

        infos = [os.path.basename(path[2]), np.array(u_image.GetDirection()), np.array(u_image.GetOrigin()), np.array(u_image.GetSpacing())]
        data = {
            'input':torch.tensor(f_array),
            'gt': torch.tensor(u_array)
        }

        if self.transform:
            data = self.transform(data)

        input = data['input'].as_tensor() if isinstance(data['input'], MetaTensor) else data['input']
        gt = data['gt'].as_tensor() if isinstance(data['gt'], MetaTensor) else data['gt']

        return input, gt, infos 
    
    def __len__(self):
        return len(self.paths)
    
