import torch
import numpy as np
from UNet3Dv2 import UNet3D
from dataset import splitting_all

torch.set_float32_matmul_precision('high')
import monai.transforms as mtransforms
import argparse
import torch.nn as nn
import SimpleITK as sitk
import nibabel as nib
import tqdm
import loss

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset
    parser.add_argument('--data_folder', type=str, default='C:\\Users\\bdenisde\\Documents\\Donnees\\tmp\\CNN_IRE_floating_potential\\data25')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seeding', type=int, default=0)
    parser.add_argument('--no_output_normalization', default=True,action='store_false')
    # Model
    # parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--DS', type=int, default=-1)

    opt = parser.parse_args()

    # torch.autograd.set_detect_anomaly(True)
    
    batch_size = opt.batch_size
    data_folder = opt.data_folder
    fold = opt.fold
    seeding = opt.seeding
    depth = opt.depth
    deep_supervision = opt.DS
    if deep_supervision==-1:
        deep_supervision=depth-2
    out_channels = 1
    output_normalization = opt.no_output_normalization

    keys_gt = ['input', 'gt']
    transform = mtransforms.Compose(
                [
                    # mtransforms.ToDevice(keys = keys, device = torch.device('cuda:0')), 
                    # Lazy Transforms
                    mtransforms.RandRotated(keys = keys_gt, prob = 0.2, range_x=30/180*np.pi, range_y=30/180*np.pi, range_z=30/180*np.pi, lazy = True, padding_mode='zeros', mode='nearest'),
                    mtransforms.RandFlipd(keys = keys_gt, prob = 0.5, spatial_axis=-1 ,lazy=True),
                    mtransforms.RandFlipd(keys = keys_gt, prob = 0.5, spatial_axis=-2 ,lazy=True),
                    mtransforms.RandFlipd(keys = keys_gt, prob = 0.5, spatial_axis=-3 ,lazy=True),
                    # Non Lazy Transforms
                    mtransforms.RandSimulateLowResolutiond(keys = keys_gt, prob = 0.1)
            ])

    train_set, val_set, test_set = splitting_all(data_folder, transform, fold=fold, seeding=seeding)
    
    
    # Create data loaders for our datasets
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)


    # Report split sizes
    print('Testing set has {} instances'.format(len(test_set)))

    dataiter = iter(test_loader)
    input, gt, _ = next(dataiter)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet3D(depth=depth, in_channels=1, out_channels=out_channels , between_channels = 64, deep_supervision=deep_supervision, size=input.shape[-3])
    model.load_state_dict(torch.load('C:\\Users\\bdenisde\\Documents\\Donnees\\tmp\\CNN_IRE_floating_potential\\model\\model_myjob_20260411_094322_best_vloss', map_location=device))
    # model.load_state_dict(torch.load('./model/model_L2_no_output_normalization_20241125_180109_best_vloss', map_location=device))
    model = model.to(device)
    # compiled_model = torch.compile(model)
    compiled_model = model
    

    vloss_MSE = []
    vloss_gradMAE = []
    vloss_normegradMAE = []
    vloss_MAE = []
    max_MAE = []
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    compiled_model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in tqdm.tqdm(enumerate(test_loader)):
            vinput, vgt, name = vdata
            vinput = vinput.to(device)
            vgt = vgt.to(device)
            
            # with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16, enabled=False):
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available()):
                voutputs = compiled_model(vinput)
                # Several unsqueeze
                if output_normalization:
                    output = (voutputs[0]-torch.min(voutputs[0]))/(torch.max(voutputs[0])-torch.min(voutputs[0]))
                else :
                    output = voutputs[0]
                gt = vgt
                for j, array in enumerate(output):
                    max_MAE.append(torch.max(torch.abs(array-gt[j])).cpu().numpy())
                    image_nifti = sitk.GetImageFromArray(array.squeeze().cpu().numpy())
                    image_nifti.SetDirection(tuple(name[1][j].numpy()))
                    image_nifti.SetOrigin(tuple(name[2][j].numpy()))
                    image_nifti.SetSpacing(tuple(name[3][j].numpy()))
                    # sitk.WriteImage(image_nifti, './output/'+name[0][j])
                    sitk.WriteImage(image_nifti, 'C:\\Users\\bdenisde\\Documents\\Donnees\\tmp\\CNN_IRE_floating_potential\\output\\'+name[0][j])

                vloss_MSE.append(torch.mean(nn.MSELoss(reduction='none')(output, gt), dim=(1,2,3,4)).cpu().numpy())
                vloss_MAE.append(torch.mean(nn.L1Loss(reduction='none')(output, gt), dim=(1,2,3,4)).cpu().numpy())
                vloss_gradMAE.append(torch.mean(nn.L1Loss(reduction='none')(loss.grad(output), loss.grad(gt)), dim=(1,2,3,4)).cpu().numpy())
                vloss_normegradMAE.append(torch.mean(nn.L1Loss(reduction='none')(torch.linalg.vector_norm(loss.grad(output), dim=1), torch.linalg.vector_norm(loss.grad(gt), dim=1)), dim=(1,2,3)).cpu().numpy())
    
    print("MSE u")
    vloss_MSE = np.concatenate(vloss_MSE)
    print(vloss_MSE.shape)
    print(np.mean(vloss_MSE))
    print(np.std(vloss_MSE))
    print(np.max(vloss_MSE))

    print("MAE u")
    vloss_MAE = np.concatenate(vloss_MAE)
    print(vloss_MAE.shape)
    print(np.mean(vloss_MAE))
    print(np.std(vloss_MAE))
    print(np.max(vloss_MAE))

    print("Erreur maximale u")

    max_MAE = np.array(max_MAE)
    print(max_MAE.shape)
    print(np.mean(max_MAE))
    print(np.std(max_MAE))
    print(np.max(max_MAE))

    print("MAE grad u")

    vloss_gradMAE = np.concatenate(vloss_gradMAE)
    print(vloss_gradMAE.shape)
    print(np.mean(vloss_gradMAE))
    print(np.std(vloss_gradMAE))
    print(np.max(vloss_gradMAE))

    print('MAE norme grad')

    vloss_normegradMAE = np.concatenate(vloss_normegradMAE)
    print(vloss_normegradMAE.shape)
    print(np.mean(vloss_normegradMAE))
    print(np.std(vloss_normegradMAE))
    print(np.max(vloss_normegradMAE))

    
