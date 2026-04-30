import torch
import os
import tqdm
import numpy as np
from UNet3Dv2 import UNet3D
from dataset import splitting_all
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

torch.set_float32_matmul_precision('high')
import monai.transforms as mtransforms
import argparse
import torch.nn as nn
import loss

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # General
    parser.add_argument('--name', type=str, default='myjob')
    parser.add_argument('--save_model', type=str, default='C:\\Users\\bdenisde\\Documents\\Donnees\\tmp\\CNN_IRE_floating_potential\\model')
    parser.add_argument('--save_writer', type=str, default='C:\\Users\\bdenisde\\Documents\\Donnees\\tmp\\CNN_IRE_floating_potential\\runs')
    parser.add_argument('--precise', default=False, action='store_true')
    # Dataset
    parser.add_argument('--data_folder', type=str, default='C:\\Users\\bdenisde\\Documents\\Donnees\\tmp\\CNN_IRE_floating_potential\\data25')
    parser.add_argument('--no_output_normalization', default=True,action='store_false')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seeding', type=int, default=0)
    # Model
    # parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--DS', type=int, default=-1)

    opt = parser.parse_args()
    
    name = opt.name
    EPOCHS = opt.epoch
    batch_size = opt.batch_size
    lr = opt.lr
    data_folder = opt.data_folder
    fold = opt.fold
    seeding = opt.seeding
    depth = opt.depth
    deep_supervision = opt.DS
    if deep_supervision==-1:
        deep_supervision=depth-2
    out_channels = 1
    precise = opt.precise
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
    
    
    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)

    # Report split sizes
    print('Training set has {} instances'.format(len(train_set)))
    print('Validation set has {} instances'.format(len(val_set)))

    dataiter = iter(training_loader)
    input, gt, _ = next(dataiter)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet3D(depth=depth, in_channels=1, out_channels=out_channels, between_channels = 64, deep_supervision=deep_supervision, size=input.shape[-3])
    model = model.to(device)
    # compiled_model = torch.compile(model)
    compiled_model = model
    
    loss_train_fn = loss.LnLoss(1)
    loss_val_fn = loss.LnLoss(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1.0e-11, mode='min')
    scaler = torch.cuda.amp.GradScaler()
    

    def train_one_epoch(epoch_index, tb_writer=None):
        running_loss = 0.
        running_loss_1 = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in tqdm.tqdm(enumerate(training_loader)):   
            
            # Every data instance is an input + label pair
            input, gt, _ = data
            
            input = input.to(device)
            gt = gt.to(device)
    
            optimizer.zero_grad()

            # Make predictions for this batch
            # with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16, enabled=True):
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available()):
                
                outputs = compiled_model(input)
                if output_normalization:
                    outputs = [(outputs[i]-torch.amin(outputs[i], dim=(-3,-2,-1), keepdim=True))/(torch.amax(outputs[i], dim=(-3,-2,-1), keepdim=True)-torch.amin(outputs[i], dim=(-3,-2,-1), keepdim=True)) for i in range(len(outputs))]
                # Compute the loss and its gradients
                mask = None
                if precise == True :
                    mask = input.clone()
                loss_1 = loss_train_fn(outputs, gt, mask)
                loss = loss_1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Gather data and report
            running_loss += loss.item()
            running_loss_1 += loss_1.item()

            number_iter_report = np.floor(len(train_set)/5/batch_size)
            if number_iter_report == 0:
                number_iter_report = 1
            if i % number_iter_report == number_iter_report-1:
                last_loss = running_loss / number_iter_report # loss per batch
                last_loss_1 = running_loss_1 / number_iter_report
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.
                running_loss_1 = 0.
        return last_loss

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0

    best_vloss = 1_000_000.
    patience = 0
    for epoch in range(epoch_number,EPOCHS):
        patience += 1
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        compiled_model.train(True)
        avg_loss = train_one_epoch(epoch)

        running_vloss = 0.0
        running_vloss_1 = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        compiled_model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinput, vgt, _ = vdata
                vinput = vinput.to(device)
                vgt = vgt.to(device)
                # with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16, enabled=False):
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available()):

                    voutputs = compiled_model(vinput)
                    if output_normalization:
                        voutputs[0] = (voutputs[0]-torch.amin(voutputs[0], dim=(-3,-2,-1), keepdim=True))/(torch.amax(voutputs[0], dim=(-3,-2,-1), keepdim=True)-torch.amin(voutputs[0], dim=(-3,-2,-1), keepdim=True))
                    vmask = None
                    if precise == True :
                        vmask = vinput
                    vloss_1 = loss_val_fn(voutputs[0], vgt, vmask)
                    vloss = vloss_1

                running_vloss += vloss
                running_vloss_1 += vloss_1
        avg_vloss = running_vloss / (i + 1)
        avg_vloss_1 = running_vloss_1 / (i + 1)
        
        scheduler.step(avg_vloss)
        
        print('LOSS train {} valid loss {}'.format(avg_loss, avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss or epoch%10 == 9:
            model_path = os.path.join(opt.save_model,'model_{}_{}'.format(name, timestamp))
            if avg_vloss < best_vloss:
                patience = 0 
                best_vloss = avg_vloss
                torch.save(model.state_dict(), model_path+'_best_vloss')
                print('Model saved to {}'.format(model_path+'_best_vloss'))
            if epoch%10 == 9:
                torch.save(model.state_dict(), model_path+'_last')
                print('Model saved to {}'.format(model_path+'_last'))
                # torch.save({
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'scheduler_state_dict': scheduler.state_dict(),
                #     'scaler_state_dict': scaler.state_dict(),
                #     'best_vloss': best_vloss,
                #     'epoch': epoch,
                #     'MAX_EPOCH': EPOCHS,
                #     'timestamp': timestamp
                # }, model_path+'_last')
                    
        if patience > 150:
            break
