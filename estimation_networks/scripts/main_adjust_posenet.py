'''
Author: Dianye Huang
Date: 2022-08-18 11:39:53
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-09-06
Description: 
    Finetune the pretrained model to mitigate the 
    sim2real gap.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

import os
from ae_pose_utils import PoseAutoEncoder
from resnet18_utils import ResNetImgAE18
from main_train_ced_sim import PoseImageAutoencoder

class PoseAdjustNet(nn.Module):
    def __init__(self, premodel:PoseImageAutoencoder, latent_dim=64, adjust_step=int(1)):
        super().__init__()
        self.latent_dim  = latent_dim
        self.adjust_step = adjust_step 
        self.eps = 1e-9
        self.clamp_eps = 1e-7
        
        self.pose_img = premodel
        self.poses_adjust_header = nn.Sequential( # directly output the number
            nn.Linear(latent_dim, latent_dim*4),
            nn.ReLU(inplace=True),
            
            nn.Linear(latent_dim*4, latent_dim*2),
            nn.ReLU(inplace=True),
            
            nn.Linear(latent_dim*2, latent_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(latent_dim, 6),            
        )    
        self.pose_final_adjust = nn.Linear(6, 6)
        self.initialize_module(self.poses_adjust_header)
        self.initialize_module(self.pose_final_adjust)
    
    def initialize_module(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def freeze_encdec(self, freeze_flag):
        if freeze_flag:
            for param in self.pose_img.pose_ae.encoder.parameters():
                param.requires_grad = False
            for param in self.pose_img.pose_ae.decoder.parameters():
                param.requires_grad = False
        else:
            for param in self.pose_img.pose_ae.encoder.parameters():
                param.requires_grad = True
            for param in self.pose_img.pose_ae.decoder.parameters():
                param.requires_grad = True
    
    def correlation_loss(self, pred:torch.Tensor, target:torch.Tensor):
        # pearson correlation loss
        mean_pred = torch.mean(pred, dim=0).unsqueeze(0)
        mean_tar  = torch.mean(target, dim=0).unsqueeze(0)
        
        pred_c    = pred - mean_pred
        tar_c     = target - mean_tar
        
        numer = torch.sum(pred_c * tar_c, dim=0)
        denom = torch.sqrt(
                    torch.sum(pred_c**2, dim=0) * \
                    torch.sum(tar_c**2, dim=0) + self.eps # due to the self.eps, the minimal value of corr is not zero
                ) 
        corr = (numer/denom).view(-1)
        
        corr = torch.clamp(corr, -1.0+self.clamp_eps, 1.0-self.clamp_eps)
        corr = 0.5*torch.log((1.0+corr)/(1.0-corr))
        corr = (-torch.mean(corr)/8.5+1.0)/2.0 # 0.0 to 1.0
        
        return corr
                
    def forward(self, images):
        pred_list = list()
        # get prediction
        img_embeds = self.pose_img.imgs_ae.encoder(images)
        pred_poses = self.pose_img.pose_ae.decoder(img_embeds)
        pred_list.append(pred_poses)
        
        pose_embeds = self.pose_img.pose_ae.encoder(pred_poses)
        adjust_poses = pred_poses + self.poses_adjust_header(pose_embeds)
        final_poses =self.pose_final_adjust(adjust_poses)
        rec_image = self.pose_img.imgs_ae.decoder(img_embeds)
        
        return final_poses, (img_embeds, pose_embeds, rec_image)

# Evaluation
import numpy as np
def get_trans_rotation_err(pred, target):
    avg_terr = np.mean(np.linalg.norm(pred[:, :3]-target[:, :3], axis=1))*1e3 # mm
    avg_rerr = np.mean(np.linalg.norm(pred[:, 3:]-target[:, 3:], axis=1))/np.pi*180.0 # deg
    return avg_terr, avg_rerr

from scipy.spatial.transform import Rotation as R

def euler_angle_error(euler1, euler2, degrees=True):
    r1 = R.from_euler('xyz', euler1, degrees=degrees)
    r2 = R.from_euler('xyz', euler2, degrees=degrees)
    r_diff = r1.inv() * r2
    angle_error = r_diff.magnitude()
    if degrees:
        return np.degrees(angle_error)
    else:
        return angle_error

def test_model(model:PoseAdjustNet, dataloader, device, offset, normal, balance):
    model.eval()
    trans_err = None
    angle_err = None
    with torch.no_grad():
        avg_trans = 0.0
        avg_angle = 0.0
        for images, labels, poses in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            pred_poses, _ = model(images)
            
            pred=predict_postproc(pred_poses.cpu().detach().numpy(),
                                    offset=offset, normal=normal, balance=balance)
            target=poses.cpu().detach().numpy()
            rec_terr = np.linalg.norm(pred[:, :3]-target[:, :3], axis=1)*1e3 # mm
            
            # compute orientation error:
            rec_rerr = list()
            for euler1, euler2 in zip(pred[:, 3:], target[:, 3:]):
                euler_err = euler_angle_error(euler1, euler2, degrees=False)
                rec_rerr.append(euler_err/np.pi*180.0)
            rec_rerr = np.array(rec_rerr)
            
            
            if trans_err is None:
                trans_err = rec_terr
            else:
                trans_err = np.hstack((trans_err, rec_terr))
                
            if angle_err is None:
                angle_err = rec_rerr
            else:
                angle_err = np.hstack((angle_err, rec_rerr))
            num_sample = len(labels)
            avg_trans  += np.mean(rec_terr)*num_sample
            avg_angle  += np.mean(rec_rerr)*num_sample
        num_all = len(dataloader.dataset)
        avg_trans  /= num_all
        avg_angle  /= num_all
        
        print('trans_err shape: ', trans_err.shape)
        print('angle_err shape: ', angle_err.shape)
        print(f'translation error: {np.mean(trans_err)}+-{np.std(trans_err)} mm. ')
        print(f'orientation error: {np.mean(angle_err)}+-{np.std(angle_err)} deg. ')
        print(f'pose terr: {avg_trans} mm, pose rerr: {avg_angle} deg.')
    
    return trans_err, angle_err


from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from dataset_utils import MultiCamDataset, predict_postproc
from ae_training_utils import set_seed, create_folder_if_not_exists, EarlyStopping
from torch import optim
from tqdm import tqdm
import os
from datetime import datetime
set_seed() # for reproducibility

# Load Dataset
# - meta parameters for loading datasets
#   -- for loading and spliting data
batch_size  = int(16)
num_workers = 4
pin_memory  = True # accelerate loading the data if has enough ram

training_data_dir   = 'xx/Realworld/training'
validation_data_dir = 'xx/Realworld/validation'
testing_data_dir    = 'xx/Realworld/testing'

#   -- for proccessing labels
offset  = True
normal  = False # TODO
balance = True
image_size = (512, 512)
train_dataset = MultiCamDataset(
            folder_path=training_data_dir,
            transform=transforms.Compose([
                transforms.ToTensor(), # pixel values is {0.0, 1.0}
            ]),
            offset=offset,
            normal=normal,
            balance=balance,
            imgresize=image_size,
            vstack=True, # the left and right images from the dual cameras stacked vertically
            gray=False,   # False: #input_channels=3, True: #input_channels=1
            pin_image=False, # TODO true when have enough cpu-ram
        )
train_dataset.set_retpose(True)

valid_dataset = MultiCamDataset(
            folder_path=validation_data_dir,
            transform=transforms.Compose([
                transforms.ToTensor(), # pixel values is {0.0, 1.0}
            ]),
            offset=offset,
            normal=normal,
            balance=balance,
            imgresize=image_size,
            vstack=True, # the left and right images from the dual cameras stacked vertically
            gray=False,   # False: #input_channels=3, True: #input_channels=1
            pin_image=False, # TODO true when have enough cpu-ram
        )
valid_dataset.set_retpose(True) # to get the original pose

test_dataset = MultiCamDataset(
            folder_path=testing_data_dir,
            transform=transforms.Compose([
                transforms.ToTensor(), # pixel values is {0.0, 1.0}
            ]),
            offset=offset,
            normal=normal,
            balance=balance,
            imgresize=image_size,
            vstack=True, # the left and right images from the dual cameras stacked vertically
            gray=False,   # False: #input_channels=3, True: #input_channels=1
            pin_image=False, # TODO true when have enough cpu-ram
        )
test_dataset.set_retpose(True) # to get the original pose

num_train = int(len(train_dataset))
# num_train = int(0.5*len(train_dataset))
ds_train_dataset, _ = random_split(train_dataset, [num_train, len(train_dataset)-num_train])
train_loader = DataLoader(ds_train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=pin_memory)

valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)
test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)

# load model
LoadModel  = False
LoadPretrain = True
model_name = 'xxxx'
package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# - save model
model_path = os.path.join(package_dir, f'model/best_autoencoder_{model_name}.pth')
last_model_path = os.path.join(package_dir, f'model/last_autoencoder_{model_name}.pth')

# - log data
log_every_step = int(50)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tb_folder = os.path.join(package_dir, f'model/logs_{model_name}/{timestamp}')
create_folder_if_not_exists(tb_folder)
log_every_step = max(log_every_step, len(ds_train_dataset.dataset)//batch_size//log_every_step)

# - init model
latent_dim = 64 # TODO
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pretrained = PoseImageAutoencoder(
            PoseAE = PoseAutoEncoder(
                in_dim=6, # trans and eulers, 6 in total
                hidden_dims=[128, 256],
                latent_dim=latent_dim,
                normalize=False,
                batchnorm=False
            ),
            ImageAE =  ResNetImgAE18(
                img_size=512,
                in_chs=3,
                latent_dim=latent_dim,
                ret_emed=True
            )
        )
if LoadPretrain:
    # premodel_name = 'resnet_20w_512_finetune'
    premodel_name = 'resnet_20w_512'
    premodel_path = os.path.join(package_dir, f'model/best_autoencoder_{premodel_name}.pth')
    pretrained.load_state_dict(torch.load(premodel_path, weights_only=True))
    print('pretrained model loading done!')

model = PoseAdjustNet(
    premodel=pretrained
).to(device)
model.freeze_encdec(False)

# Training
# - log training data
from torch.utils.tensorboard import SummaryWriter
tb_writer  = SummaryWriter(log_dir=tb_folder)

# - meta parameters for training
num_epochs = 40
patience = 8
# learning_rate = 5e-4
learning_rate = 7e-4
min_lr = 1e-5
Tmax = int(num_epochs)
# valid_loss_min = np.inf

# - meta parameters for constructing loss
weight_decay = 1e-5
max_norm     = 2.0

if not LoadModel:
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Tmax, eta_min=min_lr)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    step = 0 # step inc with each input batch
    pbar = tqdm(range(num_epochs), desc = 'description')

    scaler = GradScaler()

    for epoch in pbar:
        # -- epoch training
        model.train()
        running_loss = 0.0
        for idx, (images, labels, _) in enumerate(train_loader): # mind that the pose_ae encodes the labels
            # -- batch calc losses
            labels = labels.to(device)
            images = images.to(device)

            optimizer.zero_grad()
            with autocast(device_type=device, dtype=torch.float16):
                # pred_poses = model(images)
                poses, feas = model(images)
                
                loss = F.mse_loss(poses, labels)
                loss += 0.1*F.mse_loss(feas[2], images)

            # -- batch update network
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm) 
            scaler.step(optimizer)
            scaler.update()
            
            # -- batch log info
            step += 1
            running_loss += loss.detach()*labels.size(0)

            if step % log_every_step == 0:
                progress = float(idx)/len(train_loader)*100.0
                pbar.set_description(f'Training: {epoch+1}/{num_epochs}, Progress: {progress:3.0f}%, Loss: {loss:6f}')
                tb_writer.add_scalar('Loss/Batch/loss', loss.item(), step)
                
        if epoch < Tmax:
                scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # --epoch validation
        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            for idx, (images, labels, poses) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                pred_poses, _ = model(images)
                
                terr, rerr = get_trans_rotation_err(
                                pred=predict_postproc(pred_poses.cpu().detach().numpy(),
                                                    offset=offset, normal=normal, balance=balance),
                                target=poses.cpu().detach().numpy()
                            )
                val_loss = terr + rerr # score function to monitor the learning
                running_val_loss += val_loss*labels.size(0)
                # update progress bar
                if step % log_every_step == 0:
                    progress = float(idx)/len(valid_loader)*100.0
                    pbar.set_description(f'Validation: {epoch+1}/{num_epochs}, Progress: {progress:3.0f}%, Valid Loss: {val_loss:2f}')
            avg_val_loss = running_val_loss / len(valid_loader.dataset)

        if early_stopping(avg_val_loss):
            # Save the trained model
            torch.save(model.state_dict(), model_path)
            print(f"Avg val loss: {avg_val_loss:2f}, Model saved!")

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

        # -- epoch logging
        current_lr = scheduler.get_last_lr()[0]
        tb_writer.add_scalar('Loss/Epoch/lr', current_lr, epoch)
        tb_writer.add_scalar('Loss/Epoch/total', epoch_loss, epoch)
        tb_writer.add_scalar('Loss/Epoch/val_loss', avg_val_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.6f}, Learning Rate: {current_lr:.6f}")
    # Save the trained model
    torch.save(model.state_dict(), last_model_path)
    print(f"avg val loss: {avg_val_loss:2f}, Last Model saved!")
    
else:
    pass
    
model.load_state_dict(torch.load(model_path, weights_only=True))
model = model.to(device)
print("Model loaded!")
print('-----------------------------------------')
# print('Training results:')
# test_model(model, train_loader, device, offset, normal, balance)
print('Validation results:')
trans_err1, angle_err1 = test_model(model, valid_loader, device, offset, normal, balance)
print('Testing results:')
trans_err2, angle_err2 = test_model(model, test_loader, device, offset, normal, balance)

all_trans = np.hstack((trans_err1, trans_err2))
all_angle = np.hstack((angle_err1, angle_err2))
print(f'TOTOAL Translation error: {np.mean(all_trans)}+-{np.std(all_trans)}. ')    
print(f'TOTOAL Orientation error: {np.mean(all_angle)}+-{np.std(all_angle)}. ')   
# print('Resting results:')
# test_model(model, preserve_loader, device, offset, normal, balance)
print('-----------------------------------------')
print('Finished!')
