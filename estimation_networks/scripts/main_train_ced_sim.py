'''
Author: Dianye Huang
Date: 2022-08-18 11:39:53
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-09-06
Description: 
    Train the Cross Encoder-Decoder (CED) module with 
    the generated images
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

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


# Encoder-Decoder with latent vector expansion
from ae_pose_utils import PoseAutoEncoder, get_triplet_indices, calc_triplet_loss, get_combinations
# from ae_imgs_utils import ImageAutoEncoder
from resnet18_utils import ResNetImgAE18
class PoseImageAutoencoder(nn.Module):
    def __init__(self, PoseAE:PoseAutoEncoder, ImageAE:ResNetImgAE18):
        super(PoseImageAutoencoder, self).__init__()
        self.pose_ae = PoseAE
        self.imgs_ae = ImageAE

    def forward(self, images, labels):
        rec_imgs, emb_imgs = self.imgs_ae(images)
        rec_pose, emb_pose = self.pose_ae(labels)
        return rec_imgs, emb_imgs, rec_pose, emb_pose

# Calculate losses for training
def calc_losses(model:PoseImageAutoencoder, images:torch.Tensor,
                labels:torch.Tensor, sample_comlist=None,
                balance=True, spatial_margin=0.2, trip_input=None,
                normal=False, trip_num=None, img_emb_ratio=0.7):
    rec_imgs, emb_imgs, rec_pose, emb_pose = model(images, labels)

    # Reconstruction loss
    rec_imgs_loss = F.mse_loss(rec_imgs, images)  # - images

    if not normal:
        rec_pose_loss = F.mse_loss(rec_pose, labels)  # - poses
    else:
        rec_pose_loss  = 7.5*F.mse_loss(rec_pose[:, :3], labels[:, :3])
        rec_pose_loss += 0.5*F.mse_loss(rec_pose[:, 3:], labels[:, 3:])

    # Cross-embedding loss
    cross_latent_loss  = img_emb_ratio*F.mse_loss(emb_imgs, emb_pose.detach()) 
    cross_latent_loss += (1-img_emb_ratio)*F.mse_loss(emb_imgs.detach(), emb_pose)

    # Spatial relationship loss
    pose_trip_loss = cross_latent_loss.detach()*0.0
    if sample_comlist is not None:
        # learning with triplet loss
        # - processing input_indices
        tri_indices = get_triplet_indices( # pairs the negative and positive samples based on the poses
                            sample_comlist,
                            labels=labels,
                            balance=balance,
                            num_samples=trip_num # TODO limit the number of samples
                        )
        # - processing input sample batches
        if len(tri_indices.size()) > 1:
            pose_trip_loss = calc_triplet_loss(emb_pose, tri_indices, inputs=trip_input, margin=spatial_margin) # pose_embedding

    # Cross reconstruction
    cross_rec_pose = model.pose_ae.decoder(emb_imgs)
    crec_pose_loss = F.mse_loss(cross_rec_pose, labels)

    cross_rec_imgs = model.imgs_ae.decoder(emb_pose)
    crec_imgs_loss = F.mse_loss(cross_rec_imgs, images)

    return {'rec_imgs':rec_imgs_loss, 'rec_pose':rec_pose_loss,
            'tri_pose':pose_trip_loss, 'crs_latent':cross_latent_loss,
            'crec_imgs':crec_imgs_loss, 'crec_pose':crec_pose_loss}

# Evaluation
import numpy as np
def get_trans_rotation_err(pred, target):
    avg_terr = np.mean(np.linalg.norm(pred[:, :3]-target[:, :3], axis=1))*1e3 # mm
    avg_rerr = np.mean(np.linalg.norm(pred[:, 3:]-target[:, 3:], axis=1))/np.pi*180.0 # deg
    return avg_terr, avg_rerr

def test_model(model:PoseImageAutoencoder, dataloader, device, offset, normal, balance):
    model.eval()
    trans_err = None
    angle_err = None
    with torch.no_grad():
        avg_recimgs = 0.0
        avg_rectrans = 0.0
        avg_recangle = 0.0
        avg_crecimgs = 0.0
        avg_crectrans = 0.0
        avg_crecangle = 0.0
        avg_latent = 0.0
        for images, labels, poses in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            rec_imgs, emb_imgs, rec_pose, emb_pose = model(images, labels)
            # Reconstruction error
            rec_imgs_loss = F.mse_loss(rec_imgs, images).item()
            rec_terr, rec_rerr = get_trans_rotation_err(
                                    pred=predict_postproc(rec_pose.cpu().detach().numpy(),
                                                          offset=offset, normal=normal, balance=balance),
                                    target=poses.cpu().detach().numpy()
                                )
            # Cross embedding error
            cross_latent_loss = F.mse_loss(emb_imgs, emb_pose).item()
            # Cross reconstruction error
            cross_rec_pose = model.pose_ae.decoder(emb_imgs)
            cross_rec_imgs = model.imgs_ae.decoder(emb_pose)
            crec_imgs_err = F.mse_loss(cross_rec_imgs, images).item()
            
            pred=predict_postproc(cross_rec_pose.cpu().detach().numpy(),
                                    offset=offset, normal=normal, balance=balance)
            target=poses.cpu().detach().numpy()
            crec_terr = np.linalg.norm(pred[:, :3]-target[:, :3], axis=1)*1e3 # mm
            
            # compute orientation error:
            crec_rerr = list()
            for euler1, euler2 in zip(pred[:, 3:], target[:, 3:]):
                euler_err = euler_angle_error(euler1, euler2, degrees=False)
                crec_rerr.append(euler_err/np.pi*180.0)
            crec_rerr = np.array(crec_rerr)
            
            
            if trans_err is None:
                trans_err = crec_terr
            else:
                trans_err = np.hstack((trans_err, crec_terr))
                
            if angle_err is None:
                angle_err = crec_rerr
            else:
                angle_err = np.hstack((angle_err, crec_rerr))
            
            
            num_sample = len(labels)
            avg_recimgs   += rec_imgs_loss*num_sample
            avg_rectrans  += rec_terr*num_sample
            avg_recangle  += rec_rerr*num_sample
            avg_crecimgs  += crec_imgs_err*num_sample
            avg_crectrans += np.mean(crec_terr.reshape(-1))*num_sample
            avg_crecangle += np.mean(crec_rerr.reshape(-1))*num_sample
            avg_latent    += cross_latent_loss*num_sample

        num_all = len(dataloader.dataset)
        avg_recimgs   /= num_all
        avg_rectrans  /= num_all
        avg_recangle  /= num_all
        avg_crecimgs  /= num_all
        avg_crectrans /= num_all
        avg_crecangle /= num_all
        avg_latent    /= num_all

        print('trans_err shape: ', trans_err.shape)
        print('angle_err shape: ', angle_err.shape)
        print(f'translation error: {np.mean(trans_err)}+-{np.std(trans_err)} mm. ')
        print(f'orientation error: {np.mean(angle_err)}+-{np.std(angle_err)} deg. ')

        print(f'image rec loss: {avg_recimgs}, pose rec terr: {avg_rectrans} mm, pose rec rerr: {avg_recangle} deg.')
        print(f'image crec loss: {avg_crecimgs}, pose crec terr: {avg_crectrans} mm, pose crec rerr: {avg_crecangle} deg.')
        print(f'cross latent loss: {avg_latent}.')


from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from dataset_utils import MultiCamDataset, predict_postproc

from ae_training_utils import set_seed, create_folder_if_not_exists, EarlyStopping
from torch import optim
from tqdm import tqdm
import os
from datetime import datetime

def training():
    set_seed() # for reproducibility

    # Load Dataset
    # - meta parameters for loading datasets
    #   -- for loading and spliting data
    batch_size  = int(16) # 32
    num_workers = 4
    pin_memory  = True # accelerate loading the data if has enough ram
    data_dir    = 'dir_path_to_the_dataset/xxx'
    num_list = [int(16e4), int(2e4), int(2e4)] # 20w data for training # for 20w, 8:1:1

    LoadModel  = True
    model_name = 'resnet_20w_512'
    trip_num   = batch_size//4
    log_every_step = int(100)
    log_every_step = max(log_every_step, num_list[0]//batch_size//log_every_step)
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(package_dir, f'model/best_autoencoder_{model_name}.pth')
    last_model_path = os.path.join(package_dir, f'model/last_autoencoder_{model_name}.pth')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_folder = os.path.join(package_dir, f'model/logs_{model_name}_hybrid/{timestamp}')
    create_folder_if_not_exists(tb_folder)

    #   -- for proccessing labels
    offset  = True
    normal  = False # TODO
    balance = True
    image_size = (512, 512)
    dataset = MultiCamDataset(
                folder_path=data_dir,
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
    dataset.set_retpose(True) # to get the original pose

    num_data = len(dataset)
    num_list.append(int(num_data - np.sum(np.array(num_list))))
    train_set, valid_set, test_set, _ = random_split(dataset, num_list)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,
                                pin_memory=pin_memory)
    valid_loader = DataLoader(valid_set, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,
                                pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory)
    sample_comlist = get_combinations(batch_size, num_com=3) # for triplet loss computation

    # Model init
    # - meta parameters for model structure
    latent_dim = 64 # TODO
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PoseImageAutoencoder(
                PoseAE = PoseAutoEncoder(
                    in_dim=6, # trans and eulers, 6 in total
                    hidden_dims=[128, 256],
                    latent_dim=latent_dim,
                    normalize=normal,
                    batchnorm=False
                ),
                ImageAE =  ResNetImgAE18(
                    img_size=512,
                    in_chs=3,
                    latent_dim=latent_dim,
                    ret_emed=True
                )
            ).to(device)

    # Training
    # - log training data
    from torch.utils.tensorboard import SummaryWriter
    tb_writer  = SummaryWriter(log_dir=tb_folder)

    # - meta parameters for training
    num_epochs = 100
    patience = 10
    learning_rate = 5e-4
    min_lr = 1e-5
    Tmax = int(0.9*num_epochs)
    # valid_loss_min = np.inf

    # - meta parameters for constructing loss
    spatial_margin = 0.1 # 0.1
    img_rec_factor = 2.0
    pos_rec_factor = 1.0
    trip_factor    = 1.0 # 0.0
    crosss_factor  = 2.0 
    weight_decay   = 1e-5
    max_norm       = 2.0

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

                # hybrid gradient update
                optimizer.zero_grad()
                with autocast(device_type=device, dtype=torch.float16):
                    dict_res = calc_losses(
                                    model, images, labels,
                                    sample_comlist=sample_comlist, balance=balance,
                                    spatial_margin=spatial_margin, trip_input=labels,
                                    normal=normal, trip_num=trip_num
                                )

                    loss  = img_rec_factor*dict_res['rec_imgs'] # image reconstruction
                    loss += pos_rec_factor*dict_res['rec_pose']  + trip_factor*dict_res['tri_pose'] # pose reconstruction taking spatial info into account
                    loss += crosss_factor*dict_res['crs_latent'] # embedding cross error, hoping it to be the same
                    loss += 2.0*img_rec_factor*dict_res['crec_imgs'] + pos_rec_factor*dict_res['crec_pose']

                # -- batch update network
                scaler.scale(loss).backward()
                
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm) 
                
                # scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # -- batch log info
                step += 1
                running_loss += loss.detach()*labels.size(0)

                if step % log_every_step == 0:
                    progress = float(idx)/len(train_loader)*100.0
                    pbar.set_description(f'Training: {epoch+1}/{num_epochs}, Progress: {progress:3.0f}%, Loss: {loss:6f}')
                    tb_writer.add_scalar('Loss/Batch/rec_imgs', dict_res['rec_imgs'], step)
                    tb_writer.add_scalar('Loss/Batch/rec_pose', dict_res['rec_pose'], step)
                    tb_writer.add_scalar('Loss/Batch/tri_pose', dict_res['tri_pose'], step)
                    tb_writer.add_scalar('Loss/Batch/crs_latent', dict_res['crs_latent'], step)
                    tb_writer.add_scalar('Loss/Batch/crec_imgs', dict_res['crec_imgs'], step)
                    tb_writer.add_scalar('Loss/Batch/crec_pose', dict_res['crec_pose'], step)

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
                    _, emb_imgs, _, emb_pose = model(images, labels)
                    # Cross embedding error
                    cross_latent_loss = F.mse_loss(emb_imgs, emb_pose).item()
                    # Cross reconstruction error
                    cross_rec_pose = model.pose_ae.decoder(emb_imgs)
                    cross_rec_imgs = model.imgs_ae.decoder(emb_pose)
                    crec_imgs_err = F.mse_loss(cross_rec_imgs, images).item()
                    crec_terr, crec_rerr = get_trans_rotation_err(
                                            pred=predict_postproc(cross_rec_pose.cpu().detach().numpy(),
                                                                offset=offset, normal=normal, balance=balance),
                                            target=poses.cpu().detach().numpy()
                                        )
                    val_loss = cross_latent_loss + crec_terr + crec_rerr + crec_imgs_err # score function to monitor the learning
                    running_val_loss += val_loss*labels.size(0)
                    # update progress bar
                    if step % log_every_step == 0:
                        progress = float(idx)/len(valid_loader)*100.0
                        pbar.set_description(f'Validation: {epoch+1}/{num_epochs}, Progress: {progress:3.0f}%, Loss: {val_loss:2f}')
                avg_val_loss = running_val_loss / len(valid_loader.dataset)

            if early_stopping(avg_val_loss):
                # Save the trained model
                torch.save(model.state_dict(), model_path)
                print(f"avg val loss: {avg_val_loss:2f}, Model saved!")

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

    # Evaluation - evaluate the best model not the model from the last epoch
    # model.load_state_dict(torch.load(last_model_path, weights_only=True))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    print("Model loaded!")
    print('-----------------------------------------')
    # print('Training results:')
    # test_model(model, train_loader, device, offset, normal, balance)
    # print('Validation results:')
    # test_model(model, valid_loader, device, offset, normal, balance)
    print('Testing results:')
    test_model(model, test_loader, device, offset, normal, balance)
    # print('Resting results:')
    # test_model(model, preserve_loader, device, offset, normal, balance)
    print('-----------------------------------------')

    print('Finished!')



if __name__ == "__main__":
    # import cProfile
    # import pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    training()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # stats.print_stats()