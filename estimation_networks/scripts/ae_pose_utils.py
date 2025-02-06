'''
    Lifting the embedded pose feature and then decode to the original dimension
'''


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Encoder-Decoder with latent vector expansion
class PoseAutoEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden_dims=[128, 256], latent_dim=64, batchnorm=False, normalize=False):
        super(PoseAutoEncoder, self).__init__()
        if batchnorm: 
            encoder_list = [
                nn.Linear(in_dim, hidden_dims[0]),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.BatchNorm1d(hidden_dims[1]),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dims[1], latent_dim),
            ]
            decoder_list = [
                nn.Linear(latent_dim, hidden_dims[1]),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dims[1], hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dims[0], in_dim),
            ]
        else:
            encoder_list = [
                nn.Linear(in_dim, hidden_dims[0]),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dims[1], latent_dim),
            ]
            decoder_list = [
                nn.Linear(latent_dim, hidden_dims[1]),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dims[1], hidden_dims[0]),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dims[0], in_dim),
            ]
        
        
        # Encoder
        self.encoder = nn.Sequential(*encoder_list)
        
        # Decoder
        if normalize: decoder_list.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_list)
        
        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
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
    
    def forward(self, x):
        z = self.encoder(x) # z is the embedded vector
        x = self.decoder(z)
        return x, z
    
def distance_metric(pose_a:torch.Tensor, pose_b:torch.Tensor, balance=True):
    if balance:
        return torch.dist(pose_a, pose_b, p=2) 
    temp_a = pose_a.clone().detach() # torch.tensor(pose_a)
    temp_a[:3] *= 15.0
    temp_b = pose_b.clone().detach() # torch.tensor(pose_b)
    temp_b[:3] *= 15.0
    return torch.dist(temp_a, temp_b, p=2) 

def get_combinations(batch_size:int, num_com:int=3):
    assert num_com<batch_size, 'Not enough samples for combinations!'
    sample_indices = torch.tensor(range(batch_size), dtype=int)
    return torch.combinations(sample_indices, r=num_com)

def get_triplet_indices(sample_comlist, labels, num_samples=None, balance=True):
    '''
        labels decides whether the sample is anchor, positive or negative
        return a list of triplet_indices indicating anchor, positive, negative 
    '''
    if num_samples is None or num_samples>len(sample_comlist):
        num_samples = len(labels)
    tmp_indices = torch.randint(len(sample_comlist), (num_samples,))  # Num_triplet_samples x 3
    tri_indices = list() # updates the tmp_indices in the format of num_samplex x (anchor, positive, negative)_indices
    for idxes in sample_comlist[tmp_indices]:
        # randomly choose an anchor
        idxes = idxes.numpy()
        anchor_idx = torch.randint(3, (1,)).item()
        an_idx = idxes[anchor_idx]
        aa_idx = idxes[(anchor_idx+1)%3]
        bb_idx = idxes[(anchor_idx+2)%3]
        if an_idx >= len(labels) \
            or aa_idx >= len(labels)\
                or bb_idx >= len(labels):
            continue
        anchor = labels[an_idx]
        tmp_aa = labels[aa_idx]
        tmp_bb = labels[bb_idx]
        dis_aa = distance_metric(anchor, tmp_aa, balance)
        dis_bb = distance_metric(anchor, tmp_bb, balance)
        
        if abs(dis_aa-dis_bb) < 0.01: 
            # print(f'too close!!! {dis_aa-dis_bb}, {pn_dis}')
            continue # to avoid learning samples that are too close to each other 
        
        if dis_aa < dis_bb: 
            tri_indices.append([an_idx, aa_idx, bb_idx])
        else:
            tri_indices.append([an_idx, bb_idx, aa_idx])
    return torch.as_tensor(np.array(tri_indices))  
    
def calc_triplet_loss(embeddings, tri_indices, inputs=None, margin=0.2):
    anchors   = embeddings[tri_indices[:, 0]]
    positives = embeddings[tri_indices[:, 1]]
    negatives = embeddings[tri_indices[:, 2]]
    positive_distance = F.pairwise_distance(anchors, positives, p=2)  # positive distance
    negative_distance = F.pairwise_distance(anchors, negatives, p=2)  # negatie distance

    if inputs is not None: 
        anchors_in   = inputs[tri_indices[:, 0]]
        negatives_in = inputs[tri_indices[:, 2]]
        margin = F.pairwise_distance(anchors_in, negatives_in, p=2)*margin

    loss = torch.relu(positive_distance - negative_distance + margin)

    return loss.mean()

