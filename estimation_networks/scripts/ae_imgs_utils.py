'''
    Image AutoEncoder
'''

import torch.nn as nn

# Encoder-Decoder with latent vector compression
class ImageAutoEncoder(nn.Module):
    def __init__(self, latent_dim=64, in_chs=1, img_size=256): #, dropout_rate=0.0):
        super(ImageAutoEncoder, self).__init__()

        fea_dim = int(img_size/16)
        
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chs, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (128, 128)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (64, 64)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (32, 32)

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # (16, 16)
            nn.Flatten(),
            
            nn.Linear(512 * fea_dim * fea_dim, latent_dim),
            
        )


        # Decoder part
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * fea_dim * fea_dim),
            
            nn.Unflatten(dim=1, unflattened_size=(512, fea_dim, fea_dim)),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, in_chs, kernel_size=2, stride=2),
        )
        
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
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z


import torch
if __name__ == '__main__':
    in_img = torch.randn((32, 3, 512, 512)).to('cpu')
    im_enc = ImageAutoEncoder(
        in_chs=int(3), latent_dim=64, img_size=512
    ).to('cpu')
    out_latent = im_enc(in_img)
    print('out latent size: ', out_latent[1].size())
    