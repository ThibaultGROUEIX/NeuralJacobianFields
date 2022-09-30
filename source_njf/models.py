import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    '''
    PointNet Encoder by Qi. et.al
    '''
    def __init__(self, zdim, input_dim=3, vae = False, normalization="BATCHNORM"):
        super(Encoder, self).__init__()
        
        self.zdim = zdim
        self.vae = vae
        self.normalization = normalization
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, zdim, 1)

        if self.normalization == "BATCHNORM":
            # print("Using BATCHNORM in pointnet!")
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(256)
            self.bn4 = nn.BatchNorm1d(zdim)
            # self.fc_bn1 = nn.BatchNorm1d(256)
            # self.fc_bn2 = nn.BatchNorm1d(128)

        elif self.normalization == "GROUPNORM":
            # print("Using GROUPNORM in pointnet!")
            self.bn1 = nn.GroupNorm(num_groups=4, num_channels=128)
            self.bn2 = nn.GroupNorm(num_groups=4, num_channels=128)
            self.bn3 = nn.GroupNorm(num_groups=8, num_channels=256)
            assert(zdim%50==0), "we need num_groups to be a dividor of zdim. Following code won't work with arbitrary values of zdim."
            self.bn4 = nn.GroupNorm(num_groups=8, num_channels=zdim)
            # self.fc_bn1 = nn.GroupNorm(num_groups=8, num_channels=256)
            # self.fc_bn2 = nn.GroupNorm(num_groups=4, num_channels=128)
        elif self.normalization == "IDENTITY":
            # print("Using IDENTITY (no normalization) in pointnet!")
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()
            self.bn4 = nn.Identity()
            # self.fc_bn1 = nn.Identity()
            # self.fc_bn2 = nn.Identity()
        else:
            assert(False)

        # self.fc1 = nn.Linear(zdim, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fcm = nn.Linear(128, zdim)
        
        # if self.vae:
        #     self.fcv = nn.Linear(128, zdim)

    def forward(self, x):
        '''
        Input: Nx#ptsx3
        Output: Nxzdim
        '''
        x = x.type_as(self.conv1.weight)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.zdim)
        return x
        # ms = F.relu(self.fc_bn1(self.fc1(x)))
        # ms = F.relu(self.fc_bn2(self.fc2(ms)))
        # mean = self.fcm(ms)
        # if self.vae:
        #     var = self.fcv(ms)
        #     return mean, var
        # return mean

