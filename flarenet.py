import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from cifar10_models.densenet import densenet121
import torch.nn.functional as F
from common import*

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class flarenet(nn.Module):
    def __init__(self, num_classes=2):
        super(flarenet, self).__init__()
        # Load the pre-trained DenseNet121 model
        original_model = densenet121(pretrained=False)
        self.features = original_model.features
        self.gradients = None
        self.spp = SPP(1024, 1024)  
        self.contextual_fpn = ContextualFPN(inplanes=1024, outplanes=1024, dilat=[1, 2, 3, 4], se=True)  # Example dimensions
        self.channel_attention = ChannelAttention(in_channels=1024)  # Adjust channels as needed
        self.spatial_attention = SpatialAttention()
        # Final classification layer
        self.classifier = nn.Linear(1024, num_classes)  # Adjust channels as needed
    def activations_hook(self, grad):
        self.gradients = grad
    def forward(self, x):
        with torch.enable_grad():
            features = self.features(x)
            # print('features:', features.shape)
            # Apply SPP and Contextual FPN
            spp_features = self.spp(features)
            # print('spp_features:', spp_features.shape)
            fpn_features = self.contextual_fpn(spp_features)
            # print('fpn_features:', fpn_features.shape)
            # Apply attention mechanisms
            ch_att = self.channel_attention(fpn_features)
            sp_att = self.spatial_attention(fpn_features)
            features = fpn_features * ch_att * sp_att
            # print('features:', features.shape)
            features.register_hook(self.activations_hook)
            # Pooling and classification
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = self.classifier(out)
            return out
    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features(x)
