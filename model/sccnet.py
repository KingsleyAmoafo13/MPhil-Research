r""" SCCNetwork Implementation"""
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg

from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner

class SCCNetwork(nn.Module):
    def __init__(self, backbone, use_original_imgsize, freeze=True):
        super(SCCNetwork, self).__init__()

        # Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            loaded_model = vgg.vgg16()
            checkpoint = torch.load('C:/Users/Aking/Desktop/bestmodels/New folder/vgg16.pth')
            loaded_model.load_state_dict(checkpoint, strict=False)
            self.backbone = loaded_model
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]  # Feature IDs for VGG
            self.extract_feats = extract_feat_vgg
            self.bottleneck_ids = [2, 2, 3, 3, 3, 1]  # Example bottleneck IDs for VGG
            self.lids = [0, 1, 2]  # Example Layer IDs for VGG
        elif backbone == 'resnet50':
            loaded_model = resnet.resnet50()
            checkpoint = torch.load('C:/Users/Aking/Desktop/MPHIL_RESEARCH_3/best_models/Pruned_Resnet50.pth')
            loaded_model.load_state_dict(checkpoint, strict=True)
            self.backbone = loaded_model
            self.feat_ids = list(range(4, 17))  # Feature IDs for ResNet50
            self.extract_feats = extract_feat_res
            self.bottleneck_ids = [3, 4, 6, 3]  # Example bottleneck IDs for ResNet50
            self.lids = [0, 1, 2, 3]  # Example Layer IDs for ResNet50
        elif backbone == 'resnet101':
            loaded_model = resnet.resnet101()
            checkpoint = torch.load('C:/Users/Aking/Desktop/MPHIL_RESEARCH_3/best_models/Pruned_Resnet101.pth')
            loaded_model.load_state_dict(checkpoint, strict=False)
            self.backbone = loaded_model
            self.feat_ids = list(range(4, 34))  # Feature IDs for ResNet101
            self.extract_feats = extract_feat_res
            self.bottleneck_ids = [3, 4, 23, 3]  # Example bottleneck IDs for ResNet101
            self.lids = [0, 1, 2, 3, 4]  # Example Layer IDs for ResNet101
        else:
            raise Exception(f'Unavailable backbone: {backbone}')

        # Freeze backbone if requested
        if freeze:
            self.backbone.eval()

        # Initialize HPNLearner modules
        self.hpn_learner = HPNLearner([3, 4, 6])
        self.hpn_learner2 = HPNLearner([3, 4, 6])

        # Feature Adjustment Modules (FAM)
        self.fam1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.fam2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Merger for final logits
        self.merger = nn.Sequential(
            nn.Conv2d(4, 2, (1, 1), bias=False),
            nn.ReLU()
        )

    def mask_feature(self, features, mask):
        """Applies the support mask to each feature in the list of support features."""
        # Check if features is a list and process each feature map individually
        if isinstance(features, list):
            masked_features = []
            for feat in features:
                # Ensure that the mask is a float type for multiplication
                mask_float = mask.float()  # Convert mask to float
                
                # Apply mask to the individual feature
                masked_feat = feat * mask_float.unsqueeze(1)  # Unsqueeze to match feature dimensions
                
                # Append the masked feature to the list
                masked_features.append(masked_feat)
            
            return masked_features  # Return the list of masked features

        # If features is not a list, handle it as a single tensor
        print(f"features shape: {features.shape}, features dtype: {features.dtype}")
        print(f"mask shape: {mask.shape}, mask dtype: {mask.dtype}")
        
        mask = mask.float()  # Convert mask to float
        print(f"mask shape after conversion: {mask.shape}, mask dtype after conversion: {mask.dtype}")

        return features * mask.unsqueeze(1)  # Unsqueeze to match feature dimensions

    def forward(self, query_img, support_img, support_mask, query_mask=None):
        # Extract features from the query and support images
        query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

        # Apply the support mask to support features
        support_feats = self.mask_feature(support_feats, support_mask.clone())

        # Cross-correlation
        corr = Correlation.multilayer_correlation(query_feats, support_feats)

        # Feature adjustment with the first FAM
        corr = self.fam1(corr)

        logit_mask_ori = self.hpn_learner(corr)
        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask_ori, support_img.size()[2:], mode='bilinear', align_corners=True)

        pred_mask = logit_mask.argmax(dim=1)

        # Mask the query features using the predicted mask
        masked_qfeats = self.mask_feature(query_feats, pred_mask)

        # Cross-correlation for the second pass
        corr2 = Correlation.multilayer_correlation(query_feats, masked_qfeats)

        # Feature adjustment with the second FAM
        corr2 = self.fam2(corr2)

        logit_mask2 = self.hpn_learner2(corr2)

        # Merge logits from both passes
        logit = torch.cat([logit_mask_ori, logit_mask2], dim=1)
        logit = self.merger(logit)

        # Resize the final prediction to the original image size
        if not self.use_original_imgsize:
            logit = F.interpolate(logit, support_img.size()[2:], mode='bilinear', align_corners=True)

        return logit
