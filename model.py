import torch
import torch.nn as nn

class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class PartSegmentation(nn.Module):
    def __init__(self, in_channels, part_dict):
        super(PartSegmentation, self).__init__()
        self.part_dict = part_dict
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Count similarity between feature vectors and part dictionary
        # x's shape: (batch_size, D, H, W)
        B, D, H, W = x.size()
        part_dict = self.part_dict.unsqueeze(0).expand(B, -1, -1)  # 形状变为 (B, D, K)
        
        # x's shape: (B, D, H * W)
        x = x.view(B, D, H * W)
        
        # similarity shape: (B, K, H * W)
        similarity = torch.bmm(part_dict.permute(0, 2, 1), x)
        
        # reshape into: (B, K, H, W)
        similarity = similarity.view(B, -1, H, W)
        
        # use softmax
        part_assignment = self.softmax(similarity)
        return part_assignment

class RegionFeatureExtraction(nn.Module):
    def __init__(self, part_dict_size, in_channels):
        super(RegionFeatureExtraction, self).__init__()
        self.part_dict_size = part_dict_size
        self.attention_conv = nn.Conv2d(part_dict_size, 1, kernel_size=1)

    def forward(self, x, part_assignment):
        batch_size = x.size(0)
        c = x.size(1)
        h, w = x.size(2), x.size(3)
        
        part_attention = self.attention_conv(part_assignment)
        part_attention = torch.sigmoid(part_attention)  # shape: [batch_size, 1, h, w]
        
        part_features = []
        for i in range(self.part_dict_size):
            # get weight
            part_attention_weight = part_attention * part_assignment[:, i, :, :].unsqueeze(1)  # 形状为 [batch_size, 1, h, w]
            
            # get weighted feature
            part_feature = part_attention_weight * x  # [batch_size, in_channels, h, w]
            part_feature = part_feature.view(batch_size, c, -1).sum(dim=2)  # [batch_size, in_channels]
            part_features.append(part_feature)

        part_features = torch.stack(part_features, dim=1)  # [batch_size, part_dict_size, in_channels]
        
        # final feature
        part_features = part_features.view(batch_size, -1)

        return part_features

class FineGrainedModel(nn.Module):
    def __init__(self, num_classes=10, part_dict_size=8):
        super(FineGrainedModel, self).__init__()
        self.conv1 = BasicConvBlock(3, 64)
        self.conv2 = BasicConvBlock(64, 128)
        self.conv3 = BasicConvBlock(128, 256)

        # Initialize part dictionary
        self.part_dict = nn.Parameter(torch.randn(256, part_dict_size))  # D = 256, K = part_dict_size
        
        self.part_segmentation = PartSegmentation(256, self.part_dict)
        self.region_feature_extraction = RegionFeatureExtraction(part_dict_size, 256)
        self.fc = nn.Linear(256 * part_dict_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) # [bathc_size, 256, h, w]
        part_assignment = self.part_segmentation(x)# [batch_size, 8, h, w]
        part_features = self.region_feature_extraction(x, part_assignment)
        out = self.fc(part_features)
        return out
