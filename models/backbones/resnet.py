import torch.nn as nn
from torchvision.models import resnet18, resnet50

class CameraBackbone(nn.Module):
    def __init__(self, depth=50, pretrained=True):
        super().__init__()
        if depth == 18:
            backbone = resnet18(pretrained=pretrained)
        else:
            backbone = resnet50(pretrained=pretrained)
        
        # 去掉分类头 (fc) 和全局池化 (avgpool)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # 记录输出通道数 (ResNet50: 2048, ResNet18: 512)
        self.out_channels = 2048 if depth == 50 else 512

    def forward(self, x):
        """
        Input: [B, 3, H, W] (Images)
        Output: [B, C, H/32, W/32] (Feature Maps)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x