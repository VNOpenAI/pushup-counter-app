import torch.nn as nn

class ResNeSt_head(nn.Module):
    def __init__(self, pre_model):
        super(ResNeSt_head, self).__init__()
        # pre_model.fc = nn.Identity()
        # pre_model.avgpool = nn.Identity()
        # pre_model.layer4 = nn.Identity()
        # pre_model.layer3 = nn.Identity()
        # pre_model.layer2 = nn.Identity()
        self.pre_model = pre_model
        self.last_conv = nn.Conv2d(1024, 21, (1,1), 1)
        self.output = nn.Sigmoid()
    def forward(self, x):
        x = self.pre_model.conv1(x)
        x = self.pre_model.bn1(x)
        x = self.pre_model.relu(x)
        x = self.pre_model.maxpool(x)
        x = self.pre_model.layer1(x)
        x = self.pre_model.layer2(x)
        x = self.pre_model.layer3(x)
        x = self.last_conv(x)
        x = self.output(x)
        return x