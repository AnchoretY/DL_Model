import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual Block
    """
 
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        """
            Args:
                inchannel:输入channel
                outchannel: 出处channel
                stride: 残差块中第一个CNN的步幅，默认为1,这是不改变输出的shape
                shortcut: shortcut路径使用的模块，默认为None，表示将输入原封不动的进行输出；另一种为1*1的shortcut，用于保证shortcut路径输出与保持一致
        """
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
            )
        self.right = shortcut
 
    def forward(self, x):
        out = self.left(x)
        # 两种shortcut路径：如果shortcut为None，那么直接将模型的输入作为残差加入到三层CNN的输出，否则使用shortcut对输入进行转换后加到三层CNN的输出作为残差
        residual = x if self.right is None else self.right(x)   
        out += residual
        return F.relu(out)
    

class ResNet11(nn.Module):
    """
        ResNet11,由全连接+三个残差块（3*3）+全连接层组成，order matter函数相似性识别中使用
        ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
        ================================================================
            Conv2d-1         [10, 64, 128, 128]             576
       BatchNorm2d-2         [10, 64, 128, 128]             128
              ReLU-3         [10, 64, 128, 128]               0
            Conv2d-4         [10, 64, 128, 128]          36,864
       BatchNorm2d-5         [10, 64, 128, 128]             128
              ReLU-6         [10, 64, 128, 128]               0
            Conv2d-7         [10, 64, 128, 128]          36,864
       BatchNorm2d-8         [10, 64, 128, 128]             128
              ReLU-9         [10, 64, 128, 128]               0
           Conv2d-10         [10, 64, 128, 128]          36,864
      BatchNorm2d-11         [10, 64, 128, 128]             128
           Conv2d-12         [10, 64, 128, 128]           4,096
      BatchNorm2d-13         [10, 64, 128, 128]             128
             ReLU-14         [10, 64, 128, 128]               0
    ResidualBlock-15         [10, 64, 128, 128]               0
           Conv2d-16          [10, 128, 64, 64]          73,728
      BatchNorm2d-17          [10, 128, 64, 64]             256
             ReLU-18          [10, 128, 64, 64]               0
           Conv2d-19          [10, 128, 64, 64]         147,456
      BatchNorm2d-20          [10, 128, 64, 64]             256
             ReLU-21          [10, 128, 64, 64]               0
           Conv2d-22          [10, 128, 64, 64]         147,456
      BatchNorm2d-23          [10, 128, 64, 64]             256
           Conv2d-24          [10, 128, 64, 64]           8,192
      BatchNorm2d-25          [10, 128, 64, 64]             256
             ReLU-26          [10, 128, 64, 64]               0
    ResidualBlock-27          [10, 128, 64, 64]               0
           Conv2d-28          [10, 256, 32, 32]         294,912
      BatchNorm2d-29          [10, 256, 32, 32]             512
             ReLU-30          [10, 256, 32, 32]               0
           Conv2d-31          [10, 256, 32, 32]         589,824
      BatchNorm2d-32          [10, 256, 32, 32]             512
             ReLU-33          [10, 256, 32, 32]               0
           Conv2d-34          [10, 256, 32, 32]         589,824
      BatchNorm2d-35          [10, 256, 32, 32]             512
           Conv2d-36          [10, 256, 32, 32]          32,768
      BatchNorm2d-37          [10, 256, 32, 32]             512
             ReLU-38          [10, 256, 32, 32]               0
    ResidualBlock-39          [10, 256, 32, 32]               0
AdaptiveMaxPool2d-40            [10, 256, 1, 1]               0
           Linear-41                  [10, 128]          32,896
    """
 
    def __init__(self, embedding_size=128):
        super(ResNet11, self).__init__()
        self.model_name = 'resnet11'
 
        # 输入层
        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
            
        # 三层残差块（3*3 CNN）
        self.layer1 = self._make_layer(64,64) # 第一个block stride为1，数据维度不变
        self.layer2 = self._make_layer(64,128,stride=2) 
        self.layer3 = self._make_layer(128,256,stride=2)
        # 最大池化
        self.max_pool2d = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        # 图embedding转化，调整维度
        self.fc = nn.Linear(256,embedding_size)
        
    def _make_layer(self, inchannel, outchannel, stride=1):
        """
        构建layer,主要用于将残差调整到与经过block层的数据维度一致进行拼接输出
        Args:
            inchannel：输入channel
            outchannel：输出channel
            stride：残差block中第一个CNN的步幅
        """
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )
        
        return ResidualBlock(inchannel, outchannel, stride, shortcut)
    
    # input:(batch,1,w,h)
    def forward(self, x):
        x = self.pre(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.max_pool2d(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

