import torch
import torch.nn as nn

class Char_CNN(nn.Module):
    """
        一维CNN模型，适用于文本等1维数据，图片等需要使用二维卷积运算需要修改模型结构。结构:
        ---------------------------------------------------------------------------------------------
        Layer(type)                      ||        Kernel Shape         Output Shape         Param #
        =============================================================================================
        Char_CNN Inputs                  ||                   -          [1000, 300]               -
                                         ||                                                         
        01> Char_CNN-Cnn1-Conv1d         ||          [1, 24, 3]      [1000, 24, 300]              96
        02> Char_CNN-Cnn1-BatchNorm1d    ||                [24]      [1000, 24, 300]              48
        03> Char_CNN-Cnn1-ReLU           ||                   -      [1000, 24, 300]               0
        04> Char_CNN-Cnn1-MaxPool1d      ||                   -      [1000, 24, 150]               0
        05> Char_CNN-Cnn2-Conv1d         ||         [24, 48, 3]      [1000, 48, 150]           3,504
        06> Char_CNN-Cnn2-BatchNorm1d    ||                [48]      [1000, 48, 150]              96
        07> Char_CNN-Cnn2-ReLU           ||                   -      [1000, 48, 150]               0
        08> Char_CNN-Cnn2-MaxPool1d      ||                   -       [1000, 48, 75]               0
        09> Char_CNN-Linear1-Linear      ||         [3600, 128]          [1000, 128]         460,928
        10> Char_CNN-Linear1-Dropout     ||                   -          [1000, 128]               0
        11> Char_CNN-Linear1-BatchNorm1d ||               [128]          [1000, 128]             256
        12> Char_CNN-Linear1-ReLU        ||                   -          [1000, 128]               0
        13> Char_CNN-Linear2-Linear      ||            [128, 2]            [1000, 2]             258
        14> Char_CNN-Linear2-Dropout     ||                   -            [1000, 2]               0
        15> Char_CNN-Linear2-BatchNorm1d ||                 [2]            [1000, 2]               4
        16> Char_CNN-Linear2-Softmax     ||                   -            [1000, 2]               0
        =============================================================================================
    """
    
    def __init__(self,channel=1,feature_size=300):
        """
            Parameters:
            -------------
                channel: 深度、通道数
                feature_size: 特征向量大小
        """
        super(Char_CNN,self).__init__()
        
        self.feature_size = feature_size
        
        self.cnn1 = nn.Sequential(
            nn.Conv1d(channel,24,3,padding=1), 
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2) 
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv1d(24,48,3,padding=1,), 
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  
        )
        
        
        self.linear1 = nn.Sequential(
            nn.Linear(feature_size//4*48,128),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        self.linear2 = nn.Sequential(
            nn.Linear(128,2),
            nn.Dropout(0.3),
            nn.BatchNorm1d(2),
            nn.Softmax()
        )
        
        
    def forward(self,x):
        """
            x: (batch_size,feature_size),默认为channel为1
               (batch_size,channel,feature_size) ,channel要与初始化时一致    
        """
        # 二维向量要加入深度1再进行CNN
        if x.dim()==2:
            x = torch.unsqueeze(x,1)
        
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = x.view(-1,self.feature_size//4*48)
        x = self.linear1(x)
        x = self.linear2(x)
        return x