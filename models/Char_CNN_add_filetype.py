import torch
import torch.nn as nn

class Char_CNN_add_filetype(nn.Module):
    """
        系统中原来使用的字符级别的CNN模型
        初始化模型
            param feature_size:输入特征维度
            param channel:输入数据管道数
            param filetype_nums:文件类型的个数
        前向传播
            param x:要进行分类的数据
            parma filetype:样本url中访问的文件后缀类型
    """
    @property
    def model_name(self):
        return "Char_CNN_add_filetype"
    
    
    def __init__(self,filetype_nums,channel=1,feature_size=300):
        
        # b,1,300
        super(Char_CNN_add_filetype,self).__init__()
        
        self.feature_size = feature_size
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(channel,24,3,padding=1), #d*300*24
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  #d*150*24
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(24,48,3,padding=1,), #d*150*48
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  #d*75*48
        )
        
        
        self.layer3 = nn.Sequential(
            nn.Linear(feature_size//4*48+filetype_nums,128),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(128,2),
            nn.Dropout(0.3),
            nn.BatchNorm1d(2),
            nn.Softmax()
        )
        
        
    def forward(self,x,filetype):
        x = torch.unsqueeze(x,1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1,self.feature_size//4*48)
        x = torch.cat((x,filetype),1)
        x = self.layer3(x)
        x = self.layer4(x)
        return x