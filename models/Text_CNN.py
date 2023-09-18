import torch

import torch.nn.functional as fn
import torch.nn as nn

class Text_CNN(nn.Module):
    """
        这里是三条不同大小的
        初始化参数
            param channel:输入数据管道数
            param feature_size: 特征维度
        前向传播
            param x:输入数据，要进行判别的样本
            
    """
    
    @property
    def model_name(self):
        return "Text_CNN"
    
    def __init__(self,channel=1,feature_size=300):
        
        # 下面注释的各个部分数据的结构都为feature_size为300时的
        super(Text_CNN,self).__init__()
        
        self.feature_size = feature_size
        
        self.cnn1_1 = nn.Sequential(
            #第一层CNN
            nn.Conv1d(channel,24,2,padding=1), #d*24*300
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  #d*24*150
            #第二层CNN
            nn.Conv1d(24,48,2,padding=1,), #d*48*150
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  #d*48*75
        )
        
        self.cnn1_2  = nn.Sequential(
            nn.Conv1d(channel,24,3,padding=1), #d*24*300
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  #d*24*150
            #第二层CNN
            nn.Conv1d(24,48,3,padding=1,), #d*48*150
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  #d*48*75
        )
        
        self.cnn1_3  = nn.Sequential(
            nn.Conv1d(channel,24,4,padding=1), #d*24*300
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  #d*24*150
            #第二层CNN
            nn.Conv1d(24,48,4,padding=2), #d*48*150   这里的padding要设置成2才能保证输出的维度和前两个CNN一样是75
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  #d*48*75
        )
        
        
        
        self.linear1 = nn.Sequential(
            nn.Linear(48*feature_size//4*3,128),
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
        
        x = torch.unsqueeze(x,1)
        x1_1 = self.cnn1_1(x)
        x1_2 = self.cnn1_2(x)
        x1_3 = self.cnn1_3(x)

        x = torch.cat((x1_1,x1_2,x1_3),dim=2)
        x = x.view(-1,48*self.feature_size//4*3)
        
        x = self.linear1(x)
        x = self.linear2(x)
        return x