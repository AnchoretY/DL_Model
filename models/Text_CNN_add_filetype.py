import torch
import torch.nn as nn

class Text_CNN_add_filetype(nn.Module):
    """
        这个是Text_CNN针对webshell增加后缀名称的改进版本，在Text_CNN的基础上，增加了后缀名特征，直接增加到首个全连接层
        初始化参数
            param feature_size: 特征维度
            param filetype_nums: 文件类型个数
        前向传播
            param x1:要进行分类的样本
            param file_type:要进行分类的样本的url中请求的文件名
    """
    @property
    def model_name(self):
        return "Text_CNN_add_filetype"
    
    def __init__(self,filetype_nums,feature_size=300):
        
        # 下面注释的各个部分数据的结构都为feature_size为300时的
        super(Text_CNN_add_filetype,self).__init__()
        
        self.feature_size = feature_size
        
        self.layer1_1 = nn.Sequential(
            #第一层CNN
            nn.Conv1d(1,24,2,padding=1), #d*24*300
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  #d*24*150
            #第二层CNN
            nn.Conv1d(24,48,2,padding=1,), #d*48*150
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  #d*48*75
        )
        
        self.layer1_2  = nn.Sequential(
            nn.Conv1d(1,24,3,padding=1), #d*24*300
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  #d*24*150
            #第二层CNN
            nn.Conv1d(24,48,3,padding=1,), #d*48*150
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  #d*48*75
        )
        
        self.layer1_3  = nn.Sequential(
            nn.Conv1d(1,24,4,padding=1), #d*24*300
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  #d*24*150
            #第二层CNN
            nn.Conv1d(24,48,4,padding=2), #d*48*150   这里的padding要设置成2才能保证输出的维度和前两个CNN一样是75
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  #d*48*75
        )
        
        
        
        self.layer3 = nn.Sequential(
            nn.Linear(48*feature_size//4*3+filetype_nums,128),
#             nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(128,2),
#             nn.Dropout(0.3),
            nn.BatchNorm1d(2),
            nn.Softmax()
        )
        
        
    def forward(self,x,filetype):
        x = torch.unsqueeze(x,1)
        x1_1 = self.layer1_1(x)
        x1_2 = self.layer1_2(x)
        x1_3 = self.layer1_3(x)

        x = torch.cat((x1_1,x1_2,x1_3),dim=2)
        x = x.view(-1,48*self.feature_size//4*3)
        x = torch.cat((x,filetype),1)
        
        x = self.layer3(x)
        x = self.layer4(x)
        return x