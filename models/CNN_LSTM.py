import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    """
        单卷积核大小的CNN来进行局部特征提取后，再交给LSTM提取全局信息
        初始化模型
            param feature_size:输入特征维度
            param channel：输入数据管道数
            param lstm_hidden_size:lstm隐藏层神经元个数
            param lstm_nums_layer:lstm层数
        
    """
    @property
    def model_name(self):
        return "CNN_LSTM"
    
    def __init__(self,feature_size=300,channel=1,lstm_hidden_size=128,lstm_nums_layer=2):
        # b,1,300
        super(CNN_LSTM,self).__init__()
        
        self.feature_size = feature_size
        self.lstm_hidden_size = lstm_hidden_size
        
        
        self.cnn = nn.Sequential(
            #第一层CNN
            nn.Conv1d(channel,24,3,padding=1), #d*24*300
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  #d*24*150
            #第二层CNN
            nn.Conv1d(24,48,3,padding=1), #d*48*150
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  #d*48*75
            
            #第三层CNN
            nn.Conv1d(48,64,3,padding=1), #d*48*75
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  #d*48*37
        )
        
        
        self.rnn = nn.LSTM(input_size=feature_size//8,hidden_size=lstm_hidden_size,num_layers=lstm_nums_layer,bidirectional=True)
        
        
        
        self.linear1 = nn.Sequential(
            nn.Linear(64*lstm_hidden_size*2,128),
            nn.Dropout(0.3),
        )
        
        self.linear2 = nn.Linear(128,2) 
        
        
    def forward(self,x):
        batch_size = x.size(0)
        #添加一个一维 [b,300]->[b,1,300]
        x = torch.unsqueeze(x,1)
        #经过三层cnn转换为 [b,1,300]->[b,64,37]
        x = self.cnn(x)
        #cnn要求深度在前  转化成cnn可用的格式[64,b,37]
        x = x.permute(1, 0, 2)
        #经过bi-lstm转化成[64,b,lstm_hidden_size*2]
        x,_ = self.rnn(x)
        #恢复正常顺序
        x = x.permute(1, 0, 2)
        #使用完permute要进行view前要进行contiguous操作
        x = x.contiguous()
        
        #拉深层[b,64,lstm_hidden_size*2]->[b,64*lstm_hidden_size*2]
        x = x.view(batch_size,-1)
        #全连接+dropout层
        x = self.linear1(x)
        #softmax
        import torch.nn.functional as F
        
        x = F.softmax(self.linear2(x))
        return x