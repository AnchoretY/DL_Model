import torch
import torch.nn as nn



class Char_Embedding_RNN(nn.Module):
    @property
    def model_name(self):
        return "Char_Embedding_RNN"
    
    
    def __init__(self,input_size=300,embeding_dim=50,rnn_hidden_size=128,hidden_layer_nums=2,model_type='lstm',bidirectional=False):
        
        super(Char_Embedding_RNN,self).__init__()
        
        self.char_embedding = nn.Embedding(input_size,embeding_dim)
        
        if model_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size=embeding_dim,hidden_size=rnn_hidden_size,num_layers=hidden_layer_nums,batch_first=True,bidirectional=bidirectional)
        elif model_type.lower() =='gru':
            self.rnn = nn.GRU(input_size=embeding_dim,hidden_size=rnn_hidden_size,num_layers=hidden_layer_nums,batch_first=True,bidirectional=bidirectional)
        else:
            raise "ValueError:model_type only can input lstm/gru"
        
        if bidirectional == True:
            self.linear1 = nn.Linear1(rnn_hidden_size*2,64)
        else:
            self.linear1 = nn.Linear(rnn_hidden_size,64)
        
        self.linear2 = nn.Linear(64,2)
        
    def forward(self,x):
        x = self.char_embedding(x)
        x,_ = self.rnn(x)
        x = self.linear1(x)
        x = self.linear2(x)[:,-1,:]
        return x