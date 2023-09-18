from torch import nn

class BiLSTM(nn.Module):
    """
        这是一个双向LSTM网络
        初始化参数
            param feature_size: 特征维度
            param filetype_nums: 文件类型个数
        前向传播
            param x1:要进行分类的样本
            param file_type:要进行分类的样本的url中请求的文件名
    """
    @property
    def model_name(self):
        return "BiLSTM_wy"

    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=128, lstm_layer=2, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * lstm_layer * 2, 1)

    def forward(self, sents):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        y = self.hidden2label(self.dropout(torch.cat([c_n[i, :, :] for i in range(c_n.shape[0])], dim=1)))
        return y