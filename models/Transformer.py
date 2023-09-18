import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
        标准的scaled点乘attention层
    """

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播.
        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        

        if scale:
            attention = attention * scale
            
        
        if attn_mask is not None:
            # 给需要 mask 的地方设置一个负无穷
            attention = attention.masked_fill(attn_mask,-1e9)
        
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)

        return context, attention
    
    
class MultiHeadAttention(nn.Module):
    """
        多头Attention层
    """

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        
        # 残差连接
        residual = query
        
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)
        

        # 线性层 (batch_size,word_nums,model_dim)
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # 将一个切分成多个(batch_size*num_headers,word_nums,word//num_headers)
        """
            这里用到了一个trick就是将一个word_nums分成多层的过程中
        """
        
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)
        
        
        #将mask也复制多份和key、value、query相匹配  （batch_size*num_headers,word_nums_k,word_nums_q）
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        
        # 使用scaled-dot attention来进行向量表达
        #context:(batch_size*num_headers,word_nums,word//num_headers)
        #attention:(batch_size*num_headers,word_nums_k,word_nums_q)
        scale = (key.size(-1)) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale, attn_mask)
        
        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention
    
def padding_mask(seq_k, seq_q):
    """
        padding mask,主要是为了完全去除掉padding的0对训练结果产生的影响，具体做法是把这些位置加上一个非常大的负数，
    然后经过softmax层以后就会无线趋近于0
        
        param seq_q:(batch_size,word_nums_q)
        param seq_k:(batch_size,word_nums_k)
        return padding_mask:(batch_size,word_nums_q,word_nums_k)
    """
    
    # seq_k和seq_q 的形状都是 (batch_size,word_nums_k)
    len_q = seq_q.size(1)
    # 找到被pad填充为0的位置(batch_size,word_nums_k)
    pad_mask = seq_k.eq(0)
    #(batch_size,word_nums_q,word_nums_k)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    
    return pad_mask


class PositionalEncoding(nn.Module):
    
    """
        位置编码层
    """
    
    def __init__(self, d_model, max_seq_len):
        """初始化。
        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        
        
        
        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
          [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
          for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.Tensor(position_encoding)

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))
        
        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)
    def forward(self, input_len,max_len):
        """
            神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。
          param 

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """
        # 找出这一批序列的最大长度
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        
        #print([list(range(1, len + 1)) + [0]*(max_len.item() - len) for len in input_len.tolist()])
        input_pos = tensor(
          [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len.tolist()])
        
        return self.position_encoding(input_pos)
    
    
class PositionalWiseFeedForward(nn.Module):
    """
        前向编码，使用两层一维卷积层实现
    """

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output
    
    
class EncoderLayer(nn.Module):
    """Encoder的一层。"""

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        # feed forward network
        output = self.feed_forward(context)

        return output, attention

class Encoder(nn.Module):
    """
        多层EncoderLayer组成Encoder，用于将特征作为带attention的特征表示
        
    """

    def __init__(self,
               vocab_size,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):
        """
            param vacab_size: 输入词的长度
            param num_layers: encoder层中encoder_layer子层循环的次数，默认为6
            param model_dim: 整个模型过程中向量化使用这个统一的维度
            param num_header: 多头attentiion子层的头数
            param ffn_dim: ffn前馈神经网络子层的中间层维度
            param dropout: dropout 比例
            
        """
        
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

#         self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.seq_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, vocab_size)

    def forward(self, inputs,inputs_len):
        """
            
            param inputs:要进行encoder的原始张量 （张量） （Batch_size,src_vacab_size)
            param inputs_len:输入长度列表 （张量） （Batch_size,1）
            
            return output: Encoder后的向量 （张量） （Batch_size,src_vacab_size,model_dim）
            return attentions: 各个子层的attention列表 (列表) [(Batch_size,src_vacab_size,src_vacab_size)*header]
            
        """
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len,300)
        
        
        #获得self-attention之中的pad-mask(batch_size,word_nums_k,word_nums_q) (7,300,300)
        self_attention_mask = padding_mask(inputs, inputs)
        
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions
    
    
class Transformer(nn.Module):
    """
        Transformer模型
    """
    @property
    def model_name(self):
        return "Transformer"

    def __init__(self,
               src_vocab_size,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.2):
        """
            param src_vocab_size:输入词的长度
            param num_layer:encoder层中encoder_layer子层循环的次数，默认为6
            param model_dim: 整个模型过程中向量化使用这个统一的维度
            param num_head:多头attentiion子层的头数
            param ffn_dim: ffn前馈神经网络子层的中间层维度
            param dropout:dropout 比例
            
        """
        super(Transformer, self).__init__()
        
        self.src_vocab_size = src_vocab_size
        self.model_dim = model_dim

        self.encoder = Encoder(src_vocab_size, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)

        self.linear = nn.Linear(src_vocab_size*model_dim, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, src_seq, src_len):
        """
            param src_seq:要进入模型已经进行编码的向量矩阵，包含零填充 （张量） （Batch_size,src_vocab_size）
            param src_len:每条数据未填充前的长度 (张量) （Batch_size,1）
            
            return output:每条数据经过transformer模型判别属于两类的概率 (张量) （Batch_size,2）
            return enc_self_attn:各层的注意力列表 （列表） [(Batch_size,)]
        """
        output, enc_self_attn = self.encoder(src_seq, src_len)

        #这里的处理方式主要有三种，但是这里只两种情况适用
        # 1.将后两维整体展开，然后接全连接
        # 2.取第三维的均值，然后连全连接
        # 3.去最后一个词的向量表示，然后接全连接层，这种在RNN中可用，在这里用不太好
        
        output = output.view(-1,self.src_vocab_size*self.model_dim)
        output = self.linear(output)
        
        output = self.softmax(output)

        return output, enc_self_attn