import torch
import torch.nn as nn
import torch.nn.functional as F


class WordAttention(nn.Module): 
    
    @property
    def model_name(self):
        return "WordAttention"
        
    
    def __init__(self, vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size, dropout): 
        """ 
            :param vocab_size: number of words in the vocabulary of the model 
            :param emb_size: size of word embeddings 
            :param word_rnn_size: size of (bidirectional) word-level RNN 
            :param word_rnn_layers: number of layers in word-level RNN 
            :param word_att_size: size of word-level attention layer 
            :param dropout: dropout 
        """ 
        
        super(WordAttention, self).__init__() 

        self.embeddings = nn.Embedding(vocab_size, emb_size) 
        self.word_rnn = nn.GRU(emb_size, word_rnn_size, num_layers=word_rnn_layers, bidirectional=True, dropout=dropout, batch_first=True)  
        self.word_attention = nn.Linear(2 * word_rnn_size, word_att_size) 
        self.word_context_vector = nn.Linear(word_att_size, 1, bias=False) 
        self.dropout = nn.Dropout(dropout) 
        self.linear1 = nn.Linear(2*word_rnn_size,64)
        self.linear2 = nn.Linear(64,2)
        
        
        
    def forward(self, sentences, words_per_sentence): 
        """ 
            Forward propagation. 
            :param sentences: encoded sentence-level data, a tensor of dimension (n_sentences, word_pad_len, emb_size) 
            :param words_per_sentence: sentence lengths, a tensor of dimension (n_sentences) 
            :return: sentence embeddings, attention weights of words 
        """ 
        # 把各个句子按照长短进行排序 (n_sentences, word_pad_len) 
        words_per_sentence, sent_sort_ind = words_per_sentence.sort(dim=0, descending=True) 
        sentences = sentences[sent_sort_ind] 
        
        
        # 进行embeddings，然后在进行dropout  (n_sentences, word_pad_len, emb_size) 
        sentences = self.embeddings(sentences)
        sentences = self.dropout(sentences) 
        
        

 
        # 将各个句子转成到一起,前两维非填充的部分来进行拼接(SENTENCES -> WORDS)    (words_sum, emb_size),(no_pading_len_list)
        words, bw = nn.utils.rnn.pack_padded_sequence(sentences, lengths=words_per_sentence.tolist(), batch_first=True) 
        #print(nn.utils.rnn.PackedSequence(words, bw))
        
        
        
        # (n_words, emb_size), 得到各个词语的RNN表示法
        (words, _), _ = self.word_rnn(nn.utils.rnn.PackedSequence(words, bw)) 
        # (n_words, 2 * word_rnn_size), (max(sent_lens)) 

        
        
        # 获得RNN表示法的隐含表示(n_words, att_size) 
        att_w = self.word_attention(words) 
        att_w = F.tanh(att_w) 


        # 取注意力向量和上下文向量(也就是这里的线性层参数向量)的点乘 
        att_w = self.word_context_vector(att_w).squeeze(1) 
        
        # ==============================获得归一化的权重矩阵==================================
        max_value = att_w.max() 
        att_w = torch.exp(att_w - max_value)   #归一化
        att_w, _ = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(att_w, bw), batch_first=True)

        # (n_sentences, max_sent_len_in_batch) 
        # Calculate softmax values (n_sentences, max_sent_len_in_batch) 
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True) 

        
        
        #==========================根据权重矩阵去进行重新计算当前句子的向量表示=======================
        #这里感觉可以直接将相乘的结果过全连接层输出分类
        
        
        # Similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES) (n_sentences, max_sent_len_in_batch, 2 * word_rnn_size) 
        sentences, _ = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(words, bw), batch_first=True)
        
        # Find sentence embeddings 
        sentences = sentences * (word_alphas.unsqueeze(2))

        # (n_sentences, max_sent_len_in_batch, 2 * word_rnn_size) 
        sentences = sentences.sum(dim=1) # (n_sentences, 2 * word_rnn_size) 
        

        
        # Unsort sentences into the original order (INVERSE OF SORTING #2) 
        _, sent_unsort_ind = sent_sort_ind.sort(dim=0, descending=False) 
        # (n_sentences) 
        sentences = sentences[sent_unsort_ind] 
        # (n_sentences, 2 * word_rnn_size) 
        word_alphas = word_alphas[sent_unsort_ind] 
        # (n_sentences, max_sent_len_in_batch) 
        
        #=========================== 这里是新加的全连接层==================
        
        
        output = self.linear1(sentences)
        output = self.linear2(output)
        
        
        return output, word_alphas 