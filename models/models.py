from mxnet import nd
from mxnet.gluon import Block
from mxnet.gluon import nn
from mxnet import nd, autograd
from mxnet.gluon import Trainer
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from .vocab import Vocab
from .encoders import RNNEncoder, ParseEncoder 
from .decoders import BaseDecoder, RNNDecoder, HeadDecoder
from mxnet import cpu
import os
import pickle
from tqdm import tqdm
from mxboard import SummaryWriter


class BaseModel(Block):

    """Docstring for BaseModel. """

    def __init__(self, vocab, model_params, ctx=cpu(0)):
        """TODO: to be defined.

        :vocab: TODO
        :vocab_tag: TODO
        :model_params: TODO
        :ctx: TODO

        """
        Block.__init__(self)

        self._vocab = vocab
        self._model_params = model_params
        self._ctx = ctx

    def forward(self):
        """TODO: Docstring for forward.
        :returns: TODO

        """
        raise NotImplementedError('BaseModel is a abstract class, you must implemention forward()')
        

class ParseModel(BaseModel):
    """

    BaseLine 模型，使用论文Encoder，普通LSTMDecoder，使用Sentence级Attention，无Pointer及Coverage

    """
    def __init__(self, vocab, vocab_tag, model_params,ctx=cpu(0)):
        """

        初始化模型

        :param vocab: 词典类 Vocab
        :param vocab_tag: 标签的词典类 Vocab
        :param model_param: 用来传递模型参数的字典 dict
        """

        super(ParseModel, self).__init__(vocab, model_params, ctx)

        self._tag_embedding = nn.Embedding(vocab_tag.size, model_params['tag_emb_dim'])
        self._word_embedding = nn.Embedding(vocab.size, model_params['word_emb_dim'])
        params = [self._tag_embedding, self._word_embedding, vocab, vocab_tag]
        self._encoder = ParseEncoder(*params, model_params,ctx=ctx)
        self._decoder = BaseDecoder(self._word_embedding, vocab, model_params,ctx=ctx)

    def forward(self, inputs, targets):
        # print('starting encode')
        encoder_h, encoder_state, attention_value= self._encoder(inputs)  # N * T * 2C
        encoder_c = encoder_state[1]  # 2 * T * C
        # print('end encode')
        return self._decoder(targets, encoder_c, encoder_h, attention_value)


class Seq2SeqRNN(nn.Block):

    """implemention of alesee seq2seq and beyond with mxnet"""

    def __init__(self, vocab, rnn_type, emb_size, hidden_size, output_size, max_tgt_len, attention_type, tied_weight_type, pre_trained_vector, pre_trained_vector_type, padding_id, num_layers=1, encoder_drop=(0.2,0.3), decoder_drop=(0.2,0.3),bidirectional=True, bias=False, teacher_forcing=True, ctx=cpu()):


        nn.Block.__init__(self)
        rnn_type, attention_type, tied_weight_type = rnn_type.upper(), attention_type.title(), tied_weight_type.lower()
        if rnn_type in ['LSTM','GRU']:
            self._rnn_type = rnn_type
        else:
            raise ValueError("""Invalid option for 'rnn_type' was supplied, options are ['LSTM','GRU']""")
        
        if attention_type in ['Luong','Bahdanau']:
            self.attention_type = attention_type
        else:
            raise ValueError("""Invalid option for 'attention_type', options are ['Luong','Bahdanau']""")

        if tied_weight_type in ['three_way','two_way']:
            self.tied_weight_type = tied_weight_type
        else:
            raise ValueError("""Invalid option for 'tied_weight_type' options are ['three_way','two_way']""")

        self.vocab = vocab
        self.emb_size = emb_size
        self.hidden_size = hidden_size//2
        self.output_size = output_size
        self.max_tgt_len = max_tgt_len
        self.pre_trained_vector = pre_trained_vector
        self.pre_trained_vector_type = pre_trained_vector_type
        self.padding_id = padding_id
        self.num_layers = num_layers
        self.encoder_drop = encoder_drop
        self.decoder_drop = decoder_drop
        self.bidirectional = bidirectional
        self.bias = bias
        self.teacher_forcing = teacher_forcing
        
        if self.teacher_forcing:
            self.force_prob = 0.5

        if self.bidirectional:
            self.num_directions=2
        else:
            self.num_directions = 1


        self.encoder_dropout = nn.Dropout(self.encoder_drop[0])
        self.encoder_embedding_layer = nn.Embedding(vocab.size, self.emb_size)

        if self.pre_trained_vector:
            pass
        

        #TODO Encoder and Decoder
        self.encoder = RNNEncoder(
            rnn_type,
            self.hidden_size,
            self.emb_size,
            self.num_layers, 
            self.encoder_drop[1], 
            self.bidirectional,
            ctx=ctx
            )
        
        self.decoder = HeadDecoder(
            'DLSTM',
            self.hidden_size * self.num_directions,
            self.emb_size,
            self.output_size,
            self.decoder_drop,
            self.max_tgt_len,
            self.teacher_forcing,
            self.force_prob,
            ctx=ctx
        )

        self.decoder_dropout = nn.Dropout(self.decoder_drop[0])
        self.decoder_embedding_layer = nn.Embedding(vocab.size, self.emb_size)

    def forward(self, source, target, head_lda):
        self.batch_size = source.shape[0] 
        encoder_input = self.encoder_embedding_layer(source)
        target = self.decoder_embedding_layer(target)

        encoder_input = self.encoder_dropout(encoder_input)
        encoder_output, encoder_hidden = self.encoder(encoder_input)

        encoder_hidden[0] = encoder_hidden[0].transpose([1,0,2]).reshape([1,self.batch_size, self.hidden_size * 2])
        encoder_hidden[1] = encoder_hidden[1].transpose([1,0,2]).reshape([1,self.batch_size, self.hidden_size * 2])

        output = self.decoder(self.batch_size, encoder_output, encoder_hidden, head_lda, target)

        return output
   

class Model(object):
    """
    最顶层框架，控制模型参数、文件路径、梯度训练
    # DONE 读取数据
    # TODO 保存模型

    """

    def __init__(self, model_param, vocab_path, mode='train', vocab_tag_path=None, encoder_type='rnn', head_attention=False, decoder_cell='lstm', ctx=cpu()):
        """

        # TODO 选择模型的编码器解码器部分
        # TODO Encoder: Parsed | RNN
        # TODO Decoder: Headline | RNN
        # TODO Decoder_RNN_TYPE: DLSTM | LSMT | GRU

        根据参数与模式，构建模型

        :param mode: train|decode|test 控制当前模型的用途
        :param vocab_path: 词典路径
        :param vocab_tag_path: 句法解析标记词典路径
        :param model_param: 模型中的超参数
        """
        self.vocab_path = vocab_path
        self.vocab = Vocab(vocab_path)

        if vocab_tag_path is not None:
            self.vocab_tag_path = vocab_tag_path
            self.vocab_tag = Vocab(vocab_tag_path)

        self.mode = mode
        self.loss = SoftmaxCrossEntropyLoss()
        self.model_param = model_param
        self.encoder_type = encoder_type
        if encoder_type == 'rnn':
            pass
            # self.model = Seq2SeqRNN(self.vocab, self.model_param, ctx)
            self.model = Seq2SeqRNN(self.vocab, 'LSTM', model_param['emb_size'],model_param['hidden_size'],self.vocab.size, 60, 'Bahdanau', 'two_way', None, None, 0, 1, ctx=ctx)
        elif encoder_type == 'parse':
            self.model = ParseModel(self.vocab, self.vocab_tag, self.model_param, ctx)
        
        self.model.initialize(ctx=ctx)
        self.ctx = ctx
        self.trainer = Trainer(self.model.collect_params(), 'adam', {'learning_rate': 0.01})
        self.global_step = 0
        self.sw = SummaryWriter('./logs',flush_secs=2)

    def sequence_loss(self, logits, targets, weight=None):
        """

        计算序列的损失，也就是一个批次的损失（当然这个模型是一个批次）

        :param logits: 预测出的结果
        :param targets: 真实摘要
        :param weight: 根据句子长度来的padding weight，训练过程不需要
        :return: 损失值
        """
        if weight is None:
            logits = nd.reshape(logits, [-1, self.vocab.size])
        else:
            logits = logits * weight
            targets = logits * weight
        targets = nd.array(targets,ctx=self.ctx)
        loss = self.loss(logits, targets)
        loss = loss.sum() / len(targets)
        return loss

    def train_one_step(self, data, optimizer='adam', learning_rate=0.01, verbose=False):

        self.model(*data[0])
        loss_sum = 0.0
        for d in data:
            with autograd.record():
                logits = self.model(*d)
                loss = self.sequence_loss(logits, d[1])
            loss.backward()
            self.trainer.step(1)
            loss_sum += loss.asscalar()
        if verbose:
            print('loss= %.3f' % (loss_sum / len(data)))
        # self.model.collect_params().save('')
        return logits

    def _data_reader(self, batch_size, source_path, target_path):

        source_list= os.listdir(source_path)

        i = 0

        while i+batch_size <= len(source_list):
            batch_x = []
            batch_y = []
            for j in range(batch_size):
                batch_x.append(pickle.load(open(os.path.join(source_path, source_list[i+j]),'rb')))
                batch_y.append(pickle.load(open(os.path.join(target_path, source_list[i+j]),'rb')))
            
            yield batch_x, batch_y
        
    def _data_generator(self, batch_size, source_path, target_path):

        for x, y in self._data_reader(batch_size, source_path, target_path):
            if batch_size > 1:
                max_x_len = max(x,key=lambda t: len(t))
                max_y_len = max(y,key=lambda t: len(t))
                x = nd.zeros(shape=(batch_size, max_x_len))
                y = nd.zeros(shape=(batch_size, max_y_len))
                for i, (xt, yt) in enumerate(zip(x,y)):
                    x[i,:len(xt)] = nd.array(xt)
                    y[i,:len(yt)] = nd.array(yt)
            else:
                x = nd.array(x)
                y = nd.array(y)
            yield x, y

            
    def _data_generator_one_batch(self, source_path, target_path, lda_path):
        
        source_list= os.listdir(source_path)
        
        dataset = []
        for filename in source_list:
            
            x = nd.array([pickle.load(open(os.path.join(source_path, filename),'rb'))],ctx=self.ctx)
            y = nd.array(pickle.load(open(os.path.join(target_path, filename),'rb')), ctx=self.ctx)
            lda = nd.array([pickle.load(open(os.path.join(lda_path, filename),'rb'))],ctx=self.ctx)
            dataset.append([x,y,lda])
            
        return dataset
        
        
        
    def train(self, source_path, target_path, lda_path, batch_size=16, epoch_num=15, optimizer='adam', learning_rate=0.01, new_train=True):
        """

        根据定义好的模型，训练模型

        :param epoch_num: 训练迭代数据轮次
        :param optimizer:  优化器，或者是优化器名字str
        :param learning_rate: 学习率
        :return:
        """
        source_list, target_list = os.listdir(source_path), os.listdir(target_path)
        if len(source_list) != len(target_list):
            raise ValueError('source and target file not match')
        data_size = len(source_list)
        del source_list
        del target_list

        if self.encoder_type == 'parse':
            print('single-pass mode in parse encoder')
            batch_size = 1
        
        trainer = Trainer(self.model.collect_params(), optimizer, {'learning_rate': learning_rate})
        print('Reading data...')
        data = self._data_generator_one_batch(source_path, target_path, lda_path)
        if new_train is False:
            self.model.collect_params().load('best.model', ctx=self.ctx) 
        best_score = 9999
        for epoch in range(epoch_num):
            loss_sum = 0.0
            with tqdm(total=len(data)) as pbar:
                for x, y, lda in data:
                    with autograd.record():
                        logits = self.model(x, y, lda)
                        loss = self.sequence_loss(logits, y)
                    loss.backward()
                    trainer.step(1)
                    loss_sum += loss.asscalar()
                    pbar.update(1)
                    self.global_step += 1
                loss_mean = loss_sum / len(data)
                print('epoch {}, loss:{}'.format(epoch,loss_mean))
                if loss_mean < best_score:
                    self.model.collect_params().save('best.model')





