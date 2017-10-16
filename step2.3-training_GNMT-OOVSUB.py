
# coding: utf-8

# In[18]:


import nltk


# In[19]:


import jieba
import tflearn
import os
import sys


# In[20]:


import tensorflow as tf


# In[21]:


tf.__version__


# In[22]:


from tensorflow.contrib.keras import preprocessing


# import pickle
# with open('middleresult/en_ch_35word.pkl','wb') as whdl:
#     pickle.dump((
#         train_x,
#         test_x,
#         train_y,
#         test_y,
#         ind2ch,
#         ch2ind,
#         ind2en,
#         en2ind,
#     ),whdl)

# # problems we try to solve here:
# 1. bleu calculate method
# 2. GNMT cell 
# 3. add  model complex
# 4. beam search

# # read data

# In[23]:


import pickle
with open('data/preprocessing_tokenlizer/sentence_tokened_by_word.pkl','rb') as fhdl:
    (
         ind2ch,
         ch2ind,
         ind2en,
         en2ind,
         train_x,
         train_y,
    ) = pickle.load(fhdl)


# In[24]:


with open('data/preprocessing_subword/subwords_allwords.en','rb') as fhdl:
    en_subword = pickle.load(fhdl)


# In[25]:


en_subword_dic = dict(zip(en_subword['origin'],en_subword['segmented']))


# In[26]:


with open('middleresult/char/zh_vocab.txt',encoding='utf-8') as fhdl:
    ch_subwords = [line.strip().split("\t") for line in fhdl]


# In[27]:


len(ind2ch),len(ind2en)


# In[58]:


src_inv_size_base = 40000#len(ind2en) + 3
target_inv_size_base = 40000#len(ind2ch) + 3

USE_GPU = 1

attention_hidden_size = 1024
attention_output_size = 1024
embedding_size = 1024
seq_max_len = 60
num_units = 1024
batch_size = 64
layer_number = 4
max_grad = 1.0
dropout = 0.2
sentence_max_length = 70
beam_width = 3
length_penalty_weight = 0


# In[29]:


ch_inv = list(map(lambda x:x[0],sorted(ch2ind.items(),key=lambda x:x[1])[:target_inv_size_base]))


# In[30]:


en_inv = list(map(lambda x:x[0],sorted(en2ind.items(),key=lambda x:x[1])[:src_inv_size_base]))


# In[31]:


en_inv = en_inv[:3] + ['_' + i for i in en_inv[3:]]


# In[32]:


ch_inv_tmpdic = dict(zip(ch_inv,range(len(ch_inv))))
ch_oov = [i[0] for i in ch_subwords if i[0] not in ch_inv_tmpdic]
en_inv_tmpdic = dict(zip(en_inv,range(len(en_inv))))
en_oov = [i for i in en_subword['subwords'] if i not in en_inv_tmpdic]


# In[33]:


ind2ch_oov = dict(zip(range(len(ch_oov) + len(ch_inv)),ch_inv + ch_oov))
ch2ind_oov = dict(zip(ch_inv + ch_oov,range(len(ch_oov) + len(ch_inv))))


# In[34]:


ind2en_oov = dict(zip(range(len(en_oov) + len(en_inv)),en_inv + en_oov))
en2ind_oov = dict(zip(en_inv + en_oov,range(len(en_oov) + len(en_inv))))


# In[35]:


len(ch_oov),len(en_oov)


# In[36]:


src_vocab_size = src_inv_size_base + len(en_oov)
target_vocat_size = target_inv_size_base + len(ch_oov)


# In[37]:


src_vocab_size,len(ind2en_oov),target_vocat_size,len(ind2ch_oov)


# In[38]:


len(train_x),len(train_y)


# In[39]:


import numpy as np
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)


# In[40]:


en2ind['james'],en2ind_oov['_james']


# In[41]:


#train_x = [i[::-1] for i in train_x]


# In[42]:


#train_x = sequence.pad_sequences(train_x,seq_max_len,padding='post',value=en2ind['<eos>'])
#train_y = sequence.pad_sequences(train_y,seq_max_len,padding='post',value=ch2ind['<eos>'])


# In[43]:


import random
index = random.randint(0,len(train_x))
print(' '.join([ind2en.get(i,'') for i in train_x[index]]))
print(' '.join([ind2ch.get(i,'') for i in train_y[index]]))


# In[44]:


from sklearn.cross_validation import train_test_split


# In[45]:


train_x,test_x,train_y,test_y = train_test_split(train_x,train_y , test_size=0.01, random_state=42)


# In[46]:


len(train_x),len(test_x),len(train_y),len(test_y)


# In[47]:


def en_ind2ind(sentence):
    sentence_inds = []
    for wordind in sentence:
        if wordind < src_inv_size_base:
            sentence_inds.append(wordind)
        else:
            en_word = ind2en[wordind]
            if en_word not in en_subword_dic:
                sentence_inds.append(en2ind_oov['<unk>'])
            en_pieces = en_subword_dic[en_word]
            pieces_index = [en2ind_oov[i] for i in en_pieces]
            sentence_inds += pieces_index
    return sentence_inds


# In[48]:


def ch_ind2ind(sentence):
    sentence_inds = []
    for wordind in sentence:
        if wordind < target_inv_size_base:
            sentence_inds.append(wordind)
        else:
            ch_word = ind2ch[wordind]
            en_pieces = [i for i in ch_word]
            pieces_index = [ch2ind_oov.get(i,ch2ind_oov['<unk>']) for i in en_pieces]
            sentence_inds += pieces_index
    return sentence_inds


# In[49]:


train_x_tmp = []
for index,sentence in enumerate(train_x):
    train_x_tmp.append(en_ind2ind(sentence))


# In[50]:


train_y_tmp = []
for index,sentence in enumerate(train_y):
    train_y_tmp.append(ch_ind2ind(sentence))


# In[51]:


test_x_tmp = []
for index,sentence in enumerate(test_x):
    test_x_tmp.append(en_ind2ind(sentence))


# In[52]:


test_y_tmp = []
for index,sentence in enumerate(test_y):
    test_y_tmp.append(ch_ind2ind(sentence))


# In[53]:


train_x = np.asarray(train_x_tmp)
train_y = np.asarray(train_y_tmp)
test_x = np.asarray(test_x_tmp)
test_y = np.asarray(test_y_tmp)


# In[54]:


del train_x_tmp
del train_y_tmp
del test_x_tmp
del test_y_tmp


# In[55]:


import random
for i in range(5):
    index = random.randint(0,len(train_x))
    print(' '.join([ind2en_oov.get(i,'') for i in train_x[index]]))
    print(' '.join([ind2ch_oov.get(i,'') for i in train_y[index]]))


# In[56]:


len(train_x),len(test_x),len(train_y),len(test_y)


# In[57]:


from tensorflow.python.layers import core as layers_core


# In[59]:


import tensorflow as tf
import tflearn
tf.reset_default_graph()
config = tf.ConfigProto(log_device_placement=True,allow_soft_placement = True)
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)


with tf.device('/gpu:{}'.format(USE_GPU)):
    #initializer = tf.random_uniform_initializer(
    #    -0.08, 0.08)
    initializer = tf.truncated_normal_initializer(
        mean=0.0,stddev=0.02)
    tf.get_variable_scope().set_initializer(initializer)
    
    x = tf.placeholder("int32", [None, None])
    y = tf.placeholder("int32", [None, None])
    y_in = tf.placeholder("int32",[None,None])
    x_len = tf.placeholder("int32",[None])
    y_len = tf.placeholder("int32",[None])
    x_real_len = tf.placeholder("int32",[None])
    y_real_len = tf.placeholder("int32",[None])
    y_max_len = tf.placeholder(tf.int32, shape=[])
    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    # embedding
    embedding_encoder = tf.get_variable(
        "embedding_encoder", [src_vocab_size, embedding_size],dtype=tf.float32)
    embedding_decoder = tf.get_variable(
        "embedding_decoder", [target_vocat_size, embedding_size],dtype=tf.float32)
    
    encoder_emb_inp = tf.nn.embedding_lookup(
        embedding_encoder, x)
    decoder_emb_inp = tf.nn.embedding_lookup(
        embedding_decoder, y_in)
    
    # encoder
    num_bi_layers = int(layer_number / 2)
    cell_list = []
    for i in range(num_bi_layers):
        cell_list.append(
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(num_units), input_keep_prob=(1.0 - dropout)
            )
        )
    if len(cell_list) == 1:
        encoder_cell = cell_list[0]
    else:
        encoder_cell = tf.contrib.rnn.MultiRNNCell(cell_list)
        
    cell_list = []
    
    for i in range(num_bi_layers):
        cell_list.append(
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(num_units), input_keep_prob=(1.0 - dropout)
            )
        )
    if len(cell_list) == 1:
        encoder_backword_cell = cell_list[0]
    else:
        encoder_backword_cell = tf.contrib.rnn.MultiRNNCell(cell_list)
    
    bi_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
        encoder_cell,encoder_backword_cell, encoder_emb_inp,
        sequence_length=x_len,dtype=tf.float32)
    encoder_outputs = tf.concat(bi_outputs, -1)
    
    if num_bi_layers == 1:
        encoder_state = bi_encoder_state
    else:
        encoder_state = []
        for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
        encoder_state = tuple(encoder_state)
    
    # decoder 
    #decoder_cell = tf.contrib.rnn.BasicLSTMCell(num_units)
    cell_list = []
    for i in range(layer_number):
        cell_list.append(
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(num_units), input_keep_prob=(1.0 - dropout)
            )
        )
    if len(cell_list) == 1:
        decoder_cell = cell_list[0]
    else:
        decoder_cell = tf.contrib.rnn.MultiRNNCell(cell_list)
    
    # Helper
    
    # attention
    
    
    
    projection_layer = layers_core.Dense(
        target_vocat_size, use_bias=False)
    
    
    
    # Dynamic decoding
    with tf.variable_scope("decode_layer"):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            attention_hidden_size, encoder_outputs,
            memory_sequence_length=x_real_len,scale=True)
        decoder_cell_att = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=attention_output_size)
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp,sequence_length= y_len)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell_att, helper, initial_state = decoder_cell_att.zero_state(dtype=tf.float32,batch_size=batch_size),
            output_layer=projection_layer)
       
        outputs, _,___  = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output

        target_weights = tf.sequence_mask(
            y_real_len, y_max_len, dtype=logits.dtype)
    
    # predicting
    # Helper
    with tf.variable_scope("decode_layer", reuse=True):
        #helper_predict = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        #    embedding_decoder,
        #    tf.fill([batch_size], ch2ind['<go>']), ch2ind['<eos>'])
        #decoder_predict = tf.contrib.seq2seq.BasicDecoder(
        #    decoder_cell, helper_predict, initial_state = decoder_cell.zero_state(dtype=tf.float32,batch_size=batch_size),
        #    output_layer=projection_layer)
        
        
        #    tf.contrib.seq2seq.tile_batch(decoder_start, multiplier=beam_width)
        #tf.contrib.seq2seq.tile_batch(
        #    decoder_start, multiplier=beam_width)
        encoder_outputs_infer = tf.contrib.seq2seq.tile_batch(
          encoder_outputs, multiplier=beam_width)
        x_real_len_infer = tf.contrib.seq2seq.tile_batch(
          x_real_len, multiplier=beam_width)
        
        
        
        attention_mechanism_infer = tf.contrib.seq2seq.LuongAttention(
            attention_hidden_size, encoder_outputs_infer,
            memory_sequence_length=x_real_len_infer,scale=True)
        decoder_cell_infer = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism_infer,
            attention_layer_size=attention_output_size)
        
        decoder_start = decoder_cell_infer.zero_state(dtype=tf.float32,batch_size=batch_size * beam_width)
        decoder_initial_state = decoder_start
        
        decoder_predict = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=decoder_cell_infer,
              embedding=embedding_decoder,
              start_tokens=tf.fill([batch_size], ch2ind_oov['<go>']),
              end_token=ch2ind_oov['<eos>'],
              initial_state=decoder_initial_state,
              beam_width=beam_width,
              output_layer=projection_layer,
              length_penalty_weight=length_penalty_weight)
        
        outputs_predict,_, __ = tf.contrib.seq2seq.dynamic_decode(
            decoder_predict, maximum_iterations=sentence_max_length)
    translations = outputs_predict.predicted_ids

    # calculate loss
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits)
    train_loss = (tf.reduce_sum(crossent * target_weights) /
        batch_size)
    
    optimizer_ori = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, max_grad)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = optimizer_ori.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=global_step)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(train_loss)
    #trainop = tflearn.TrainOp(loss=train_loss, optimizer=optimizer,
    #                          metric=train_loss, batch_size=64)


# In[60]:


def cal_acc(logits,target):
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target[:,:seq_max_len], logits[:,:seq_max_len]))


# In[61]:


session.run(tf.global_variables_initializer())


# In[61]:


saver = tf.train.Saver()


# In[62]:


#saver.restore(session,'middleresult/align/result_1_20847')


# In[ ]:


saver.save(session,'middleresult/result_deep')


# In[ ]:


#saver.save(session,'middleresult/result_char')


# In[62]:


from utils import Dataset,ProgressBar


# In[63]:


def get_bleu_score(predict,target):
    try:
        target = [[[j for index,j in enumerate(i)]] for i in target]
        predict = [[j for index,j in enumerate(i)] for i in predict]
        BLEUscore = nltk.translate.bleu_score.corpus_bleu(target,predict)
    except:
        BLEUscore = -1
    return BLEUscore


# In[64]:


print(len(test_x),len(test_y))


# In[65]:


test_x_len = [len(i) for i in test_x]


# In[66]:


len(test_x_len)


# In[67]:


tmp = list(filter(lambda x:x[2] < 50,sorted(zip(test_x,test_y,test_x_len),key=lambda x:x[2])))


# In[68]:


test_x = np.asarray([i[0] for i in tmp])
test_y = np.asarray([i[1] for i in tmp])


# In[69]:


del test_x_len


# In[70]:


import random
for i in range(5):
    index = random.randint(0,len(test_x[:1500]))
    print(' '.join([ind2en_oov.get(i,'') for i in test_x[index]]))
    print(' '.join([ind2ch_oov.get(i,'') for i in test_y[index]]))


# In[71]:


len(test_x),len(test_y)


# In[74]:


import numpy as np
def calc_test_loss(test_x,test_y,display=True):
    accs = []
    worksum = int(len(test_x) / batch_size)
    loss_list = []
    predict_list = []
    target_list = []
    source_list = []
    pb = ProgressBar(worksum=worksum,info="validating...",auto_display=display)
    pb.startjob()
    #test_set = Dataset(test_x,test_y)
    for j in range(0,len(test_x),batch_size):
        batch_x,batch_y = test_x[j:j + batch_size],test_y[j:j + batch_size]#test_set.next_batch(batch_size)
        if len(batch_x) < batch_size:
            continue
        bx = [len(m) + 1 for m in batch_x]
        by = [len(m) + 1 for m in batch_y]
        
        lx = [max(bx)] * batch_size
        ly = [max(by)] * batch_size
        
        batch_x = preprocessing.sequence.pad_sequences(batch_x,max(bx),padding='post',value=en2ind_oov['<eos>'])
        batch_y = preprocessing.sequence.pad_sequences(batch_y,max(by),padding='post',value=ch2ind_oov['<eos>'])
        
        tmp_loss,tran = session.run([train_loss,translations],feed_dict={x:batch_x,y:batch_y,
                                                     y_in:
                                                     np.concatenate((
                                                     np.ones((batch_y.shape[0],1),dtype=np.int) * ch2ind['<go>'],batch_y[:,:-1]) ,axis=1)
                                                     ,x_len:lx,y_len:ly,
                                                                        y_real_len:by,
                                                                        x_real_len:bx,
                                                                        y_max_len:max(by)
                                                                        })
        loss_list.append(tmp_loss)
        tmp_acc = cal_acc(tran[:,:,0],batch_y)
        accs.append(tmp_acc)
        predict_list += [i for i in tran[:,:,0]]
        target_list += [i for i in batch_y]
        source_list += [i for i in batch_x]
        pb.complete(1)
    return np.average(loss_list),np.average(accs),get_bleu_score(predict_list,target_list),predict_list,target_list,source_list


# In[75]:


w_loss,w_acc,bleu_score,predict_list,target_list,source_list = calc_test_loss(train_x[::10000],train_y[::10000])


# In[76]:


print(w_loss,w_acc,bleu_score)


# In[77]:


def get_all_text(x):
    return [' '.join([ind2ch_oov.get(j,'') for j in i]) for i in x]
def get_all_en_text(x):
    return [' '.join([ind2en_oov.get(j,'') for j in i]) for i in x]


# In[78]:


#loss,tran = session.run([train_loss,translations],feed_dict={x:batch_x,y:batch_y,x_len:lx,y_len:ly,learning_rate:lr,y_in:
#                                                                np.concatenate((
#                                                                np.ones((batch_y.shape[0],1),dtype=np.int) * ch2ind['<go>'],batch_y[:,:-1]) ,axis=1)
#                                                               })


# In[79]:


#tran.shape
i_save = 0
j_save = 0


# In[80]:


print(i_save,j_save)


# In[81]:


model_path = 'OOVSUB_beamsearch'


# In[87]:


if not os.path.exists('middleresult/{}'.format(model_path)):os.mkdir('middleresult/{}'.format(model_path))
if not os.path.exists('eval/{}'.format(model_path)):os.mkdir('eval/{}'.format(model_path))
if not os.path.exists('val/{}'.format(model_path)):os.mkdir('val/{}'.format(model_path))


# In[88]:


def get_tmpindexs(train_index_set):
    tmp_indexs,_ = train_index_set.next_batch(batch_size * batch_gen)
    tmp_length = [len(train_x[i]) for i in tmp_indexs]
    tmp_indexs = [i[0] for i in sorted(zip(tmp_indexs,tmp_length),key=lambda x:x[1])]
    tmp = []
    for i in random.sample(range(batch_gen),batch_gen):
        tmp += tmp_indexs[i * batch_size:(i + 1) * batch_size]
    tmp_indexs = tmp
    return tmp_indexs


# In[89]:


tmp_indexs = []


# In[196]:


n_epoch = 60
restore = False
lr = 1

batch_gen = 100

from utils import *
if not restore:
    train_index_set = Dataset(np.arange(len(train_x)),np.arange(len(train_y)))
    tmp_indexs = []
    
exp_loss = None
alpha = 0.97
for i in range(i_save,n_epoch):
    one_epoch = i + 1
    i_save = i
    worksum = int(len(train_y)/batch_size)
    pb = ProgressBar(worksum=worksum)
    pb.startjob()
    train_loss_list = []
    train_acc_list = []
    for j in range(worksum):
        one_batch = j
        if restore == True and j < j_save:
            pb.finishsum += 1
            continue
        restore = False
        
        j_save = j
        
        if tmp_indexs == []:
            tmp_indexs = get_tmpindexs(train_index_set)
        batch_indexs,tmp_indexs = tmp_indexs[:batch_size],tmp_indexs[batch_size:]
        batch_x,batch_y = train_x[batch_indexs],train_y[batch_indexs]

        bx = [min(len(m) + 1,seq_max_len) for m in batch_x]
        by = [min(len(m) + 1,seq_max_len) for m in batch_y]
        
        lx = [max(bx)] * batch_size
        ly = [max(by)] * batch_size
        
        batch_x = preprocessing.sequence.pad_sequences(batch_x,max(bx),padding='post',value=en2ind_oov['<eos>'])
        batch_y = preprocessing.sequence.pad_sequences(batch_y,max(by),padding='post',value=ch2ind_oov['<eos>'])
        #print(batch_x.shape,batch_y.shape)
        
        _, loss = session.run([optimizer,train_loss],feed_dict={x:batch_x,y:batch_y,x_len:lx,y_len:ly,learning_rate:lr,y_in:
                                                                np.concatenate((
                                                                np.ones((batch_y.shape[0],1),dtype=np.int) * ch2ind['<go>'],batch_y[:,:-1]) ,axis=1)
                                                                ,y_real_len:by,
                                                                x_real_len:bx,
                                                                y_max_len:max(by)
                                                               })
        train_loss_list.append(loss)
        #tmp_train_acc = cal_acc(tran,batch_y)
        #train_acc_list.append(tmp_train_acc)
        exp_loss = loss if exp_loss == None else alpha * exp_loss + (1 - alpha) * loss
        pb.info = "iter {} loss:{} lr:{}".format(i + 1,exp_loss,lr)
        with open('val/{}/train_loss.txt'.format(model_path),'a') as whdl:
            whdl.write("{}\t{}\t{}\n".format(one_epoch,one_batch,loss))
        val_step = int(worksum / 4)
        if j % val_step == 0 and j != 0:
            test_loss,test_acc,bleu_score,predict_list,target_list,source_list = calc_test_loss(test_x[::4],test_y[::4])
            _,train_acc,train_bleu_score,train_predict_list,train_target_list,train_source_list = calc_test_loss(train_x[::1000],train_y[::1000])
            predict_texts = get_all_text(predict_list)
            target_texts = get_all_text(target_list)
            source_texts = get_all_en_text(source_list)
            
            train_predict_texts = get_all_text(train_predict_list)
            train_target_texts = get_all_text(train_target_list)
            train_source_texts = get_all_en_text(train_source_list)
            
            with open('eval/{}/{}_{}_predict'.format(model_path,i + 1,j),'w',encoding='utf-8') as whdl:
                for line in predict_texts:
                    whdl.write("{}\n".format(line))
            with open('eval/{}/{}_{}_target'.format(model_path,i + 1,j),'w',encoding='utf-8') as whdl:
                for line in target_texts:
                    whdl.write("{}\n".format(line))
            with open('eval/{}/{}_{}_source'.format(model_path,i + 1,j),'w',encoding='utf-8') as whdl:
                for line in source_texts:
                    whdl.write("{}\n".format(line))
                    
            with open('eval/{}/{}_{}_predict_train'.format(model_path,i + 1,j),'w',encoding='utf-8') as whdl:
                for line in train_predict_texts:
                    whdl.write("{}\n".format(line))
            with open('eval/{}/{}_{}_target_train'.format(model_path,i + 1,j),'w',encoding='utf-8') as whdl:
                for line in train_target_texts:
                    whdl.write("{}\n".format(line))
            with open('eval/{}/{}_{}_source_train'.format(model_path,i + 1,j),'w',encoding='utf-8') as whdl:
                for line in train_source_texts:
                    whdl.write("{}\n".format(line))
            print("\niter {} step {} train loss {} train acc {} test loss {} test acc {} bleu {} lr {}\n".format(i+1,j,np.average(train_loss_list[-val_step:]),train_acc,test_loss,test_acc,bleu_score,lr))
            with open('val/{}/test_loss.txt'.format(model_path),'a') as whdl:
                whdl.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(i+1,j,np.average(train_loss_list[-val_step:]),train_acc,test_loss,test_acc,bleu_score,lr))
            try:
                saver = tf.train.Saver()
                saver.save(session,'middleresult/{}/result_{}_{}'.format(model_path,i + 1,j))
            except:
                print('save fail')
        lr_step = int(worksum / 2) - 1
        if j % lr_step == 0 and j != 0:
            if (i + 1) >= 6:
                lr = lr / 2
        pb.complete(1)

