
# coding: utf-8

# In[1]:

import nltk


# In[2]:

import jieba
import tflearn


# In[3]:

import tensorflow as tf


# In[4]:



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

# In[24]:

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

print("read complete")

# In[25]:

len(ind2ch),len(ind2en)


# In[26]:

src_vocab_size = 80000#len(ind2en) + 3
target_vocat_size = 90000#len(ind2ch) + 3
attention_hidden_size = 512
attention_output_size = 512
embedding_size = 512
seq_max_len = 50
num_units = 512
batch_size = 64
layer_number = 2
max_grad = 1.0
dropout = 0.2


# In[27]:

len(train_x),len(train_y)


# In[28]:

#train_x = [i[::-1] for i in train_x]


# In[29]:

train_x = tf.contrib.keras.preprocessing.sequence.pad_sequences(train_x,seq_max_len,padding='post',value=en2ind['<eos>'])
train_y = tf.contrib.keras.preprocessing.sequence.pad_sequences(train_y,seq_max_len,padding='post',value=ch2ind['<eos>'])


# In[30]:

import random
index = random.randint(0,len(train_x))
print(' '.join([ind2en.get(i,'') for i in train_x[index]]))
print(' '.join([ind2ch.get(i,'') for i in train_y[index]]))


# In[31]:

print(train_x.shape,train_y.shape)


# In[32]:

from sklearn.cross_validation import train_test_split


# In[33]:

train_x,test_x,train_y,test_y = train_test_split(train_x,train_y , test_size=0.01, random_state=42)


# In[34]:

train_x[train_x >= src_vocab_size] = en2ind['<unk>']
test_x[test_x >= src_vocab_size] = en2ind['<unk>'] 


# In[35]:

train_y[train_y >= target_vocat_size] = ch2ind['<unk>']
test_y[test_y >= target_vocat_size] = ch2ind['<unk>']


# In[38]:

import random
for i in range(5):
    index = random.randint(0,len(train_x))
    print(' '.join([ind2en.get(i,'') for i in train_x[index]]))
    print(' '.join([ind2ch.get(i,'') for i in train_y[index]]))


# In[39]:

from tensorflow.python.layers import core as layers_core


# In[40]:

import tensorflow as tf
import tflearn
tf.reset_default_graph()
config = tf.ConfigProto(log_device_placement=True,allow_soft_placement = True)
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)


with tf.device('/gpu:1'):
    #initializer = tf.random_uniform_initializer(
    #    -0.02, 0.02)
    initializer = tf.truncated_normal_initializer(
        0,0.02)
    tf.get_variable_scope().set_initializer(initializer)
    
    x = tf.placeholder("int32", [None, None])
    y = tf.placeholder("int32", [None, None])
    y_in = tf.placeholder("int32",[None,None])
    x_len = tf.placeholder("int32",[None])
    y_len = tf.placeholder("int32",[None])
    x_real_len = tf.placeholder("int32",[None])
    y_real_len = tf.placeholder("int32",[None])
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
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        attention_hidden_size, encoder_outputs,
        memory_sequence_length=x_real_len,scale=True)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell, attention_mechanism,
        attention_layer_size=attention_output_size)
    
    
    projection_layer = layers_core.Dense(
        target_vocat_size, use_bias=False)
    
    
    
    # Dynamic decoding
    with tf.variable_scope("decode_layer"):
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp,sequence_length= y_len)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, initial_state = decoder_cell.zero_state(dtype=tf.float32,batch_size=batch_size),
            output_layer=projection_layer)
       
        outputs, _,___  = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output

        target_weights = tf.sequence_mask(
            y_real_len, seq_max_len, dtype=logits.dtype)
    
    # predicting
    # Helper
    with tf.variable_scope("decode_layer", reuse=True):
        helper_predict = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding_decoder,
            tf.fill([batch_size], ch2ind['<go>']), ch2ind['<eos>'])
        decoder_predict = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper_predict, initial_state = decoder_cell.zero_state(dtype=tf.float32,batch_size=batch_size),
            output_layer=projection_layer)
        outputs_predict,_, __ = tf.contrib.seq2seq.dynamic_decode(
            decoder_predict, maximum_iterations=test_y.shape[1] * 2)
    translations = outputs_predict.sample_id

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


# In[26]:

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


# In[27]:

session.run(tf.global_variables_initializer())


# In[28]:

saver = tf.train.Saver()


# In[29]:

#saver.restore(session,'middleresult/align/result_1_20847')


# In[31]:

saver.save(session,'models/result_deep/sample')


# In[40]:

#saver.save(session,'middleresult/result_char')


# In[32]:

from utils import Dataset,ProgressBar


# In[33]:

from utils import *
train_set = Dataset(train_x,train_y)
test_set = Dataset(test_x,test_y)


# In[34]:

def get_bleu_score(predict,target):
    try:
        target = [[[j for index,j in enumerate(i) if j > 0 or index < 4]] for i in target]
        predict = [[j for index,j in enumerate(i) if j > 0 or index < 4] for i in predict]
        BLEUscore = nltk.translate.bleu_score.corpus_bleu(target,predict)
    except:
        BLEUscore = -1
    return BLEUscore


# In[35]:

import numpy as np
def calc_test_loss(test_set = Dataset(test_x,test_y),display=True):
    accs = []
    worksum = int(len(test_x) / batch_size)
    loss_list = []
    predict_list = []
    target_list = []
    source_list = []
    pb = ProgressBar(worksum=worksum,info="validating...",auto_display=display)
    pb.startjob()
    #test_set = Dataset(test_x,test_y)
    for j in range(worksum):
        batch_x,batch_y = test_set.next_batch(batch_size)
        lx = [seq_max_len] * batch_size
        ly = [seq_max_len] * batch_size
        bx = [np.sum(m > 0) for m in batch_x]
        by = [np.sum(m > 0) for m in batch_y]
        tmp_loss,tran = session.run([train_loss,translations],feed_dict={x:batch_x,y:batch_y,
                                                     y_in:
                                                     np.concatenate((
                                                     np.ones((batch_y.shape[0],1),dtype=np.int) * ch2ind['<go>'],batch_y[:,:-1]) ,axis=1)
                                                     ,x_len:lx,y_len:ly,
                                                                        y_real_len:by,
                                                                        x_real_len:bx})
        loss_list.append(tmp_loss)
        tmp_acc = cal_acc(tran,batch_y)
        accs.append(tmp_acc)
        predict_list += [i for i in tran]
        target_list += [i for i in batch_y]
        source_list += [i for i in batch_x]
        pb.complete(1)
    return np.average(loss_list),np.average(accs),get_bleu_score(predict_list,target_list),predict_list,target_list,source_list


# In[36]:

#w_loss,w_acc,bleu_score,predict_list,target_list,source_list = calc_test_loss(Dataset(train_x[::100],train_y[::100]))


# In[37]:

#print('loss, acc,bleu',w_loss,w_acc,bleu_score)


# In[38]:

def get_all_text(x):
    return [' '.join([ind2ch.get(j,'') for j in i]) for i in x]
def get_all_en_text(x):
    return [' '.join([ind2en.get(j,'') for j in i]) for i in x]


## In[39]:
#
#texts = get_all_text(predict_list)
#texts[:10]
#
#
## In[40]:
#
#texts = get_all_text(target_list)
#texts[:10]
#
#with open('eval/tranwer.txt','w',encoding='utf-8') as whdl:
#    for line in texts:
#        whdl.write("{}\n".format(line))
# In[41]:

#loss,tran = session.run([train_loss,translations],feed_dict={x:batch_x,y:batch_y,x_len:lx,y_len:ly,learning_rate:lr,y_in:
#                                                                np.concatenate((
#                                                                np.ones((batch_y.shape[0],1),dtype=np.int) * ch2ind['<go>'],batch_y[:,:-1]) ,axis=1)
#                                                               })


# In[42]:

#tran.shape
i_save = 0
j_save = 0


# In[43]:

print(i_save,j_save)


# In[44]:

model_path = 'align_word_shallow'


# In[45]:

#os.mkdir('middleresult/{}'.format(model_path))
#os.mkdir('eval/{}'.format(model_path))


# In[ ]:

n_epoch = 60
restore = True
lr = 1
for i in range(i_save,n_epoch):
    
    i_save = i
    worksum = int(len(train_y)/batch_size)
    pb = ProgressBar(worksum=worksum)
    pb.startjob()
    train_loss_list = []
    train_acc_list = []
    for j in range(worksum):
        if restore == True and j < j_save:
            pb.finishsum += 1
            continue
        restore = False
        
        j_save = j
        batch_x,batch_y = train_set.next_batch(batch_size)
        lx = [seq_max_len] * batch_size
        ly = [seq_max_len] * batch_size
        bx = [np.sum(m > 0) for m in batch_x]
        by = [np.sum(m > 0) for m in batch_y]
        by =[m + 2  if m < seq_max_len - 1 else m for m in by ]
        _, loss = session.run([optimizer,train_loss],feed_dict={x:batch_x,y:batch_y,x_len:lx,y_len:ly,learning_rate:lr,y_in:
                                                                np.concatenate((
                                                                np.ones((batch_y.shape[0],1),dtype=np.int) * ch2ind['<go>'],batch_y[:,:-1]) ,axis=1)
                                                                ,y_real_len:by,
                                                                x_real_len:bx
                                                               })
        train_loss_list.append(loss)
        #tmp_train_acc = cal_acc(tran,batch_y)
        #train_acc_list.append(tmp_train_acc)
        pb.info = "iter {} loss:{} lr:{}".format(i + 1,loss,lr)
        val_step = int(worksum / 4)
        if j % val_step == 0 and j != 0:
            test_loss,test_acc,bleu_score,predict_list,target_list,source_list = calc_test_loss()
            _,train_acc,train_bleu_score,train_predict_list,train_target_list,train_source_list = calc_test_loss(Dataset(train_x[::100],train_y[::100]),display=False)
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
            with open('eval/{}/train_loss'.format(model_path),'w',encoding='utf-8') as whdl:
                whdl.write("iter {} step {} train loss {} train acc {} test loss {} test acc {} bleu {} lr {}\n".format(i+1,j,np.average(train_loss_list[-val_step:]),train_acc,test_loss,test_acc,bleu_score,lr))
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

