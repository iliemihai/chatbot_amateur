
# In[1]:

import tensorflow as tf
import numpy as np

# preprocessed data
import data
import data_utils
import sys
import pandas as pd
# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='Eminescu/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 32
xvocab_size = len(metadata['idx2w'])  
print (xvocab_size)
yvocab_size = xvocab_size
emb_dim = 1024

import seq2seq_wrapper

# In[7]:

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='Eminescu/ckpt/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )


# In[8]:

val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)

test_batch_gen = data_utils.rand_batch_gen(testX, testY, 256)
# In[9]:
sess = model.restore_last_session()
sys.stdout.write('\n<Incepe antrenarea\n')
#sess = model.train(train_batch_gen, val_batch_gen)
sys.stdout.write('\nDupa\n')
input_ = []
replies = []

while 1:

    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()

    sentence = sentence.lower()
    sentence = sentence.replace("\n","")
    sentence = sentence.split(" ")
    #input_ = test_batch_gen.__next__()[0]

    print (sentence)
    inputs = data_utils.decode_sentence(sequence=sentence, lookup=metadata['w2idx'])
    for i in range(model.xseq_len):
        if i < len(inputs):
            input_.append([inputs[i]])
        else:
            input_.append([0])

    output = model.predict(sess, input_)
    print (input_)
    print (output)
    #q = data_utils.decode(sequence=input_, lookup=metadata['idx2w'], separator=' ')
    decoded = data_utils.decode(sequence=output[0], lookup=metadata['idx2w'], separator=' ').split(' ')
    print ("-",decoded)
    input_[:] = []
#    if decoded.count('unk') == 0:
#        if decoded not in replies:
#            print(' a : [{1}]'.format( ' '.join(decoded)))
#            replies.append(decoded)
