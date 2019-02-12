# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 19:28:57 2019

@author: Mohit Kumar
"""
#importing Library
import tensorflow as tf
import numpy as np
import re
import time

lines= open("movie_lines.txt",encoding='utf-8',errors='ignore').read().split('\n')
conversations= open("movie_conversations.txt",encoding='utf-8',errors='ignore').read().split('\n')

#creating dictionary
id2line={}
for line in lines:
    k=line.split(' +++$+++ ')
    if len(k)==5:
        id2line[k[0]]=k[4]
conversations_ids=[]
#1:-1 replace []
for conv in conversations[:-1]:
    k1=conv.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(k1.split(','))

#create question and answer
#print(id2line)
question=[]
answer=[]
for i in conversations_ids:
    for j in range(len(i)-1):
        question.append(id2line[i[j]])
        answer.append(id2line[i[j+1]])
#cleaning
def clean(a):
    a=a.lower()
    a=re.sub(r"i'm"," i am",a)
    a=re.sub(r"he's"," he is",a)
    a=re.sub(r"she's"," she is",a)
    a=re.sub(r"that's"," that is",a)
    a=re.sub(r"what's"," what is",a)
    a=re.sub(r"where's"," where is",a)
    a=re.sub(r"\'ll"," will",a)
    a=re.sub(r"\'ve"," have",a)
    a=re.sub(r"\'d"," would",a)
    a=re.sub(r"\'re"," are",a)
    a=re.sub(r"won't","will not",a)
    a=re.sub(r"can't","cannot",a)
    a=re.sub(r"[-{}\"/@;:<>()+?.,'!&*^#]"," ",a)
    return a
cq=[]
ca=[]
for i in question:
    cq.append(clean(i))
for i in answer:
    ca.append(clean(i))
#count words
countword={}
for i in cq:
    for j in i.split():
        if j not in countword:
            countword[j]=1
        else:
            countword[j]+=1
for i in ca:
    for j in i.split():
        if j not in countword:
            countword[j]=1
        else:
            countword[j]+=1
    
thresold=20
q2int={}
a2int={}
num=1
for i,j in countword.items():
    if j>thresold:
        q2int[i]=num
        num+=1
num=1
for i,j in countword.items():
    if j>thresold:
        a2int[i]=num
        num+=1
token=['<PAD>','<EOS>','<OUT>','<SOS>']
for i in token:
    q2int[i]=len(q2int)+1
    a2int[i]=len(a2int)+1
reva2int={a:b for b,a in a2int.items()}  
for i in range(len(ca)):
    ca[i]=ca[i]+'<EOS>'
que2intfull=[]
ans2intfull=[]
#8372=q2int['<OUT>']
for que in cq:
    inti=[] 
    for k in que:
        if k in q2int:
            inti.append(q2int[k])
        else:
            inti.append(q2int['<OUT>'])
    que2intfull.append(inti)
for ans in ca:
    inti=[] 
    for k in ans:
        if k in a2int:
            inti.append(a2int[k])
        else:
            inti.append(a2int['<OUT>'])
    ans2intfull.append(inti)

#finals are q2int,a2int,reva2int,que2intfull,ans2intfull
#scq=sorted clean questions
sortedqueint=[]
sortedansint=[]
for i in range(1,27):
    for line in enumerate(que2intfull):
        if len(line[1])==i:
            sortedqueint.append(que2intfull[line[0]])
            sortedansint.append(ans2intfull[line[0]])
    
    
    
######################################################################################
######################################################################################
def modelinput():
    inputs=tf.placeholder(tf.int32,[None,None],name='input')
    targets=tf.placeholder(tf.int32,[None,None],name='target')
    lr=tf.placeholder(tf.float32,name='learning_rate')
    keep_prob=tf.placeholder(tf.float32,name='keep_prob')
    return inputs,targets,lr,keep_prob
def preproccess(target,word2int,batchsize):
    left_side=tf.fill([batchsize,1],word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batchsize, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets
#achitecure is two fold encoder and decoder
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state

#training the decoder
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions

# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions

# Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preproccess(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
#now we will use the functions created above
epochs=100#the one whole proccees of getting input to backpropagation
#use less epoch if take more time
batch_size=64 # try 128 if this works
rnn_size=512
num_layers=3 #no. of layers in both decoder and encoder
encoding_embedding_size= 512#no of column in embedding matrixcand each line - the text of question or naswer
decoding_embedding_size=512
learning_rate=0.01#the rate at which it learn neither to fast  nor low as if fast learn early or if low learn slowly
learning_rate_decay= 0.9# the rate at which it decay as to learn better and also in lstm
#minimum learning rate so that it can go below it
min_learning_rate=0.0001
keep_probability=0.5 #keep_probe=1- dropout rate and in trraining we apply dropout but in testing we didint we want alll neurons to be present at that time.
#above are called hpyerparameter as this control the accuracy of the neural networks
#we reset th-e tenserflow grpah and create a session in which we work
tf.reset_default_graph()
session =tf.InteractiveSession()
#we load the inputs by he hyperparameters
inputs,targets,lr,keep_prob= modelinput()
#we will set sequence length to maximum length
sequence_length=tf.placeholder_with_default(25,None,name='sequence_length')
#we want the shape of inputs tensor
input_shape=tf.shape(inputs)
#now we will get the predictions 
training_predictions,test_predictions=seq2seq_model(tf.reverse(inputs,[-1]),targets,keep_prob,batch_size,sequence_length,len(a2int),len(q2int),encoding_embedding_size,decoding_embedding_size,rnn_size,num_layers,q2int)
#we apply gradientclippling to reducee expoldinng problem vanishing problem
#loss error
#optimizer atom onebest
with tf.name_scope("optimization"):
    loss_error=tf.contrib.seq2seq.sequence_loss(training_predictions,targets,tf.ones([input_shape[0],sequence_length]))
    optimizer=tf.train.AdamOptimizer(learning_rate)
    #compute gradient
    gradients=optimizer.compute_gradients(loss_error)
    clipped_gradients= [(tf.clip_by_value(a,-5.,5.),b) for a,b,in gradients if a is not None]
    optimizer_gradient_clipping=optimizer.apply_gradients(clipped_gradients)

#we add [padding]?
#ass both question and answer should have samen length so we saddt thses
#i am mohit =<pad> he is
def apply_padding(batch_of_seq,word2int):
    max_seq = max([len(i) for i in batch_of_seq])
    return [i+[word2int['<PAD>']]*(max_seq-len(i))for i in batch_of_seq]
def spilt_into_batches(que,ans,batch_size):
    for i in range(len(que)//batch_size):
        si=i*batch_size
        queinbac=que[si:si+batch_size]
        ansinbac=ans[si:si+batch_size]
        pad_que_in_bac=np.array(apply_padding(queinbac,q2int))
        pad_ans_in_bac=np.array(apply_padding(ansinbac,a2int))
        yield pad_que_in_bac,pad_ans_in_bac
total_validation_list=int(len(sortedansint)*0.15)
training_question_set=sortedqueint[total_validation_list:]
training_answer_set=sortedansint[total_validation_list:]
validation_question_set=sortedqueint[:total_validation_list]
validation_answer_set=sortedansint[:total_validation_list]
        
#########################################################
###Training
batch_index_check_training_loss=100 #we will chcek trainng loss every 100 batch after
batch_index_check_validation_loss=(len(training_question_set)//batch_size)//2 -1#we will check it at halfway end of epoch
total_training_loss_error=0#use to compute the sum of training losses every 100 batch
list_validation_loss_error=[]#if we reach a loss which is min of all 
early_stopping_check=0##each time the validation didnt improve we increament and stop when i reachh stop
early_stopping_stop=1000#chose 100 better
checkpoint="./chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())#we initaisaize all global variables
for epoch in range(1,epochs+1):
    for batch_index,(padded_questions_in_batch,padded_answers_in_batch) in enumerate(spilt_into_batches(training_question_set,training_answer_set,batch_size)):
        starting_time=time.time()
        _,batch_training_loss_error=session.run([optimizer_gradient_clipping,loss_error],{inputs:padded_questions_in_batch,targets:padded_answers_in_batch,lr:learning_rate,sequence_length:padded_answers_in_batch.shape[1],keep_prob:keep_probability})
        
        total_training_loss_error+=batch_training_loss_error
        ending_time=time.time()
        #training time
        batch_time=ending_time-starting_time
        print('Working')
        if batch_index % batch_index_check_training_loss==0:
            print("Epoch:{:>3}/{},Batch:{:>4},Training Loss error :{:>6.3f},Training time on 100 : {:f} seconds".format(epoch,epochs,batch_index,len(training_question_set) // batch_size,total_training_loss_error/batch_index_check_training_loss, int((batch_time*batch_index_check_training_loss))))
            total_training_loss_error=0
        if batch_index%batch_index_check_validation_loss==0 and batch_index>0:#reach halfway
            total_validation_loss_error=0
            starting_time=time.time()
            for batch_index_validation,(padded_questions_in_batch,padded_answers_in_batch) in enumerate(spilt_into_batches(validation_question_set,validation_answer_set,batch_size)):
                batch_validation_loss_error=session.run(loss_error,{inputs:padded_questions_in_batch,targets:padded_answers_in_batch,lr:learning_rate,sequence_length:padded_answers_in_batch.shape[1],keep_prob:1})
                total_validation_loss_error+=batch_validation_loss_error
            ending_time=time.time()
            #training time
            batch_time=ending_time-starting_time
            average_validation_loss_error=total_validation_loss_error/len(validation_question_set)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate*=learning_rate_decay
            if learning_rate<min_learning_rate:
                learning_rate=min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error<=min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check=0
                saver=tf.train.Saver()
                saver.save(session,checkpoint)
            else:
                print('Sorry I do not speak better, I need ro practice more.')
                early_stopping_check+=1
                if early_stopping_check>=early_stopping_stop:
                    break
    if early_stopping_check>=early_stopping_stop:
        print('My apologies,I cannot speak better anymore.This is the best I can do')
        break
print('Game Over')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

