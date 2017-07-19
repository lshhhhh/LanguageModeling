# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
import reader
import helpers
tf.set_random_seed(777)

class SeriesPredictor:
    def __init__(self, batch_size, seq_size, dic_size, hidden_size, embedding_size, learning_rate):
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.dic_size = dic_size
        self.hidden_size = hidden_size

        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.int32, [None, None])
        self.w = tf.Variable(tf.random_normal([hidden_size, dic_size]))
        self.b = tf.Variable(tf.random_normal([dic_size]))
        
        self.embeddings = tf.Variable(tf.random_uniform([dic_size, embedding_size], -1.0, 1.0), 
                                      dtype=tf.float32)
        self.x_embedded = tf.nn.embedding_lookup(self.embeddings, self.x)

        self.loss = tf.reduce_mean(
            tf.contrib.seq2seq.sequence_loss(
                logits=self.model(), targets=self.y, 
                weights=tf.ones([batch_size, tf.reduce_max(seq_size)], dtype=tf.float32)))
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

    def model(self):
        cell = rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        initial_state = cell.zero_state(self.batch_size, tf.float32)

        outputs, states = tf.nn.dynamic_rnn(
            cell=cell, inputs=self.x_embedded, sequence_length=self.seq_size, 
            initial_state=initial_state, dtype=tf.float32, time_major=True)

        outputs = tf.reshape(outputs, [-1, self.hidden_size])
        logits = tf.matmul(outputs, self.w) + self.b
        logits = tf.reshape(logits, [-1, self.batch_size, self.dic_size])
        return logits  

    def train(self, train_x, train_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                _, mse = sess.run([self.train_op, self.loss], feed_dict={self.x: train_x, self.y: train_y})
            save_path = self.saver.save(sess, './model.ckpt')

    def test(self, test_x):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess, './model.ckpt')
            output = sess.run(self.model(), feed_dict={self.x: test_x})  
            result = np.argmax(output, axis=2)
            return result

if __name__ == '__main__':
    '''
    s1 = "안녕 나@@ 는 다나@@ 야. 오늘@@ 도 좋@@은 하루 보내.".split(' ')
    s2 = "행복@@ 하다. 오늘@@ 도 역시!".split(' ')
    s3 = "오늘@@ 도 행복@@ 한 하루@@ 다.".split(' ')
    sample = []
    sample.append(s1); sample.append(s2); sample.append(s3)
    for s in sample:
        s.insert(0, '<SOS>')
        s.append('<EOS>')
    '''
    padding_token = '<PAD>'
    sentence_start_token = '<SOS>'
    sentence_end_token = '<EOS>'
    unknown_token = '<UNK>'
    
    sentence_list = reader.read_file('./data/3.tok.bpe')
    word2idx, idx2word = reader.match_word_idx(sentence_list)
    print(word2idx)
    print(idx2word)
    sample = []
    for s in sentence_list:
        #sample.append([w if w in word2idx else unknown_token for w in s])
        sample.append(s)
    sample_idx = []
    for s in sample:
        sample_idx.append([word2idx[w] for w in s])
    print('SAMPLE: ', sample)
    print('WORD DIC: ', word2idx)

    batch_size = len(sample)
    dic_size = len(word2idx)
    hidden_size = 32
    embedding_size = 64
    learning_rate = 0.1
    
    x_data = []; y_data = []
    for s in sample_idx:
        x_data.append(s[:-1])
        y_data.append(s[1:])
    x_data, seq_size = helpers.batch(x_data)
    y_data, _ = helpers.batch(y_data) 
    print('BATCH SIZE: ', batch_size)
    print('SEQ SIZE: ', seq_size)
    print('DIC SIZE: ', dic_size)
    print('HIDDEN SIZE: ', hidden_size)
    print('EMBED SIZE: ', embedding_size)
    print('LEARN RATE: ', learning_rate)
    print('X DATA:'); print(x_data)
    print('Y DATA:'); print(y_data)
     
    predictor = SeriesPredictor(
        batch_size=batch_size, seq_size=seq_size, dic_size=dic_size,
        hidden_size=hidden_size, embedding_size=embedding_size, learning_rate=learning_rate)

    train_x = x_data
    train_y = y_data
    predictor.train(train_x, train_y)

    test_x = x_data
    result = predictor.test(test_x)
    result = np.transpose(result)
    result_str = []
    for s in result:
        result_str.append(' '.join([idx2word[i] for i in s]))
    print('Y DATA:'); print(y_data)
    print('RESULT:'); print(result)
    print('RESULT STR: ', result_str)
    
