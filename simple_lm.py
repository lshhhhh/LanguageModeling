# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
tf.set_random_seed(777)

class SeriesPredictor:
    def __init__(self, batch_size, seq_size, dic_size, hidden_size, embedding_size, learning_rate):
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.dic_size = dic_size
        self.hidden_size = hidden_size

        self.x = tf.placeholder(tf.int32, [None, seq_size])
        self.y = tf.placeholder(tf.int32, [None, seq_size])
        self.w = tf.Variable(tf.random_normal([hidden_size, dic_size]))
        self.b = tf.Variable(tf.random_normal([dic_size]))
        
        self.embeddings = tf.Variable(tf.random_uniform([dic_size, embedding_size], -1.0, 1.0), 
                                      dtype=tf.float32)
        self.x_embedded = tf.nn.embedding_lookup(self.embeddings, self.x)

        self.loss = tf.reduce_mean(
            tf.contrib.seq2seq.sequence_loss(
                logits=self.model(), targets=self.y, 
                weights=tf.ones([batch_size, seq_size], dtype=tf.float32)))
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

    def model(self):
        cell = rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        initial_state = cell.zero_state(self.batch_size, tf.float32)

        outputs, states = tf.nn.dynamic_rnn(
            cell, self.x_embedded, sequence_length=[self.seq_size], 
            initial_state=initial_state, dtype=tf.float32)

        tf.Print(outputs, [outputs])
        output = tf.reshape(outputs, [-1, self.hidden_size])
        logits = tf.matmul(output, self.w) + self.b
        outputs = tf.reshape(logits, [self.batch_size, self.seq_size, self.dic_size])
        return outputs  

    def train(self, train_x, train_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                _, mse = sess.run([self.train_op, self.loss], feed_dict={self.x: train_x, self.y: train_y})
            save_path = self.saver.save(sess, 'model.ckpt')

    def test(self, test_x):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess, './model.ckpt')
            output = sess.run(self.model(), feed_dict={self.x: test_x})  
            result = np.argmax(output, axis=2)
            return result

if __name__ == '__main__':
    """
    vocabulary_size = 8000
    unknown_token = "UNK_TOKEN"
    sentence_start_token = "START_TOKEN"
    sentence_end_token = "END_TOKEN"
    sample = sentence_start_token + " Hello, my name is dana. " + sentence_end_token
    """

    sample = "안녕 나@@ 는 다나@@ 야. 오늘@@ 도 좋@@은 하루 보내.".split(' ')
    word_list = list(set(sample))
    word_dic = {w: i for i, w in enumerate(word_list)}
    print('SAMPLE: ', sample)
    print('WORD DIC: ', word_dic)

    batch_size = 1
    seq_size = len(sample) - 1
    dic_size = len(word_dic)
    
    hidden_size = 32
    embedding_size = 64 
    learning_rate = 0.1

    sample_idx = [word_dic[w] for w in sample]
    x_data = [sample_idx[:-1]]
    y_data = [sample_idx[1:]]
    print('X DATA: ', x_data)
    print('Y DATA: ', y_data)

    predictor = SeriesPredictor(
        batch_size=batch_size, seq_size=seq_size, dic_size=dic_size,
        hidden_size=hidden_size, embedding_size=embedding_size, learning_rate=learning_rate)

    train_x = x_data
    train_y = y_data
    predictor.train(train_x, train_y)

    test_x = x_data
    result = predictor.test(test_x)
    result_str = [word_list[i] for i in result[0]]
    print('Y DATA: ', y_data)
    print('RESULT: ', result)
    print('RESULT: ', ' '.join(result_str))

