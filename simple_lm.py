# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
tf.set_random_seed(777)

class SeriesPredictor:
    def __init__(self, seq_size, batch_size, dic_size, hidden_dim=10):
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dic_size = dic_size
        
        self.weights = tf.Variable(tf.random_normal([self.batch_size, self.seq_size]))
        self.x = tf.placeholder(tf.int32, [None, seq_size])
        self.y = tf.placeholder(tf.int32, [None, seq_size])
        self.x_one_hot = tf.one_hot(self.x, self.dic_size)
        #self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        #self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        #self.weights = tf.ones([self.batch_size, self.seq_size], name="W_out")
        #self.x_for_fc = tf.placeholder(tf.float32, [None, hidden_dim])

        self.loss = tf.reduce_mean(
            tf.contrib.seq2seq.sequence_loss(
                logits=self.model(), targets=self.y, weights=self.weights))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss)
        self.saver = tf.train.Saver()

    def model(self):
        cell = rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)
        initial_state = cell.zero_state(self.batch_size, tf.float32)
        outputs, states = tf.nn.dynamic_rnn(
            cell, self.x_one_hot, initial_state=initial_state, dtype=tf.float32)

        #self.x_for_fc = tf.reshape(outputs, [-1, self.hidden_dim])
        #outputs = tf.contrib.layers.fully_connected(self.x_for_fc, self.dic_size, activation_fn=None)
        outputs = tf.reshape(outputs, [self.batch_size, self.seq_size, self.dic_size])
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

    dic_size = len(word_dic)
    hidden_size = len(word_dic)
    num_classes = len(word_dic)
    batch_size = 1
    seq_size = len(sample)-1
    learning_rate = 0.1

    sample_idx = [word_dic[w] for w in sample]
    x_data = [sample_idx[:-1]]
    y_data = [sample_idx[1:]]
    print('X DATA: ', x_data)
    print('Y DATA: ', y_data)

    predictor = SeriesPredictor(
        seq_size=seq_size, batch_size=batch_size, dic_size=dic_size, hidden_dim=hidden_size)

    train_x = x_data
    train_y = y_data
    predictor.train(train_x, train_y)

    test_x = x_data
    result = predictor.test(test_x)
    result_str = [word_list[i] for i in result[0]]
    print('RESULT: ', result)
    print('RESULT: ', ' '.join(result_str))

