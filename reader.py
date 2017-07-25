import re
import nltk
#import tensorflow as tf
#from tensorflow.contrib.data.python.ops.dataset_ops import TextLineDataset

padding_token = '<PAD>'
sentence_start_token = '<SOS>'
sentence_end_token = '<EOS>'
unknown_token = '<UNK>'


def read_file(file_name):
    with open(file_name, 'r') as f:
        sentence_list = []
        while True:
            l = f.readline()
            if not l: break
            s = re.sub('1\n', sentence_end_token, sentence_start_token+' '+l).split(' ')
            sentence_list.append(s)
        return sentence_list


def match_word_idx(sentence_list):
    word_list = []
    for s in sentence_list:
        word_list += s
    word_list = list(set(word_list))
    word_list.remove(sentence_start_token)
    word_list.remove(sentence_end_token)
    
    vocabulary_size = 8000
    freq_dist = nltk.FreqDist(word_list)
    freq_list = freq_dist.most_common(vocabulary_size)
    
    word2idx = {w: i+4 for i, (w, f) in enumerate(freq_list)}
    word2idx[padding_token] = 0
    word2idx[sentence_start_token] = 1
    word2idx[sentence_end_token] = 2
    word2idx[unknown_token] = 3
    
    idx2word = {i: w for w, i in word2idx.items()}
    
    return (word2idx, idx2word)

'''
dataset = TextLineDataset('./data/simple.tok.bpe')
print(dataset.output_types)
print(dataset.output_shapes)
print(dataset.enumerate(start=1).output_types)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    for i in range(100):
        value = sess.run(next_element)
        print(value)
        print(i)
'''
