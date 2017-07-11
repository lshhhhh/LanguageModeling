# =*- coding: utf-8 -*-

vocabulary_size = 8000
unknown_token = "UNK_TOKEN"
sentence_start_token = "START_TOKEN"
sentence_end_token = "END_TOKEN"

with open("./train.tok.bpe", 'r') as f:
    while True
        line = f.readline()
        if not line: break
        sentence = ["%s %s %s" % (sentence_start_token, line, sentence_end_token)]

