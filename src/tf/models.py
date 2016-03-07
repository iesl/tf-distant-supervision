import tensorflow as tf
from tensorflow.models.rnn import rnn
from util import *
import os
import subprocess
import numpy as np


class DSModel:

    def __init__(self, FLAGS, label_size, vocab_size):
        self.label_size = label_size
        self.vocab_size = vocab_size
        with tf.device('/gpu:'+str(FLAGS.gpuid)):
            self.is_training = tf.placeholder(tf.bool)
            self.batch_size = tf.placeholder(tf.float32)
            self.input_x = tf.placeholder(tf.int32, [None, FLAGS.seq_len], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, label_size], name="input_y")

            with tf.device('/cpu:0'):
                lookup_table = tf.Variable(tf.random_uniform([vocab_size, FLAGS.word_dim], -1.0, 1.0))
                inputs = tf.nn.embedding_lookup(lookup_table, self.input_x)
            inputs = tf.nn.dropout(inputs, 1 - FLAGS.dropout)
            inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, FLAGS.seq_len, inputs)]

            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.hidden_dim, input_size=FLAGS.word_dim)
            if self.is_training and 1 - FLAGS.dropout < 1:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1 - FLAGS.dropout)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * FLAGS.num_layers)
            if FLAGS.bi:
                back_cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.hidden_dim, input_size=FLAGS.word_dim)
                if self.is_training and 1 - FLAGS.dropout < 1:
                    back_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1 - FLAGS.dropout)
                back_cell = tf.nn.rnn_cell.MultiRNNCell([back_cell] * FLAGS.num_layers)
                outputs = rnn.bidirectional_rnn(cell, back_cell, inputs, dtype=tf.float32)
                state = outputs[-1] + outputs[len(outputs)/2]
            else:
                outputs, state = rnn.rnn(cell, inputs, dtype=tf.float32)

            # lstm returns [hiddenstate+cell] -- extact just the hidden state
            state = tf.slice(state, [0, 0], tf.cast(tf.pack([self.batch_size, FLAGS.hidden_dim]), tf.int32))
            softmax_w = tf.get_variable("softmax_w", [FLAGS.hidden_dim, label_size])
            softmax_b = tf.get_variable("softmax_b", [label_size])

            self._logits = tf.nn.xw_plus_b(state, softmax_w, softmax_b, name="logits")
            # training loss
            loss = tf.nn.softmax_cross_entropy_with_logits(self._logits, self.input_y)
            self._cost = tf.reduce_sum(loss) / self.batch_size

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars, aggregation_method=2), FLAGS.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(FLAGS.lr)
            self._train_op = optimizer.apply_gradients(zip(grads, tvars))

            # eval
            correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self.input_y, 1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def step(self, session, x, y):
        cost, _ = session.run([self._cost, self._train_op], feed_dict={self.input_x: x, self.input_y: y, self.batch_size: len(x), self.is_training: True})
        return cost

    def accuracy(self, session, x, y):
        acc = session.run(self._accuracy, feed_dict={self.input_x: x, self.input_y: y, self.batch_size: len(x), self.is_training: False})
        return acc




        # score tac candidate file
    def score_tac(self, session, tac_x, tac_y, epoch, FLAGS):
        dense_tac_y = []
        for i, j in enumerate(tac_y):
            dense_tac_y.append([0] * self.label_size)
            dense_tac_y[i][j] = 1
        tac_x, dense_tac_y = np.array(tac_x), np.array(dense_tac_y)
        out_lines, scores = [], []

        # read in original candidate file that we will attach the scores to
        with open(FLAGS.candidate_file) as f:
            out_lines = ['\t'.join([out_prefix, s1, e1, s2, e2])
                         for out_prefix, s1, e1, s2, e2, pattern in [line.strip().rsplit('\t', 5) for line in f]
                         # don't consider the length of the entities when computing sequence length
                         if len(pattern.split(' ')) - (int(e1)-int(s1)-1) - (int(e2)-int(s2)-1) <= FLAGS.seq_len]

        # score each line and attach the score to original file
        offset = 0
        for x, y in zip(BatchIter(tac_x, FLAGS.batch_size), BatchIter(dense_tac_y, FLAGS.batch_size)):
            label_scores = session.run(self._logits, feed_dict={self.input_x: x, self.input_y: y, self.batch_size:len(x), self.is_training: False})
            # take the score of the query label
            scores.extend([l[tac_y[i+offset]] for i, l in enumerate(label_scores)])
            offset += len(x)

        min_score, max_score = min(scores), max(scores)
        delta = max_score - min_score
        scores = [(s-min_score)/delta for s in scores]

        scored_candidate = FLAGS.result_dir+'/scored_'+str(epoch)
        if not os.path.exists(FLAGS.result_dir):
            os.makedirs(FLAGS.result_dir)
        with open(scored_candidate, 'w') as f:
            for score, out_line in zip(scores, out_lines):
                f.write(out_line + '\t' + str(score) + '\n')
            subprocess.Popen('bin/tac-evaluation/tune-thresh.sh 2012 ' + scored_candidate + ' ' + FLAGS.result_dir + '/tuned_' + str(epoch) + ' &', shell=True)



# class BiLSTM(DSModel):
