import os
import subprocess
from random import shuffle
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn

from util import *
import time
import sys


class DSModel:
    def __init__(self, label_size, vocab_size, data_x_seq, data_x_ep, data_y, ep_pattern_map, FLAGS):
        self.ep_pattern_map = ep_pattern_map
        self.label_size = label_size
        self.vocab_size = vocab_size
        self.FLAGS = FLAGS

        # shuffle data
        zipped_data = zip(data_x_seq, data_x_ep, data_y)
        shuffle(zipped_data)
        data_x_seq, data_x_ep, data_y = zip(*zipped_data)

        # convert data to numpy arrays - labels must be dense one-hot vectors
        dense_y = []
        for epoch, j in enumerate(data_y):
            dense_y.append([0] * label_size)
            dense_y[epoch][j] = 1
        data_x_seq, data_x_ep, data_y = np.array(data_x_seq), np.array(data_x_ep), np.array(dense_y)
        self.train_x, self.dev_x = data_x_seq[:-FLAGS.dev_samples], data_x_seq[-FLAGS.dev_samples:]
        self.train_x_ep, self.dev_x_ep = data_x_ep[:-FLAGS.dev_samples], data_x_ep[-FLAGS.dev_samples:]
        self.train_y, self.dev_y = data_y[:-FLAGS.dev_samples], data_y[-FLAGS.dev_samples:]

        # set up graph
        with tf.device('/gpu:'+str(FLAGS.gpuid)):
            self.is_training = tf.placeholder(tf.bool)
            self.batch_size = tf.placeholder(tf.float32)
            self.input_x = tf.placeholder(tf.int32, [None, FLAGS.seq_len], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, label_size], name="input_y")
            self.state = tf.placeholder(tf.float32)


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
            self._state = tf.slice(state, [0, 0], tf.cast(tf.pack([self.batch_size, FLAGS.hidden_dim]), tf.int32))
            softmax_w = tf.get_variable("softmax_w", [FLAGS.hidden_dim, label_size])
            softmax_b = tf.get_variable("softmax_b", [label_size])

            self._logits = tf.nn.xw_plus_b(self.state, softmax_w, softmax_b, name="logits")
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

    def calc_state(self, session, x, is_training):
        state = session.run(self._state, feed_dict={self.input_x: x, self.batch_size: len(x), self.is_training: is_training})
        return state

    def step(self, session, x, y):
        cost, _ = session.run([self._cost, self._train_op], feed_dict={self.state: self.calc_state(session, x, True), self.input_y: y, self.batch_size: len(x), self.is_training: True})
        return cost

    def accuracy(self, session, x, y):
        acc = session.run(self._accuracy, feed_dict={self.state: self.calc_state(session, x, False), self.input_y: y, self.batch_size: len(x), self.is_training: False})
        return acc

    def train_iteration(self, session, epoch):
        print 'Training - epoch : ' + str(epoch)
        self.FLAGS.lr_decay = self.FLAGS.lr_decay ** max(epoch - self.FLAGS.max_epoch, 0.0)
        # train_x, train_y = shuffle_data(train_x, train_y)
        costs = []
        total_cost = 0
        start_time = time.time()
        # shuffle training data
        p = np.random.permutation(len(self.train_y))
        self.train_x, self.train_y = self.train_x[p], self.train_y[p]
        for step, (x, y) in enumerate(
                zip(BatchIter(self.train_x, self.FLAGS.batch_size), BatchIter(self.train_y, self.FLAGS.batch_size))):
            if int(np.sum(y)) == 0:
                print 'no label!', np.sum(x), np.sum(y)
            else:
                # print len(x)
                cost = self.step(session, x, y)
                costs.append(cost)
                total_cost += cost
                exp_per_sec = self.FLAGS.batch_size * ((time.time() - start_time) * 1000 / (step + 1))
                sys.stdout.write('\r{:4.3f} last err, {:4.3f} avg err, {:2.2f} % done, examples/sec {:0.3f}'
                                 .format(cost, total_cost / len(costs),
                                         (100 * step * self.FLAGS.batch_size / float(len(self.train_x))), exp_per_sec))
                sys.stdout.flush()

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
            label_scores = session.run(self._logits, feed_dict={self.state: self.calc_state(session, x, False), self.input_y: y,
                                                                self.batch_size: len(x), self.is_training: False})
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
            subprocess.Popen('bin/tac-evaluation/tune-thresh.sh 2012 ' + scored_candidate + ' ' + FLAGS.result_dir +
                             '/tuned_' + str(epoch) + ' &', shell=True)


class PooledDSModel(DSModel):

    def __init__(self, label_size, vocab_size, data_x_seq, data_x_ep, data_y, ep_pattern_map, FLAGS):
        DSModel.__init__(self, label_size, vocab_size, data_x_seq, data_x_ep, data_y, ep_pattern_map, FLAGS)
        # seperate training data by ep-pattern counts
        self.train_x = defaultdict(list)
        train_y = defaultdict(list)
        for i, ep_patterns in enumerate([ep_pattern_map[ep] for ep in self.train_x_ep]):
            self.train_x[len(ep_patterns)].append(ep_patterns)
            train_y[len(ep_patterns)].append(self.train_y[i])
        self.train_y = train_y

    def train_iteration(self, session, epoch):
        print 'Training - epoch : ' + str(epoch)
        print(len(self.train_x_ep))
        self.FLAGS.lr_decay = self.FLAGS.lr_decay ** max(epoch - self.FLAGS.max_epoch, 0.0)
        costs = []
        total_cost = 0
        start_time = time.time()
        # shuffle training data
        step = 0
        for size, x_size in self.train_x.iteritems():
            print(size/float(len(self.train_x)))
            y_size = self.train_y[size]
            for x, y in zip(PoolBatchIter(x_size, self.FLAGS.batch_size), PoolBatchIter(y_size, self.FLAGS.batch_size)):
                if int(np.sum(y)) == 0:
                    print 'no label!', np.sum(x), np.sum(y)
                else:
                    # print len(x)
                    cost = self.step(session, x, y)
                    costs.append(cost)
                    total_cost += cost
                    exp_per_sec = self.FLAGS.batch_size * ((time.time() - start_time) * 1000 / (step + 1))
                    sys.stdout.write('\r{:4.3f} last err, {:4.3f} avg err, {:2.2f} % done, examples/sec {:0.3f}'
                                     .format(cost, total_cost / len(costs),
                                             (step / float(len(self.train_x_ep))), exp_per_sec))
                    sys.stdout.flush()
                step += len(x)

    ''' x is a list of lists of pattern sequences id's, y is corresponding target labels  '''
    def step(self, session, x, y):
        y_idx = np.argmax(y, 1)
        # flatten x so that we can encode all patterns at the same time using lstm
        flat_x = [pattern for pattern_list in x for pattern in pattern_list]
        state = self.calc_state(session, flat_x, True)

        pooled = self.aggregate_patterns(session, state, x, y_idx)

        cost, _ = session.run([self._cost, self._train_op], feed_dict={self.state: pooled, self.input_y: y, self.batch_size: len(x), self.is_training: True})
        return cost

    def aggregate_patterns(self, session, state, x, y_idx):
        pass


class MaxRelationDSModel(PooledDSModel):
    def aggregate_patterns(self, session, state, x, y_idx):
        logits = session.run(self._logits, feed_dict={self.state: state, self.input_y: y, self.batch_size: len(x), self.is_training: False})
        start = 0
        unflat_x = []
        unflat_logits = []
        for l in [len(_x) for _x in x]:
            unflat_x.append(state[start:start + l])
            unflat_logits.append(logits[start:start + l])
            start += l
        pooled_logits = [[label_score[y_idx[i]] for label_score in label_scores] for i, label_scores in enumerate(unflat_logits)]
        max_pattern = [np.argmax(pattern_scores, 0) for pattern_scores in pooled_logits]
        pooled = np.vstack([encoded_patterns[max_pattern[i]] for i, encoded_patterns in enumerate(unflat_x)])
        return pooled


class MeanPooledDSModel(PooledDSModel):
    def aggregate_patterns(self, session, state, x, y_idx):
        start = 0
        unflat_x = []
        for l in [len(_x) for _x in x]:
            unflat_x.append(state[start:start + l])
            start += l

        # max pool or mean pool
        pooled = np.vstack([np.mean(encoded_patterns, 0) if len(encoded_patterns) > 1 else encoded_patterns for encoded_patterns in unflat_x])
        return pooled


class MaxPooledDSModel(PooledDSModel):
    def aggregate_patterns(self, session, state, x, y_idx):
        start = 0
        unflat_x = []
        for l in [len(_x) for _x in x]:
            unflat_x.append(state[start:start + l])
            start += l

        # max pool
        pooled = np.vstack([np.amax(encoded_patterns, 0) if len(encoded_patterns) > 1 else encoded_patterns for encoded_patterns in unflat_x])
        return pooled
