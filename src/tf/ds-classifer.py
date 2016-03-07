import tensorflow as tf
import time
from tensorflow.models.rnn import rnn
import numpy as np
from collections import defaultdict
import sys
from random import shuffle
import subprocess
import os
#
#         Params
#
flags = tf.flags
flags.DEFINE_string("in_file", 'data/merge_2013.tab_min25.ints', "input file")
# tac related
flags.DEFINE_string("int_file", 'data/candidates/candidates_2012.wc.ints', "int mapped input file")
flags.DEFINE_string("candidate_file", 'data/candidates/candidates_2012', "original candidate file")
flags.DEFINE_string("result_dir", 'results/tmp', "output for scored candidate file")

flags.DEFINE_integer("gpuid", 0, "gpu id to use")
flags.DEFINE_integer("seed", 0, "random seed")
flags.DEFINE_integer("dev_samples", 1024, "number of training instances to hold out to calculate dev accuracy")
flags.DEFINE_integer("pad_token", 2, "int mapping for pad token")
flags.DEFINE_integer("batch_size", 1000, "max mini batch size")
flags.DEFINE_integer("word_dim", 25, "dimension for word embeddings")
flags.DEFINE_integer("hidden_dim", 10, "dimension for hidden state")
flags.DEFINE_integer("max_grad_norm", 100, "maximum gradient norm")
flags.DEFINE_integer("num_layers", 1, "number of layers for network")
flags.DEFINE_integer("max_epoch", 25, "numbechange dr of epochs to run for")
flags.DEFINE_integer("tac_eval_freq", 5, "run tac evaluation every kth iteration")
flags.DEFINE_integer("seq_len", 50, "max length of token sequences")
flags.DEFINE_boolean("bi", True, "Use bi-directional lstm")
flags.DEFINE_float("lr", .01, "initial learning rate")
flags.DEFINE_float("lr_decay", .01, "learning rate decay")
flags.DEFINE_float("dropout", .1, "dropout probability")
FLAGS = flags.FLAGS

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)
is_training = True
gpu_mem_fraction = .33


#
#          Data
#
def read_int_file(int_file):
    ep_pat_map = defaultdict(list)
    l_size, v_size, line_count, too_long = 0, 0, 0, 0
    x, y = [], []
    with open(int_file) as f:
        for line in f:
            line_count += 1
            e1, e2, ep, pattern, tokens, label = line.strip().split('\t')
            label = int(label) - 1
            l_size = max(l_size, label + 1)
            token_list = map(int, tokens.strip().split(' '))
            if len(token_list) <= FLAGS.seq_len:
                if len(token_list) < FLAGS.seq_len:
                    token_list += [FLAGS.pad_token] * (FLAGS.seq_len - len(token_list))
                ep_pat_map[ep].append(token_list)
                v_size = max(v_size, max(token_list) + 1)
                x.append(token_list)
                y.append(label)
            else:
                too_long += 1
    print 'Read in ' + str(line_count) + ' lines. ' + str(too_long) + ' lines were greater than max seq length.'
    return x, y, ep_pat_map, l_size, v_size

data_x, data_y, ep_pattern_map, label_size, vocab_size = read_int_file(FLAGS.in_file)

print(str(len(data_x)) + ' examples\t'
      + str(len(ep_pattern_map)) + ' entity pairs\t'
      + str(label_size) + ' labels\t'
      + str(vocab_size) + ' unique tokens')

zipped_data = zip(data_x, data_y)
shuffle(zipped_data)
data_x, data_y = zip(*zipped_data)

# convert data to numpy arrays - labels must be dense one-hot vectors
dense_y = []
for epoch, j in enumerate(data_y):
    dense_y.append([0] * label_size)
    dense_y[epoch][j] = 1
data_y = np.array(dense_y)
data_x = np.array(data_x)
train_x, dev_x = data_x[:-FLAGS.dev_samples], data_x[-FLAGS.dev_samples:]
train_y, dev_y = data_y[:-FLAGS.dev_samples], data_y[-FLAGS.dev_samples:]

#
#       Model stuff
#
with tf.device('/gpu:'+str(FLAGS.gpuid)):
    batch_size = tf.placeholder(tf.float32)
    input_x = tf.placeholder(tf.int32, [None, FLAGS.seq_len], name="input_x")
    input_y = tf.placeholder(tf.float32, [None, label_size], name="input_y")

    with tf.device('/cpu:0'):
        lookup_table = tf.Variable(tf.random_uniform([vocab_size, FLAGS.word_dim], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(lookup_table, input_x)
    inputs = tf.nn.dropout(inputs, 1 - FLAGS.dropout)
    inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, FLAGS.seq_len, inputs)]

    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.hidden_dim, input_size=FLAGS.word_dim)
    if is_training and 1 - FLAGS.dropout < 1:
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1 - FLAGS.dropout)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * FLAGS.num_layers)
    if FLAGS.bi:
        back_cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.hidden_dim, input_size=FLAGS.word_dim)
        if is_training and 1 - FLAGS.dropout < 1:
            back_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1 - FLAGS.dropout)
        back_cell = tf.nn.rnn_cell.MultiRNNCell([back_cell] * FLAGS.num_layers)
        outputs = rnn.bidirectional_rnn(cell, back_cell, inputs, dtype=tf.float32)
        state = outputs[-1] + outputs[len(outputs)/2]
    else:
        outputs, state = rnn.rnn(cell, inputs, dtype=tf.float32)

    # lstm returns [hiddenstate+cell] -- extact just the hidden state
    state = tf.slice(state, [0, 0], tf.cast(tf.pack([batch_size, FLAGS.hidden_dim]), tf.int32))
    softmax_w = tf.get_variable("softmax_w", [FLAGS.hidden_dim, label_size])
    softmax_b = tf.get_variable("softmax_b", [label_size])

    logits = tf.nn.xw_plus_b(state, softmax_w, softmax_b, name="logits")
    # training loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits, input_y)
    _cost = tf.reduce_sum(loss) / batch_size

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(_cost, tvars, aggregation_method=2), FLAGS.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(FLAGS.lr)
    _train_op = optimizer.apply_gradients(zip(grads, tvars))

    # eval
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#
#           Train
#
class BatchIter:
    def __init__(self, data_array, size):
        self.data_array = data_array
        self.num_rows = data_array.shape[0]
        self.batch_size = size
        self.start_idx = 0

    def __iter__(self):
        self.start_idx = 0
        return self

    def next(self):
        if self.start_idx >= self.num_rows:
            raise StopIteration
        else:
            end_idx = min(self.start_idx + self.batch_size, self.num_rows)
            to_return = self.data_array[self.start_idx:end_idx]
            self.start_idx = end_idx
            return to_return


# score tac candidate file
def score_tac(_epoch):
    tac_x, tac_y, tac_ep_pattern_map, _, _ = read_int_file(FLAGS.int_file)
    dense_tac_y = []
    for i, j in enumerate(tac_y):
        dense_tac_y.append([0] * label_size)
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
        label_scores = session.run(logits, feed_dict={input_x: x, input_y: y, batch_size:len(x)})
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
        subprocess.Popen('bin/tac-evaluation/tune-thresh.sh 2012 ' + scored_candidate + ' ' + FLAGS.result_dir + '/tuned_' + str(_epoch) + ' &', shell=True)

with tf.Graph().as_default() and tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True,
                              gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_fraction))) as session:
    tf.initialize_all_variables().run()

    for epoch in range(1, FLAGS.max_epoch):
        is_training = False
        if FLAGS.dev_samples > 0:
            accuracies = [session.run(accuracy, feed_dict={input_x: x, input_y: y, batch_size: len(x)})
                          for x, y in zip(BatchIter(dev_x, FLAGS.batch_size), BatchIter(dev_y, FLAGS.batch_size))]
            print '\nAccuracy : ' + str(reduce(lambda a, b: a+b, accuracies) / len(accuracies))

        if epoch % FLAGS.tac_eval_freq == 0:
            print('\nScoring tac candidate file :' + FLAGS.candidate_file)
            score_tac(epoch)

        print 'Training - epoch : ' + str(epoch)
        FLAGS.lr_decay = FLAGS.lr_decay ** max(epoch - FLAGS.max_epoch, 0.0)
        # train_x, train_y = shuffle_data(train_x, train_y)
        costs = []
        total_cost = 0
        start_time = time.time()
        is_training = True
        # shuffle training data
        p = np.random.permutation(len(train_x))
        train_x, train_y = train_x[p], train_y[p]

        for step, (x, y) in enumerate(zip(BatchIter(train_x, FLAGS.batch_size), BatchIter(train_y, FLAGS.batch_size))):
            if int(np.sum(y)) == 0:
                print 'no label!', np.sum(x), np.sum(y)
            else:
                # print len(x)
                cost, _ = session.run([_cost, _train_op], feed_dict={input_x: x, input_y: y, batch_size: len(x)})
                costs.append(cost)
                total_cost += cost
                exp_per_sec = FLAGS.batch_size * ((time.time() - start_time) * 1000 / (step + 1))
                sys.stdout.write('\r{:4.3f} last err, {:4.3f} avg err, {:2.2f} % done, examples/sec {:0.3f}'
                     .format(cost, total_cost / len(costs), (100*step*FLAGS.batch_size/float(len(train_x))), exp_per_sec))
                sys.stdout.flush()
