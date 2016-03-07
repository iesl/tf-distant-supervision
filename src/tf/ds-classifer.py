from collections import defaultdict

import sys
import time

from models import *
from util import *

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
flags.DEFINE_boolean("pool", False, "pool examples by ep when training")
flags.DEFINE_boolean("testing", False, "Take subset of data for fast testing")
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
    print 'Loading data from : ' + int_file
    ep_pat_map = defaultdict(list)
    l_size, v_size, line_count, too_long = 0, 0, 0, 0
    x_seq, x_ep,  y = [], [], []
    with open(int_file) as f:
        for line in f:
            line_count += 1
            if FLAGS.testing and line_count >= 100000:
                break
            else:
                e1, e2, ep, pattern, tokens, label = line.strip().split('\t')
                label = int(label) - 1
                l_size = max(l_size, label + 1)
                token_list = map(int, tokens.strip().split(' '))
                if len(token_list) <= FLAGS.seq_len:
                    if len(token_list) < FLAGS.seq_len:
                        token_list += [FLAGS.pad_token] * (FLAGS.seq_len - len(token_list))
                    v_size = max(v_size, max(token_list) + 1)
                    ep_pat_map[ep].append(token_list)
                    x_seq.append(token_list)
                    x_ep.append(ep)
                    y.append(label)
                else:
                    too_long += 1
    print 'Read in ' + str(line_count) + ' lines. ' + str(too_long) + ' lines were greater than max seq length.'
    return x_seq, x_ep, y, ep_pat_map, l_size, v_size

data_x_seq, data_x_ep, data_y, ep_pattern_map, label_size, vocab_size = read_int_file(FLAGS.in_file)

print(str(len(data_x_seq)) + ' examples\t' + str(len(ep_pattern_map)) + ' entity pairs\t'
       + str(label_size) + ' labels\t' + str(vocab_size) + ' unique tokens')



#
#           Train
#
model = PooledDSModel(FLAGS, label_size, vocab_size) if FLAGS.pool else DSModel(FLAGS, label_size, vocab_size)
train_x, train_y, dev_x, dev_y = model.set_up_data(data_x_seq, data_x_ep, data_y, ep_pattern_map, FLAGS)

with tf.Graph().as_default() and tf.Session(
        config=tf.ConfigProto(intra_op_parallelism_threads=1, allow_soft_placement=True,
                              gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_fraction))
) as session:
    tf.initialize_all_variables().run()

    for epoch in range(1, FLAGS.max_epoch):

        if epoch % FLAGS.tac_eval_freq == 0:
            print('\nScoring tac candidate file :' + FLAGS.candidate_file)
            tac_x_seq, tac_x_ep, tac_y, tac_ep_pattern_map, _, _ = read_int_file(FLAGS.int_file)
            model.score_tac(session, tac_x_seq, tac_y, epoch, FLAGS)

        print 'Training - epoch : ' + str(epoch)
        FLAGS.lr_decay = FLAGS.lr_decay ** max(epoch - FLAGS.max_epoch, 0.0)
        # train_x, train_y = shuffle_data(train_x, train_y)
        costs = []
        total_cost = 0
        start_time = time.time()
        # shuffle training data
        p = np.random.permutation(len(train_x))
        train_x, train_y = train_x[p], train_y[p]

        for step, (x, y) in enumerate(zip(BatchIter(train_x, FLAGS.batch_size), BatchIter(train_y, FLAGS.batch_size))):
            if int(np.sum(y)) == 0:
                print 'no label!', np.sum(x), np.sum(y)
            else:
                # print len(x)
                cost = model.step(session, x, y)
                costs.append(cost)
                total_cost += cost
                exp_per_sec = FLAGS.batch_size * ((time.time() - start_time) * 1000 / (step + 1))
                sys.stdout.write('\r{:4.3f} last err, {:4.3f} avg err, {:2.2f} % done, examples/sec {:0.3f}'
                     .format(cost, total_cost / len(costs), (100*step*FLAGS.batch_size/float(len(train_x))), exp_per_sec))
                sys.stdout.flush()

        if FLAGS.dev_samples > 0:
            accuracies = [model.accuracy(session, x, y) for x, y
                          in zip(BatchIter(dev_x, FLAGS.batch_size), BatchIter(dev_y, FLAGS.batch_size))]
            print '\nAccuracy : ' + str(reduce(lambda a, b: a+b, accuracies) / len(accuracies))
