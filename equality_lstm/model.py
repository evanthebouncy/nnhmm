import tensorflow as tf
import numpy as np
from data import *

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# set up placeholders
ph_obs_x = [tf.placeholder(tf.float32, [N_BATCH, L], 
            name="ph_ob_x"+str(i)) for i in range(OBS_SIZE)]
ph_obs_y = [tf.placeholder(tf.float32, [N_BATCH, L],
            name="ph_ob_y"+str(j)) for j in range(OBS_SIZE)]
ph_obs_tf = [tf.placeholder(tf.float32, [N_BATCH, 2],
            name="ph_ob_tf"+str(k)) for k in range(OBS_SIZE)]

ph_new_ob_x = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_new_ob_x")
ph_new_ob_y = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_new_ob_y")
ph_new_ob_tf = tf.placeholder(tf.float32, [N_BATCH, 2], name="ph_new_ob_tf")

def gen_feed_dict(obs_x, obs_y, obs_tf,
                  new_ob_x, new_ob_y, new_ob_tf):
  ret = {}

  for a, b in zip(ph_obs_x, obs_x):
    ret[a] = b
  for a, b in zip(ph_obs_y, obs_y):
    ret[a] = b
  for a, b in zip(ph_obs_tf, obs_tf):
    ret[a] = b

  ret[ph_new_ob_x] = new_ob_x
  ret[ph_new_ob_y] = new_ob_y
  ret[ph_new_ob_tf] = new_ob_tf

  return ret

# some constants
n_hidden = 400
n_score_hidden = 400

# a list of variables for different tasks
VAR_pred = []

# --------------------------------------------------------------------- initial hidden h(X)
# set up weights for input outputs!
state = tf.zeros([N_BATCH, n_hidden])

# ------------------------------------------------------------------ convolve in the observations

# initialize some weights
# stacked lstm
lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(100), tf.nn.rnn_cell.LSTMCell(100)])

hiddens = [state]

with tf.variable_scope("LSTM") as scope:
  for i in range(OBS_SIZE):
    if i > 0:
      scope.reuse_variables()
    cell_input = tf.concat(1, [ph_obs_x[i], ph_obs_y[i], ph_obs_tf[i]])
    output, state = lstm(cell_input, state)
    hiddens.append(state)

lstm_variables = [v for v in tf.all_variables()
                    if v.name.startswith("LSTM")]

print lstm_variables

VAR_pred += lstm_variables

# --------------------------------------------------------------------- make the future prediction
# # get the entropy for each inversion
# ent_a_min = [-tf.reduce_sum(tf.log(x) * x, 1) for x in inv_a_mins]
# ent_a_max = [-tf.reduce_sum(tf.log(x) * x, 1) for x in inv_a_maxs]
# print "entropy shape ", show_dim(ent_a_min)
# ent_sum = [x[0] + x[1] for x in zip(ent_a_min, ent_a_max)]
# ent_diffs = [(ent_sum[i] - ent_sum[i+1]) for i in range(0, len(ent_sum) - 1)]
# 
# # get the diff of costs, cost will decrease over time (in general)
# print "cost_diff shapes ", show_dim(ent_diffs)

# put the predicted x, y, o back into here
W_score1 = weight_variable([n_hidden + L + L, n_score_hidden])
b_score1 = bias_variable([n_score_hidden])
W_score2 = weight_variable([n_score_hidden, 2])
b_score2 = bias_variable([2])
VAR_pred += [W_score1, b_score1, W_score2, b_score2]

hiddens_zip = [tf.concat(1, [hh, ph_new_ob_x, ph_new_ob_y]) for hh in hiddens]

print "zippy shape ", show_dim(hiddens_zip)

query_hidden = [tf.nn.relu(\
  tf.matmul(hh ,W_score1) + b_score1)\
  for hh in hiddens_zip]

e2 = tf.constant(1e-10, shape=[N_BATCH, 2])

print "query_hidden shape ", show_dim(query_hidden)
query_pred = [tf.nn.softmax(tf.matmul(hcq, W_score2) + b_score2) + e2 for hcq in query_hidden]
print "query pred shape ", show_dim(query_pred)

print "truefalse shape ", show_dim(ph_new_ob_tf)

query_cost = [-tf.reduce_sum(tf.log(op) * ph_new_ob_tf) for op in query_pred]

print "query cost shape ", show_dim(query_cost)
cost_pred = sum(query_cost)

# ------------------------------------------------------------------------ training steps
optimizer = tf.train.RMSPropOptimizer(0.0002)
train_pred = optimizer.minimize(cost_pred, var_list = VAR_pred)
# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# ------------------------------------------------------------------------- using the model!

# train the model and save checkpt to a file location
def train_model(save_loc):
  # Launch the graph.
  sess = tf.Session()
  sess.run(init)
  saver = tf.train.Saver()


  for i in range(5000001):
    obs_x, obs_y, obs_tfs, xxx,yyy,tftf, orig = gen_data()
    feed_dic = gen_feed_dict(obs_x, obs_y, obs_tfs, xxx,yyy,tftf)

    # train query prediction
    cost_pred_pre = sess.run([cost_pred], feed_dict=feed_dic)[0]
    sess.run([train_pred], feed_dict=feed_dic)
    cost_pred_post = sess.run([cost_pred], feed_dict=feed_dic)[0]
    print "train pred ", cost_pred_pre, " ", cost_pred_post, " ", True if cost_pred_post < cost_pred_pre else False


    if i % 100 == 0:
      seen_obs = zip([np.argmax(x[0]) for x in obs_x], [np.argmax(y[0]) for y in obs_y], [trufal[0] for trufal in obs_tfs])
      preds = [x[0] for x in sess.run(query_pred, feed_dict=feed_dic)]
      print "truth pred"
      print np.argmax(xxx[0]), " ", np.argmax(yyy[0]), " ", tftf[0], np.argmax(tftf[0])
      print "preds "
      for idx, pp in enumerate(preds[1:]):
        print seen_obs[idx], " ", pp, np.argmax(pp)
      save_path = saver.save(sess, save_loc)
      print("Model saved in file: %s" % save_path)


# load the model and give back a session
def load_model(saved_loc):
  saver = tf.train.Saver()
  sess = tf.Session()
  saver.restore(sess, saved_loc)
  print("Model restored.")
  return sess

