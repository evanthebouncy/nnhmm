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
ph_a_min = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_a_min")
ph_a_max = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_a_max")

ph_obs_x = [tf.placeholder(tf.float32, [N_BATCH, L], 
            name="ph_ob_x"+str(i)) for i in range(OBS_SIZE)]
ph_obs_y = [tf.placeholder(tf.float32, [N_BATCH, L],
            name="ph_ob_y"+str(j)) for j in range(OBS_SIZE)]
ph_obs_tf = [tf.placeholder(tf.float32, [N_BATCH, 2],
            name="ph_ob_tf"+str(k)) for k in range(OBS_SIZE)]

ph_new_ob_x = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_new_ob_x")
ph_new_ob_y = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_new_ob_y")
ph_new_ob_tf = tf.placeholder(tf.float32, [N_BATCH, 2], name="ph_new_ob_tf")

def gen_feed_dict(a_min, a_max,
                  obs_x, obs_y, obs_tf,
                  new_ob_x, new_ob_y, new_ob_tf):
  ret = {}

  ret[ph_a_min] = a_min
  ret[ph_a_max] = a_max

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
VAR_inv = []
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

VAR_inv += lstm_variables
VAR_pred += lstm_variables

# ------------------------------------------------------------------- answer the inversion
W_inv_a_min = weight_variable([n_hidden, L])
b_inv_a_min = bias_variable([L])
W_inv_a_max = weight_variable([n_hidden, L])
b_inv_a_max = bias_variable([L])

VAR_inv += [\
W_inv_a_min,
b_inv_a_min,
W_inv_a_max,
b_inv_a_max]

eL = tf.constant(1e-10, shape=[N_BATCH, L])
e2 = tf.constant(1e-10, shape=[N_BATCH, 2])
inv_a_mins = [tf.nn.softmax(tf.matmul(h, W_inv_a_min) + b_inv_a_min)+eL for h in hiddens]
inv_a_maxs = [tf.nn.softmax(tf.matmul(h, W_inv_a_max) + b_inv_a_max)+eL for h in hiddens]
print "inv_s1_x shape ", show_dim(inv_a_mins)

inv_a_min_costs = [-tf.reduce_sum(ph_a_min * tf.log(op), 1) for op in inv_a_mins]
inv_a_max_costs = [-tf.reduce_sum(ph_a_max * tf.log(op), 1) for op in inv_a_maxs]

inv_a_costs = [x[0]+x[1] for x in zip(inv_a_min_costs, inv_a_max_costs)]

print "costs shapes ", show_dim(inv_a_costs)
cost_inv = sum([tf.reduce_sum(x) for x in inv_a_costs])

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

print "query_hidden shape ", show_dim(query_hidden)
query_pred = [tf.nn.softmax(tf.matmul(hcq, W_score2) + b_score2) + e2 for hcq in query_hidden]
print "query pred shape ", show_dim(query_pred)

print "tf shape ", show_dim(ph_new_ob_tf)

query_cost = [-tf.reduce_sum(tf.log(op) * ph_new_ob_tf) for op in query_pred]

print "query cost shape ", show_dim(query_cost)
cost_pred = sum(query_cost)


# --------------------------------------------------- compute mutual information of a new query
# get true/false distribution of the new observation...
new_ob_prob = query_pred
print "new ob prob shape ", show_dim(new_ob_prob)

# create artificial true/false for the new observation as potential inputs
ttrue = tf.tile(tf.constant([[1.0, 0.0]]), [N_BATCH,1])
ffalse = tf.tile(tf.constant([[0.0, 1.0]]), [N_BATCH,1])
print "ttrue shape ", show_dim(ttrue)

pretend_true_input = tf.concat(1, [ph_new_ob_x, ph_new_ob_y, ttrue])
pretend_false_input = tf.concat(1, [ph_new_ob_x, ph_new_ob_y, ffalse])

pretend_true_states = []
pretend_false_states = []
with tf.variable_scope("LSTM") as scope:
  for hh in hiddens:
    scope.reuse_variables()
    _, true_state = lstm(pretend_true_input, hh)
    pretend_true_states.append(true_state)
    scope.reuse_variables()
    _, false_state = lstm(pretend_false_input, hh)
    pretend_false_states.append(false_state)

print "ttrue state shape ", show_dim(pretend_true_states)

true_preds_min = [tf.nn.softmax(tf.matmul(h, W_inv_a_min) + b_inv_a_min)+eL\
              for h in pretend_true_states]
true_preds_max = [tf.nn.softmax(tf.matmul(h, W_inv_a_max) + b_inv_a_max)+eL\
              for h in pretend_true_states]
false_preds_min = [tf.nn.softmax(tf.matmul(h, W_inv_a_min) + b_inv_a_min)+eL\
               for h in pretend_false_states]
false_preds_max = [tf.nn.softmax(tf.matmul(h, W_inv_a_max) + b_inv_a_max)+eL\
               for h in pretend_false_states]

print "true preds shape ", show_dim(true_preds_min)

true_entropy_min = [-tf.reduce_sum(tf.log(x) * x, 1) for x in true_preds_min]
true_entropy_max = [-tf.reduce_sum(tf.log(x) * x, 1) for x in true_preds_max]
true_entropy = [tf.reshape(x[0] + x[1], [N_BATCH, 1])\
                for x in zip(true_entropy_min, true_entropy_max)]

false_entropy_min = [-tf.reduce_sum(tf.log(x) * x, 1) for x in false_preds_min]
false_entropy_max = [-tf.reduce_sum(tf.log(x) * x, 1) for x in false_preds_max]
false_entropy = [tf.reshape(x[0] + x[1], [N_BATCH, 1])\
                 for x in zip(false_entropy_min, false_entropy_max)]

print "true entropy shape ", show_dim(true_entropy)

true_false_ent = [tf.concat(1, x) for x in zip(true_entropy, false_entropy)]
print "true_false shape ", show_dim(true_false_ent)

true_false_weighted = [x[0] * x[1] for x in zip(new_ob_prob, true_false_ent)]
print "true false weighted shape ", show_dim(true_false_weighted)

conditional_ent = [tf.reduce_sum(tfw, 1) for tfw in true_false_weighted]
print "conditional ent ", show_dim(conditional_ent)

# ------------------------------------------------------------------------ training steps
optimizer = tf.train.RMSPropOptimizer(0.0002)
train_inv = optimizer.minimize(cost_inv, var_list = VAR_inv)
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
    a_min, a_max, obs_x, obs_y, obs_tfs, xxx,yyy,tftf = gen_data()
    feed_dic = gen_feed_dict(a_min, a_max, obs_x, obs_y, obs_tfs, xxx,yyy,tftf)

    # train query prediction
    cost_inv_pre = sess.run([cost_inv], feed_dict=feed_dic)[0]
    sess.run([train_inv], feed_dict=feed_dic)
    cost_inv_post = sess.run([cost_inv], feed_dict=feed_dic)[0]
    print "train inv ", cost_inv_pre, " ", cost_inv_post, " ", True if cost_inv_post < cost_inv_pre else False

    # train query prediction
    cost_pred_pre = sess.run([cost_pred], feed_dict=feed_dic)[0]
    sess.run([train_pred], feed_dict=feed_dic)
    cost_pred_post = sess.run([cost_pred], feed_dict=feed_dic)[0]
    print "train pred ", cost_pred_pre, " ", cost_pred_post, " ", True if cost_pred_post < cost_pred_pre else False


    if i % 100 == 0:
      print [x[0] for x in sess.run(query_pred, feed_dict=feed_dic)]
      save_path = saver.save(sess, save_loc)
      print("Model saved in file: %s" % save_path)


# load the model and give back a session
def load_model(saved_loc):
  saver = tf.train.Saver()
  sess = tf.Session()
  saver.restore(sess, saved_loc)
  print("Model restored.")
  return sess

