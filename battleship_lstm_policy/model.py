import tensorflow as tf
import numpy as np
from data import *

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def get_entropy(xx):
  x_shape = xx.get_shape()
  e_small = tf.constant(1e-10, shape=x_shape)
  xx_cc = xx + e_small
  H = -tf.reduce_sum(xx_cc * tf.log(xx_cc), 1)
  return H
 
  
  

# set up placeholders
ph_s1_x = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_s1_x")
ph_s1_y = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_s1_y")
ph_s1_o = tf.placeholder(tf.float32, [N_BATCH, 2], name="ph_s1_o")
ph_s2_x = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_s2_x")
ph_s2_y = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_s2_y")
ph_s2_o = tf.placeholder(tf.float32, [N_BATCH, 2], name="ph_s2_o")

ph_obs_x = [tf.placeholder(tf.float32, [N_BATCH, L], 
            name="ph_ob_x"+str(i)) for i in range(OBS_SIZE)]
ph_obs_y = [tf.placeholder(tf.float32, [N_BATCH, L],
            name="ph_ob_y"+str(j)) for j in range(OBS_SIZE)]
ph_obs_tf = [tf.placeholder(tf.float32, [N_BATCH, 2],
            name="ph_ob_tf"+str(k)) for k in range(OBS_SIZE)]
ph_new_ob_x = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_new_ob_x")
ph_new_ob_y = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_new_ob_y")
ph_new_ob_tf = tf.placeholder(tf.float32, [N_BATCH,2], name="ph_new_ob_tf")

def gen_feed_dict(s1_x, s1_y, s1_o, s2_x, s2_y, s2_o,
                  obs_x, obs_y, obs_tf, 
                  new_ob_x, new_ob_y, new_ob_tf):
  ret = {}

  ret[ph_s1_x] = s1_x
  ret[ph_s1_y] = s1_y
  ret[ph_s1_o] = s1_o
  ret[ph_s2_x] = s2_x
  ret[ph_s2_y] = s2_y
  ret[ph_s2_o] = s2_o

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
n_pred_hidden = 400

# a list of variables for different tasks
VAR_action = []
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

# ----------------------------------------------------------------- answer the query
# put the predicted x, y, o back into here
W_query1 = weight_variable([n_hidden + L + L, n_pred_hidden])
b_query1 = bias_variable([n_pred_hidden])
W_query2 = weight_variable([n_pred_hidden, 2])
b_query2 = bias_variable([2])
VAR_pred += [W_query1, b_query1, W_query2, b_query2]

hidden_cat_query = [tf.nn.relu(\
  tf.matmul(tf.concat(1, [ph_new_ob_x, ph_new_ob_y, hidden]),W_query1) + b_query1)\
  for hidden in hiddens]

e2 = tf.constant(1e-10, shape=[N_BATCH, 2])

print "hidden_cat_query shape ", show_dim(hidden_cat_query)
query_preds = [tf.nn.softmax(tf.matmul(hcq, W_query2) + b_query2)+e2 for hcq in hidden_cat_query]
print "query_preds shape ", show_dim(query_preds)

query_pred_costs = [-tf.reduce_sum(ph_new_ob_tf * tf.log(op)) for op in query_preds]
print "costs shapes ", show_dim(query_pred_costs)
cost_query_pred = sum(query_pred_costs)

# -------------------------------------------------------------------- generate a policy
# create the policy network
W_action_x = weight_variable([n_hidden, L])
W_action_y = weight_variable([n_hidden, L])
b_action_x = bias_variable([L])
b_action_y = bias_variable([L])

VAR_action += [W_action_x, W_action_y, b_action_x, b_action_y]

action_xs = [tf.nn.softmax(tf.matmul(hidden, W_action_x) + b_action_x) for hidden in hiddens]
action_ys = [tf.nn.softmax(tf.matmul(hidden, W_action_y) + b_action_y) for hidden in hiddens]

action_xs = [tf.nn.softmax(ax) for ax in action_xs]
action_ys = [tf.nn.softmax(ay) for ay in action_ys]

# ------------------- add regularization cost to force these things to be more 1 hot
action_xs_H = [tf.reduce_sum(get_entropy(xx)) for xx in action_xs]
action_ys_H = [tf.reduce_sum(get_entropy(yy)) for yy in action_ys]
cost_action_H = sum(action_xs_H)+sum(action_ys_H)

print "action x shape ", show_dim(action_xs)

# feed the confusion into the prediction kekekekeekeke
# zip everything together
confuse_zip = zip(action_xs, action_ys, hiddens)

_confuse = [tf.nn.relu(\
  tf.matmul(tf.concat(1, xxx),W_query1) + b_query1)\
  for xxx in confuse_zip]

print "hidden_cat_query shape ", show_dim(hidden_cat_query)
confuse_preds = [tf.nn.softmax(tf.matmul(hcq, W_query2) + b_query2)+e2 for hcq in _confuse]

print "confuse pred shape ", show_dim(confuse_preds)

confuse_entropys = [get_entropy(xx) for xx in confuse_preds]
print "entropy shape ", show_dim(confuse_entropys)

cost_action = -sum([tf.reduce_sum(xx) for xx in confuse_entropys]) +\
              cost_action_H

# ------------------------------------------------------------------------ training steps
optimizer = tf.train.RMSPropOptimizer(0.0002)
train_query_pred = optimizer.minimize(cost_query_pred, var_list = VAR_pred)
train_action = optimizer.minimize(cost_action, var_list = VAR_action)
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
    s1_x, s1_y, s1_o, s2_x, s2_y, s2_o, obs_x, obs_y, obs_tfs, new_ob_x, new_ob_y, new_ob_tf = gen_data()
    feed_dic = gen_feed_dict(s1_x, s1_y, s1_o, s2_x, s2_y, s2_o, obs_x, obs_y, obs_tfs, new_ob_x, new_ob_y, new_ob_tf)

    # train query prediction
    cost_query_pred_pre = sess.run([cost_query_pred], feed_dict=feed_dic)[0]
    sess.run([train_query_pred], feed_dict=feed_dic)
    cost_query_pred_post = sess.run([cost_query_pred], feed_dict=feed_dic)[0]
    print "train query pred ", cost_query_pred_pre, " ", cost_query_pred_post, " ", True if cost_query_pred_post < cost_query_pred_pre else False

    # train action
    cost_action_pre = sess.run([cost_action], feed_dict=feed_dic)[0]
    sess.run([train_action], feed_dict=feed_dic)
    cost_action_post = sess.run([cost_action], feed_dict=feed_dic)[0]
    print "----------------- - - -- train action ", cost_action_pre, " ", cost_action_post, " ", True if cost_action_post < cost_action_pre else False


    if i % 100 == 0:
      # print "for inversion "
      # ran_predss = sess.run(query_preds, feed_dict=feed_dic)[9]


      print "for query prediction"
      print "query loc "
      print new_ob_x[0]
      print new_ob_y[0]
      print "predicted <===> true"
      ran_predss = sess.run(query_preds, feed_dict=feed_dic)[9]
      total_cor = 0.0
      for haha in zip(ran_predss, new_ob_tf):
        print haha
        if np.argmax(haha[0]) == np.argmax(haha[1]):
          total_cor += 1
      cor_ratio = total_cor / len(ran_predss)
      print "total correct ", cor_ratio

      print "for confusion!"
      action_xxx = sess.run(action_xs, feed_dict=feed_dic)[9]
      action_yyy = sess.run(action_ys, feed_dict=feed_dic)[9]
      act_confused = sess.run(confuse_preds, feed_dict=feed_dic)[9]
      for xxxyyy in zip(action_xxx, action_yyy, act_confused):
        print xxxyyy
        print np.argmax(xxxyyy[0]), np.argmax(xxxyyy[1]), xxxyyy[2]
      
      # something special!
      global RAND_HIT
      NEW_RAND_HIT = 1.0 - cor_ratio
      RAND_HIT = RAND_HIT * 0.9 + NEW_RAND_HIT * 0.1
      print "rand hit now ", RAND_HIT


      save_path = saver.save(sess, save_loc)
      print("Model saved in file: %s" % save_path)


# load the model and give back a session
def load_model(saved_loc):
  saver = tf.train.Saver()
  sess = tf.Session()
  saver.restore(sess, saved_loc)
  print("Model restored.")
  return sess

