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


# ----------------------------------------------------------------- answer the inversion
W_inv_s1_x = weight_variable([n_hidden, L])
b_inv_s1_x = bias_variable([L])
W_inv_s1_y = weight_variable([n_hidden, L])
b_inv_s1_y = bias_variable([L])
W_inv_s1_o = weight_variable([n_hidden, 2])
b_inv_s1_o = bias_variable([2])

W_inv_s2_x = weight_variable([n_hidden, L])
b_inv_s2_x = bias_variable([L])
W_inv_s2_y = weight_variable([n_hidden, L])
b_inv_s2_y = bias_variable([L])
W_inv_s2_o = weight_variable([n_hidden, 2])
b_inv_s2_o = bias_variable([2])

VAR_inv += [\
W_inv_s1_x,
b_inv_s1_x,
W_inv_s1_y,
b_inv_s1_y,
W_inv_s1_o,
b_inv_s1_o,
W_inv_s2_x,
b_inv_s2_x,
W_inv_s2_y,
b_inv_s2_y,
W_inv_s2_o,
b_inv_s2_o]

eL = tf.constant(1e-10, shape=[N_BATCH, L])
e2 = tf.constant(1e-10, shape=[N_BATCH, 2])
inv_s1_xs = [tf.nn.softmax(tf.matmul(h, W_inv_s1_x) + b_inv_s1_x)+eL for h in hiddens]
inv_s1_ys = [tf.nn.softmax(tf.matmul(h, W_inv_s1_y) + b_inv_s1_y)+eL for h in hiddens]
inv_s1_os = [tf.nn.softmax(tf.matmul(h, W_inv_s1_o) + b_inv_s1_o)+e2 for h in hiddens]
inv_s2_xs = [tf.nn.softmax(tf.matmul(h, W_inv_s2_x) + b_inv_s2_x)+eL for h in hiddens]
inv_s2_ys = [tf.nn.softmax(tf.matmul(h, W_inv_s2_y) + b_inv_s2_y)+eL for h in hiddens]
inv_s2_os = [tf.nn.softmax(tf.matmul(h, W_inv_s2_o) + b_inv_s2_o)+e2 for h in hiddens]
print "inv_s1_x shape ", show_dim(inv_s1_xs)

inv_s1x_costs = [-tf.reduce_sum(ph_s1_x * tf.log(op)) for op in inv_s1_xs]
inv_s1y_costs = [-tf.reduce_sum(ph_s1_y * tf.log(op)) for op in inv_s1_ys]
inv_s1o_costs = [-tf.reduce_sum(ph_s1_o * tf.log(op)) for op in inv_s1_os]
inv_s2x_costs = [-tf.reduce_sum(ph_s2_x * tf.log(op)) for op in inv_s2_xs]
inv_s2y_costs = [-tf.reduce_sum(ph_s2_y * tf.log(op)) for op in inv_s2_ys]
inv_s2o_costs = [-tf.reduce_sum(ph_s2_o * tf.log(op)) for op in inv_s2_os]
# print "costs shapes ", show_dim(query_pred_costs)
cost_inv = sum(inv_s1x_costs) + sum(inv_s1y_costs) + sum(inv_s1o_costs) +\
           sum(inv_s2x_costs) + sum(inv_s2y_costs) + sum(inv_s2o_costs)

# ----------------------------------------------------------------- answer the query
# put the predicted x, y, o back into here
W_query1 = weight_variable([n_hidden + L + L + 4 * L + 2 + 2, n_pred_hidden])
b_query1 = bias_variable([n_pred_hidden])
W_query2 = weight_variable([n_pred_hidden, 2])
b_query2 = bias_variable([2])
VAR_pred += [W_query1, b_query1, W_query2, b_query2]

hiddens_zip = zip(hiddens, inv_s1_xs, inv_s1_ys, inv_s1_os, inv_s2_xs, inv_s2_ys, inv_s2_os)
hiddens_zip = [tf.concat(1, x) for x in hiddens_zip]

hidden_cat_query = [tf.nn.relu(\
  tf.matmul(tf.concat(1, [ph_new_ob_x, ph_new_ob_y, hidden]),W_query1) + b_query1)\
  for hidden in hiddens_zip]

print "hidden_cat_query shape ", show_dim(hidden_cat_query)
query_preds = [tf.nn.softmax(tf.matmul(hcq, W_query2) + b_query2)+e2 for hcq in hidden_cat_query]
print "query_preds shape ", show_dim(query_preds)

query_pred_costs = [-tf.reduce_sum(ph_new_ob_tf * tf.log(op)) for op in query_preds]
print "costs shapes ", show_dim(query_pred_costs)
cost_query_pred = sum(query_pred_costs)


# ------------------------------------------------------------------------ training steps
optimizer = tf.train.RMSPropOptimizer(0.0002)
train_query_pred = optimizer.minimize(cost_query_pred, var_list = VAR_pred)
train_inv = optimizer.minimize(cost_inv, var_list = VAR_inv)
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
    cost_inv_pre = sess.run([cost_inv], feed_dict=feed_dic)[0]
    sess.run([train_inv], feed_dict=feed_dic)
    cost_inv_post = sess.run([cost_inv], feed_dict=feed_dic)[0]
    print "train inv ", cost_inv_pre, " ", cost_inv_post, " ", True if cost_inv_post < cost_inv_pre else False

    # train query prediction
    cost_query_pred_pre = sess.run([cost_query_pred], feed_dict=feed_dic)[0]
    sess.run([train_query_pred], feed_dict=feed_dic)
    cost_query_pred_post = sess.run([cost_query_pred], feed_dict=feed_dic)[0]
    print "train query pred ", cost_query_pred_pre, " ", cost_query_pred_post, " ", True if cost_query_pred_post < cost_query_pred_pre else False


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

