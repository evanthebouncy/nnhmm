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
ph_x_x = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_x_x")
ph_obs_x = [tf.placeholder(tf.float32, [N_BATCH, L], 
            name="ph_ob_x"+str(i)) for i in range(OBS_SIZE)]
ph_obs_tf = [tf.placeholder(tf.float32, [N_BATCH, 2],
            name="ph_ob_tf"+str(k)) for k in range(OBS_SIZE)]
ph_new_ob_x = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_new_ob_x")
ph_new_ob_tf = tf.placeholder(tf.float32, [N_BATCH,2], name="ph_new_ob_tf")

def gen_feed_dict(x_x, obs_x, obs_tf, 
                  new_ob_x, new_ob_tf):
  ret = {}
  for a, b in zip(ph_obs_x, obs_x):
    ret[a] = b
  for a, b in zip(ph_obs_tf, obs_tf):
    ret[a] = b

  ret[ph_x_x] = x_x
  ret[ph_new_ob_x] = new_ob_x
  ret[ph_new_ob_tf] = new_ob_tf
  return ret

# some constants
n_hidden = 400
n_pred_hidden = 400

# a list of variables for different tasks
VAR_inv = []
VAR_pred = []

# ------------------------------------------------------------------ convolve in the observations

state = tf.zeros([N_BATCH, n_hidden])
# initialize some weights
# initialize some weights
# stacked lstm
lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(100), tf.nn.rnn_cell.LSTMCell(100)])

hiddens = [state]

with tf.variable_scope("LSTM") as scope:
  for i in range(OBS_SIZE):
    if i > 0:
      scope.reuse_variables()
    cell_input = tf.concat(1, [ph_obs_x[i], ph_obs_tf[i]])
    output, state = lstm(cell_input, state)
    hiddens.append(state)

lstm_variables = [v for v in tf.all_variables()
                    if v.name.startswith("LSTM")]

print lstm_variables

print "state shape ", show_dim(state)

VAR_inv += lstm_variables
VAR_pred += lstm_variables


# -------------------------------------------------------------------- invert to predict hidden X
W_inv_x = weight_variable([n_hidden,L])
b_inv_x = bias_variable([L])

VAR_inv += [W_inv_x, b_inv_x]

epsilon1 = tf.constant(1e-10, shape=[N_BATCH, L])
x_invs = [tf.nn.softmax(tf.matmul(volvoo, W_inv_x) + b_inv_x)+epsilon1 for volvoo in hiddens]
print "invs shapes ", show_dim(x_invs)

# compute costs
inv_costs_x = [-tf.reduce_sum(ph_x_x * tf.log(x_pred)) for x_pred in x_invs]
print "costs shapes ", show_dim(inv_costs_x)
cost_inv = sum(inv_costs_x)

# ----------------------------------------------------------------- answer the query
W_query1 = weight_variable([n_hidden + L, n_pred_hidden])

b_query1 = bias_variable([n_pred_hidden])
W_query2 = weight_variable([n_pred_hidden, 2])
b_query2 = bias_variable([2])
VAR_pred += [W_query1, b_query1, W_query2, b_query2]

hidden_cat_query = [tf.nn.relu(\
  tf.matmul(tf.concat(1, [ph_new_ob_x, hidden]),W_query1) + b_query1)\
  for hidden in hiddens]

print "hidden_cat_query shape ", show_dim(hidden_cat_query)
e2 = tf.constant(1e-10, shape=[N_BATCH, 2])
query_preds = [tf.nn.softmax(tf.matmul(hcq, W_query2) + b_query2)+e2 for hcq in hidden_cat_query]
print "query_preds shape ", show_dim(query_preds)

query_pred_costs = [-tf.reduce_sum(ph_new_ob_tf * tf.log(op)) for op in query_preds]
print "costs shapes ", show_dim(query_pred_costs)
cost_query_pred = sum(query_pred_costs)

# ------------------------------------------------------------------------ training steps
optimizer = tf.train.RMSPropOptimizer(0.0002)
train_inv = optimizer.minimize(cost_inv, var_list = VAR_inv)
train_query_pred = optimizer.minimize(cost_query_pred, var_list = VAR_pred)
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
    x_x, obs_x, obs_tfs, new_ob_x, new_ob_tf = gen_data()
    feed_dic = gen_feed_dict(x_x, obs_x, obs_tfs, new_ob_x, new_ob_tf)

    # train inversion
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
      print "for inversion"
      ran_x_invs = sess.run(x_invs, feed_dict=feed_dic)
      print "inverted "
      print ran_x_invs[OBS_SIZE-1][0], np.argmax(ran_x_invs[OBS_SIZE-1][0])
      print "true " 
      print x_x[0], np.argmax(x_x[0])

      print "for query prediction"
      print "query loc "
      print new_ob_x[0]
      ctr = 0
      print "predicted <===> true"
      for haha in zip(sess.run(query_preds, feed_dict=feed_dic)[OBS_SIZE-1], new_ob_tf):
        print haha,
        if np.argmax(haha[0]) == np.argmax(haha[1]):
          print True
          ctr += 1
        else:
          print False
      print float(ctr) / N_BATCH
      save_path = saver.save(sess, save_loc)
      print("Model saved in file: %s" % save_path)


# load the model and give back a session
def load_model(saved_loc):
  saver = tf.train.Saver()
  sess = tf.Session()
  saver.restore(sess, saved_loc)
  print("Model restored.")
  return sess

