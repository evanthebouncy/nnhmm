import tensorflow as tf
import numpy as np
from data import *
from active_learning import *

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# set up placeholders
ph_x_x = tf.placeholder(tf.float32, [N_BATCH, X_L], name="ph_x_x")
ph_obs_x = [tf.placeholder(tf.float32, [N_BATCH, L], 
            name="ph_ob_x"+str(i)) for i in range(OBS_SIZE)]
ph_obs_y = [tf.placeholder(tf.float32, [N_BATCH, L],
            name="ph_ob_y"+str(j)) for j in range(OBS_SIZE)]
ph_obs_tf = [tf.placeholder(tf.float32, [N_BATCH, 2],
            name="ph_ob_tf"+str(k)) for k in range(OBS_SIZE)]
ph_new_ob_x = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_new_ob_x")
ph_new_ob_y = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_new_ob_y")
ph_new_ob_tf = tf.placeholder(tf.float32, [N_BATCH,2], name="ph_new_ob_tf")

def gen_feed_dict(x_x, obs_x, obs_y, obs_tf, 
                  new_ob_x, new_ob_y, new_ob_tf):
  ret = {}
  for a, b in zip(ph_obs_x, obs_x):
    ret[a] = b
  for a, b in zip(ph_obs_y, obs_y):
    ret[a] = b
  for a, b in zip(ph_obs_tf, obs_tf):
    ret[a] = b

  ret[ph_x_x] = x_x
  ret[ph_new_ob_x] = new_ob_x
  ret[ph_new_ob_y] = new_ob_y
  ret[ph_new_ob_tf] = new_ob_tf
  return ret

# some constants
n_hidden = 1200
n_pred_hidden = 1000

# a list of variables for different tasks
VAR_inv = []
VAR_pred = []

# ------------------------------------------------------------------ convolve in the observations

state = tf.zeros([N_BATCH, n_hidden])
# initialize some weights
# initialize some weights
# stacked lstm
lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(300), tf.nn.rnn_cell.LSTMCell(300)])

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


# -------------------------------------------------------------------- invert to predict hidden X
W_inv_x = weight_variable([n_hidden,X_L])
b_inv_x = bias_variable([X_L])

VAR_inv += [W_inv_x, b_inv_x]

epsilon1 = tf.constant(1e-10, shape=[N_BATCH, X_L])
x_invs = [tf.nn.softmax(tf.matmul(volvoo, W_inv_x) + b_inv_x)+epsilon1 for volvoo in hiddens]
print "invs shapes ", show_dim(x_invs)

# compute costs
inv_costs_x = [-tf.reduce_sum(ph_x_x * tf.log(x_pred)) for x_pred in x_invs]
print "costs shapes ", show_dim(inv_costs_x)
cost_inv = sum(inv_costs_x)

# ----------------------------------------------------------------- answer the query
W_query1 = weight_variable([n_hidden + L + L, n_pred_hidden])

b_query1 = bias_variable([n_pred_hidden])
W_query2 = weight_variable([n_pred_hidden, 2])
b_query2 = bias_variable([2])
VAR_pred += [W_query1, b_query1, W_query2, b_query2]

hidden_cat_query = [tf.nn.relu(\
  tf.matmul(tf.concat(1, [ph_new_ob_x, ph_new_ob_y, hidden]),W_query1) + b_query1)\
  for hidden in hiddens]

print "hidden_cat_query shape ", show_dim(hidden_cat_query)
e2 = tf.constant(1e-10, shape=[N_BATCH, 2])
query_preds = [tf.nn.softmax(tf.matmul(hcq, W_query2) + b_query2)+e2 for hcq in hidden_cat_query]
print "query_preds shape ", show_dim(query_preds)

query_pred_costs = [-tf.reduce_sum(ph_new_ob_tf * tf.log(op)) for op in query_preds]
print "costs shapes ", show_dim(query_pred_costs)
cost_query_pred = sum(query_pred_costs)

# ------------------------------------------------------------------------ training steps
# gvs = optimizer.compute_gradients(cost)
# capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
# train_op = optimizer.apply_gradients(capped_gvs)


optimizer = tf.train.RMSPropOptimizer(0.0001)

inv_gvs = optimizer.compute_gradients(cost_inv, var_list = VAR_inv)
capped_inv_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in inv_gvs]
#train_inv = optimizer.minimize(cost_inv, var_list = VAR_inv)
train_inv = optimizer.apply_gradients(capped_inv_gvs)

pred_gvs = optimizer.compute_gradients(cost_query_pred, var_list = VAR_pred)
capped_pred_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in pred_gvs]
#train_pred = optimizer.minimize(cost_pred, var_list = VAR_pred)
train_query_pred = optimizer.apply_gradients(capped_pred_gvs)



# train_query_pred = optimizer.minimize(cost_query_pred, var_list = VAR_pred)
# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# ------------------------------------------------------------------------- using the model!

# train the model and save checkpt to a file location
def train_model(save_loc, train_inv=False, restart=False):
  # Launch the graph.
  sess = tf.Session()
  saver = tf.train.Saver()
  if restart:
    sess.run(init)
  else:
    saver.restore(sess, save_loc)
    print("Model restored.")

  for i in range(5000001):
    # get_policy_trace(sess)
    x_x, obs_x, obs_y, obs_tfs, new_ob_x, new_ob_y, new_ob_tf, _ = gen_data()
    feed_dic = gen_feed_dict(x_x, obs_x, obs_y, obs_tfs, new_ob_x, new_ob_y, new_ob_tf)

    # train inversion conditionally
    if train_inv:
      cost_inv_pre = sess.run([cost_inv], feed_dict=feed_dic)[0]
      sess.run([train_inv], feed_dict=feed_dic)
      cost_inv_post = sess.run([cost_inv], feed_dict=feed_dic)[0]
      print "------------------------------------ train inv ", cost_inv_pre, " ", cost_inv_post, " ", True if cost_inv_post < cost_inv_pre else False
    # train query prediction
    cost_query_pred_pre = sess.run([cost_query_pred], feed_dict=feed_dic)[0]
    sess.run([train_query_pred], feed_dict=feed_dic)
    cost_query_pred_post = sess.run([cost_query_pred], feed_dict=feed_dic)[0]
    print "train query pred ", cost_query_pred_pre, " ", cost_query_pred_post, " ", True if cost_query_pred_post < cost_query_pred_pre else False

    if i % 100 == 0:
      if train_inv:
        print "for inversion"
        ran_x_invs = sess.run(x_invs, feed_dict=feed_dic)
        print "inverted "
        print ran_x_invs[OBS_SIZE-1][0], np.argmax(ran_x_invs[OBS_SIZE-1][0])
        print "true " 
        print x_x[0], np.argmax(x_x[0])

      print "for query prediction"
      print "query loc "
      print new_ob_x[0]
      print new_ob_y[0]
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

# some on-policy data generation
def get_policy_trace(sess):
  # x_x, obs_x, obs_y, obs_tfs, new_ob_x, new_ob_y, new_ob_tf, _ = gen_data()
  img, _x = get_img_class()
  qry = mk_query(img)
  acti_inv = get_active_inv(sess, qry)
  obs = [a_v[0] for a_v in acti_inv]
  print obs
  assert 0
    
