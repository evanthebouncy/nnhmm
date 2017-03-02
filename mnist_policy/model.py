import tensorflow as tf
import numpy as np
from data import *

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

class Implynet:

  def gen_feed_dict(self, x_x, obs_x, obs_y, obs_tf, 
                    new_ob_x, new_ob_y, new_ob_tf):
    ret = {}
    for a, b in zip(self.ph_obs_x, obs_x):
      ret[a] = b
    for a, b in zip(self.ph_obs_y, obs_y):
      ret[a] = b
    for a, b in zip(self.ph_obs_tf, obs_tf):
      ret[a] = b

    ret[self.ph_x_x] = x_x
    ret[self.ph_new_ob_x] = new_ob_x
    ret[self.ph_new_ob_y] = new_ob_y
    ret[self.ph_new_ob_tf] = new_ob_tf
    return ret

  # load the model and give back a session
  def load_model(self, sess, saved_loc):
    self.saver = tf.train.Saver()
    self.saver.restore(sess, saved_loc)
    print("Model restored.")

  # make the model
  def __init__(self):
    # set up placeholders
    self.ph_x_x = tf.placeholder(tf.float32, [N_BATCH, X_L], name="ph_x_x")
    self.ph_obs_x = [tf.placeholder(tf.float32, [N_BATCH, L], 
                name="ph_ob_x"+str(i)) for i in range(OBS_SIZE)]
    self.ph_obs_y = [tf.placeholder(tf.float32, [N_BATCH, L],
                name="ph_ob_y"+str(j)) for j in range(OBS_SIZE)]
    self.ph_obs_tf = [tf.placeholder(tf.float32, [N_BATCH, 2],
                name="ph_ob_tf"+str(k)) for k in range(OBS_SIZE)]
    self.ph_new_ob_x = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_new_ob_x")
    self.ph_new_ob_y = tf.placeholder(tf.float32, [N_BATCH, L], name="ph_new_ob_y")
    self.ph_new_ob_tf = tf.placeholder(tf.float32, [N_BATCH,2], name="ph_new_ob_tf")


    # some constants
    self.n_hidden = 1200
    self.n_pred_hidden = 1000

    # a list of variables for different tasks
    self.VAR_inv = []
    self.VAR_pred = []

    # ------------------------------------------------------------------ convolve in the observations
    # initial lstm state
    state = tf.zeros([N_BATCH, self.n_hidden])
    # initialize some weights
    # initialize some weights
    # stacked lstm
    lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(300), tf.nn.rnn_cell.LSTMCell(300)])

    hiddens = [state]

    with tf.variable_scope("LSTM") as scope:
      for i in range(OBS_SIZE):
        if i > 0:
          scope.reuse_variables()
        cell_input = tf.concat(1, [self.ph_obs_x[i], self.ph_obs_y[i], self.ph_obs_tf[i]])
        output, state = lstm(cell_input, state)
        hiddens.append(state)

    lstm_variables = [v for v in tf.all_variables()
                        if v.name.startswith("LSTM")]

    print lstm_variables

    # self.VAR_inv += lstm_variables
    self.VAR_pred += lstm_variables


    # -------------------------------------------------------------------- invert to predict hidden X
    W_inv_x = weight_variable([self.n_hidden,X_L])
    b_inv_x = bias_variable([X_L])

    self.VAR_inv += [W_inv_x, b_inv_x]

    epsilon1 = tf.constant(1e-10, shape=[N_BATCH, X_L])
    self.x_invs = [tf.nn.softmax(tf.matmul(volvoo, W_inv_x) + b_inv_x)+epsilon1 for volvoo in hiddens]
    print "invs shapes ", show_dim(self.x_invs)

    # compute costs
    inv_costs_x = [-tf.reduce_sum(self.ph_x_x * tf.log(x_pred)) for x_pred in self.x_invs]
    print "costs shapes ", show_dim(inv_costs_x)
    self.cost_inv = sum(inv_costs_x)

    # ----------------------------------------------------------------- answer the query
    W_query1 = weight_variable([self.n_hidden + L + L, self.n_pred_hidden])

    b_query1 = bias_variable([self.n_pred_hidden])
    W_query2 = weight_variable([self.n_pred_hidden, 2])
    b_query2 = bias_variable([2])
    self.VAR_pred += [W_query1, b_query1, W_query2, b_query2]

    hidden_cat_query = [tf.nn.relu(\
      tf.matmul(tf.concat(1, [self.ph_new_ob_x, self.ph_new_ob_y, hidden]),W_query1) + b_query1)\
      for hidden in hiddens]

    print "hidden_cat_query shape ", show_dim(hidden_cat_query)
    e2 = tf.constant(1e-10, shape=[N_BATCH, 2])
    self.query_preds = [tf.nn.softmax(tf.matmul(hcq, W_query2) + b_query2)+e2 for hcq in hidden_cat_query]
    print "query_preds shape ", show_dim(self.query_preds)

    query_pred_costs = [-tf.reduce_sum(self.ph_new_ob_tf * tf.log(op)) for op in self.query_preds]
    print "costs shapes ", show_dim(query_pred_costs)
    self.cost_query_pred = sum(query_pred_costs)

    # ------------------------------------------------------------------------ training steps
    # gvs = optimizer.compute_gradients(cost)
    # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    # train_op = optimizer.apply_gradients(capped_gvs)


    optimizer = tf.train.RMSPropOptimizer(0.0001)

    inv_gvs = optimizer.compute_gradients(self.cost_inv, var_list = self.VAR_inv)
    capped_inv_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in inv_gvs]
    #train_inv = optimizer.minimize(cost_inv, var_list = VAR_inv)
    self.train_inv = optimizer.apply_gradients(capped_inv_gvs)

    pred_gvs = optimizer.compute_gradients(self.cost_query_pred, var_list = self.VAR_pred)
    capped_pred_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in pred_gvs]
    #train_pred = optimizer.minimize(cost_pred, var_list = VAR_pred)
    self.train_query_pred = optimizer.apply_gradients(capped_pred_gvs)



    # train_query_pred = optimizer.minimize(cost_query_pred, var_list = VAR_pred)
    # Before starting, initialize the variables.  We will 'run' this first.
    self.init = tf.initialize_all_variables()

  # save the model
  def save(self, sess, model_loc="model.ckpt"):
    save_path = self.saver.save(sess, model_loc)
    print("Model saved in file: %s" % save_path)

  # train on a particular data batch
  def train(self, sess, data_batch, train_inv=False):
    x_x, obs_x, obs_y, obs_tfs, new_ob_x, new_ob_y, new_ob_tf, _ = data_batch
    feed_dic = self.gen_feed_dict(x_x, obs_x, obs_y, obs_tfs, new_ob_x, new_ob_y, new_ob_tf)

    # train inversion conditionally
    if train_inv:
      cost_inv_pre = sess.run([self.cost_inv], feed_dict=feed_dic)[0]
      sess.run([self.train_inv], feed_dict=feed_dic)
      cost_inv_post = sess.run([self.cost_inv], feed_dict=feed_dic)[0]
      print "--------------------------- train inv ",\
        cost_inv_pre, " ", cost_inv_post, " ", True if cost_inv_post < cost_inv_pre else False
    # train query prediction
    if not train_inv:
      cost_query_pred_pre = sess.run([self.cost_query_pred], feed_dict=feed_dic)[0]
      sess.run([self.train_query_pred], feed_dict=feed_dic)
      cost_query_pred_post = sess.run([self.cost_query_pred], feed_dict=feed_dic)[0]
      print "train query pred ", cost_query_pred_pre, " ",\
        cost_query_pred_post, " ", True if cost_query_pred_post < cost_query_pred_pre else False

  # =========== HELPERS =============

  # a placeholder to feed in a single observation
  def get_feed_dic_obs(self, obs):
    # needing to create all the nessisary feeds
    obs_x = []
    obs_y = []
    obs_tf = []
    
    for _ in range(OBS_SIZE):
      obs_x.append(np.zeros([N_BATCH,L]))
      obs_y.append(np.zeros([N_BATCH,L]))
      obs_tf.append(np.zeros([N_BATCH,2]))

    num_obs = len(obs)
    for ob_idx in range(num_obs):
      ob_coord, ob_lab = obs[ob_idx]
      ob_x, ob_y = vectorize(ob_coord)
      obs_x[ob_idx] = np.tile(ob_x, [50,1])
      obs_y[ob_idx] = np.tile(ob_y, [50,1])
      obs_tf[ob_idx] = np.tile(ob_lab, [50,1])

    feed_dic = dict(zip(self.ph_obs_x + self.ph_obs_y + self.ph_obs_tf, 
                        obs_x + obs_y + obs_tf))
    return feed_dic

  def get_preds_batch(self, sess, obs, batch_querys):
    ret = [[] for _ in range(OBS_SIZE+1)]

    feed_dic = self.get_feed_dic_obs(obs)
    assert len(batch_querys) == N_BATCH

    new_ob_x = []
    new_ob_y = []

    for q in batch_querys:
      q_x, q_y = vectorize(q)
      new_ob_x.append(q_x)
      new_ob_y.append(q_y)

      
    feed_dic[self.ph_new_ob_x] = np.array(new_ob_x)
    feed_dic[self.ph_new_ob_y] = np.array(new_ob_y)

    pred_tfs = sess.run(self.query_preds, feed_dict=feed_dic)
    for key_ob in range(OBS_SIZE+1):
      for q_idx, q in enumerate(batch_querys):
        ret[key_ob].append((q, pred_tfs[key_ob][q_idx]))

    return ret

  def get_all_preds_fast(self, sess, obs):
    all_querys = []
    for i in range(L):
      for j in range(L):
        all_querys.append((i,j))

    def batch_qrys(all_qs):
      ret = []
      while len(all_qs) != 0:
        to_add = [(0,0) for _ in range(N_BATCH)]
        for idk in range(N_BATCH):
          if len(all_qs) == 0:
            break
          to_add[idk] = all_qs.pop()
        ret.append(to_add)
      return ret

    ret = [[] for _ in range(OBS_SIZE+1)]
    batched_qrysss = batch_qrys(all_querys)
    for batched_q in batched_qrysss:
      ppp = self.get_preds_batch(sess, obs, batched_q)
      for ijk in range(OBS_SIZE+1):
        ret[ijk] += ppp[ijk]

    return ret

  def get_most_confuse(self, sess, obs):
    key_ob = len(obs)
    all_preds = self.get_all_preds_fast(sess, obs)
    
    all_pred_at_key = all_preds[key_ob]

    most_confs = [(abs(x[1][0] - x[1][1]), x[0]) for x in all_pred_at_key]
    most_conf = min(most_confs)

    return most_conf[1]

  def get_active_inv(self, sess, query):
    obs = []

    for i in range(OBS_SIZE):
      most_conf = self.get_most_confuse(sess, obs)
      obs.append((most_conf, query(most_conf)))

    feed_dic = self.get_feed_dic_obs(obs)
    invs = [(x[0], np.argmax(x[0])) for x in sess.run(self.x_invs, feed_dict=feed_dic)]
    return zip([None] + obs, self.get_all_preds_fast(sess, obs), invs)

  def get_random_inv(self, sess, query):
    ob_pts = [sample_coord_bias(query) for _ in range(OBS_SIZE)]
    obs = [(op, query(op)) for op in ob_pts]
    
    feed_dic = self.get_feed_dic_obs(obs)
    invs = [(x[0], np.argmax(x[0])) for x in sess.run(self.x_invs, feed_dict=feed_dic)]
    return zip([None] + obs, self.get_all_preds_fast(sess, obs), invs)


    
