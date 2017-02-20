import tensorflow as tf
import numpy as np
from data import *

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# agent act on batches N_BATCH
class Qnetwork():

  def __init__(self, name):
    # some constants
    self.n_hidden = 400
    self.VAR = []

    # inputs
    self.ph_obs_x = [tf.placeholder(tf.float32, [N_BATCH, L], 
            name="ph_ob_x"+str(i)) for i in range(OBS_SIZE)]
    self.ph_obs_tf = [tf.placeholder(tf.float32, [N_BATCH, 2],
            name="ph_ob_tf"+str(k)) for k in range(OBS_SIZE)]


    # inputs for learning only
    # the target for an action
    self.learn_a = [tf.placeholder(tf.float32, [N_BATCH], 
            name="learn_a"+str(i)) for i in range(OBS_SIZE)]
    # a mask telling you which action it is
    self.learn_a_mask = [tf.placeholder(tf.float32, [N_BATCH, L], 
            name="learn_a_mask"+str(i)) for i in range(OBS_SIZE)]
    # a mask telling you on which batch this action actually mattered
    self.learn_a_batch_mask = [tf.placeholder(tf.float32, [N_BATCH], 
            name="learn_a_batch_mask"+str(i)) for i in range(OBS_SIZE)]
    # the supervised value for the tru guess, use this for xentropy
    self.learn_guess = tf.placeholder(tf.float32, [N_BATCH, L], name="learn_guess")
    # a mask telling you which batch this guess actually mattered
    self.learn_guess_batch_mask = tf.placeholder(tf.float32, [N_BATCH],
            name="learn_guess_batch_mask")

    # lstm initial state
    state = tf.zeros([N_BATCH, self.n_hidden])
    # stacked lstm
    lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(100), 
                                        tf.nn.rnn_cell.LSTMCell(100)])
    hiddens = [state]

    with tf.variable_scope("LSTM"+name) as scope:
      for i in range(OBS_SIZE):
        if i > 0:
          scope.reuse_variables()
        cell_input = tf.concat(1, [self.ph_obs_x[i], self.ph_obs_tf[i]])
        output, state = lstm(cell_input, state)
        hiddens.append(state)

    lstm_variables = [v for v in tf.all_variables()
                        if v.name.startswith("LSTM"+name)]

    print lstm_variables

    print "state shape ", show_dim(state)

    self.VAR += lstm_variables

    # --------------------------------------------------------------------- get the Q values
    W_q = weight_variable([self.n_hidden, L])
    b_q = bias_variable([L])

    self.VAR += [W_q, b_q]

    self.qs = [tf.matmul(hidden ,W_q) + b_q for hidden in hiddens]

    # ----------------------------------------------------------- guess the hidden hypothesis
    last_state = hiddens[-1]
    W_guess = weight_variable([self.n_hidden, L])
    b_guess = bias_variable([L])
    self.VAR += [W_guess, b_guess]

    self.guess = tf.nn.softmax(tf.matmul(last_state, W_guess) + b_guess)

    # -------------------------------------------------------------- the networks for learning
    # learning the actions
    blah = [tf.reduce_sum(self.qs[i] * self.learn_a_mask[i], 1) for i in range(OBS_SIZE)]
    print show_dim(blah)
    batch_costs = [self.learn_a_batch_mask[i] * tf.square(blah[i] - self.learn_a[i])\
                   for i in range(OBS_SIZE)]
    print show_dim(batch_costs)
    cost_act = sum([tf.reduce_sum(x) for x in batch_costs])
    print show_dim(cost_act)

    # learning the guesses
    blah2 = tf.reduce_sum(self.learn_guess * tf.log(self.guess), 1)
    cost_guess = tf.reduce_sum(self.learn_guess_batch_mask * blah2)
    print show_dim(cost_guess)

    self.cost = cost_act + cost_guess
    self.optimizer = tf.train.RMSPropOptimizer(0.0002)
    self.train_node = self.optimizer.minimize(self.cost, var_list = self.VAR)

  # clone weights from another network
  def clone_from(self, sess, other):
    copy_ops = []
    for selfVar, otherVar in zip(self.VAR, other.VAR):
      copy_ops.append(selfVar.assign(otherVar))
    sess.run(copy_ops)
    
  def gen_feed_dict(self, obs_x, obs_tf):
    ret = {}
    for a, b in zip(self.ph_obs_x, obs_x):
      ret[a] = b
    for a, b in zip(self.ph_obs_tf, obs_tf):
      ret[a] = b
    return ret

  def get_all_actions(self, sess, obs_x, obs_tf):
    fed_dic = self.gen_feed_dict(obs_x, obs_tf)
    actionz = sess.run(self.qs, feed_dict = fed_dic)
    return actionz

  # get action on the current state batch
  def get_action(self, sess, states_batch):
    fed_obs_x =  [np.zeros(shape=[N_BATCH,L]) for _ in range(OBS_SIZE)]
    fed_obs_tf = [np.zeros(shape=[N_BATCH,2]) for _ in range(OBS_SIZE)]

    state_idxs = [len(xxxx) for xxxx in states_batch]
    for batch_id, s_b in enumerate(states_batch):
      for state_id, s in enumerate(s_b):
        s_x, s_tf = s
        s_x = onehot(s_x, L) 
        fed_obs_x[state_id][batch_id] = s_x
        fed_obs_tf[state_id][batch_id] = s_tf

    # fed_dic = self.gen_feed_dict(fed_obs_x, fed_obs_tf)
    # actionz = sess.run(self.qs[state_idxs[0]], feed_dict=fed_dic)
    # print show_dim(actionz)

    all_acts = self.get_all_actions(sess, fed_obs_x, fed_obs_tf)
    actionz = []
    for batch_idx in range(N_BATCH):
      actionz.append(all_acts[state_idxs[batch_idx]][batch_idx]) 
    return actionz

  def get_guess(self, sess, states_batch):
    fed_obs_x =  [np.zeros(shape=[N_BATCH,L]) for _ in range(OBS_SIZE)]
    fed_obs_tf = [np.zeros(shape=[N_BATCH,2]) for _ in range(OBS_SIZE)]

    state_idx = len(states_batch[0])
    for batch_id, s_b in enumerate(states_batch):
      # supresed for now
      # assert len(s_b) == OBS_SIZE, "how can I guess without full observation"
      for state_id, s in enumerate(s_b):
        s_x, s_tf = s
        fed_obs_x[state_id][batch_id] = s_x
        fed_obs_tf[state_id][batch_id] = s_tf

    fed_dic = self.gen_feed_dict(fed_obs_x, fed_obs_tf)
    guess = sess.run(self.guess, feed_dict=fed_dic)
    return guess


  def fake_change(self, sess):
    cur_val = sess.run(self.VAR[0])
    change_op = self.VAR[0].assign(cur_val + cur_val)
    sess.run(change_op)

class Experience:
  
  def __init__(self, buf_len):
    self.buf = []
    self.buf_len = buf_len

  def trim(self):
    while len(self.buf) > self.buf_len:
      self.buf.pop()

  def add(self, trace):
    for exp in trace[:-1]:
      self.buf.append(exp)
    last_s, guess, last_s_again, guess_xentropy, true_hidden, _ = trace[-1]
    self.buf.append((last_s, None, last_s, guess_xentropy, true_hidden, "guess"))
     
    self.trim()
  
  def sample(self):
    idxxs = np.random.choice(len(self.buf), size=N_BATCH, replace=False)
    return [self.buf[idxxs[i]] for i in range(N_BATCH)]

# trace generation ! ! !
def gen_batch_trace(sess, qnet, envs):
  ret = [[] for _ in range(N_BATCH)]
  truths = [eenv.X for eenv in envs]
  # start with the empty trace
  states = [[] for _ in range(N_BATCH)]

  for _ in range(OBS_SIZE):
    new_states = []

    # all the moves that involves a query
    actions = qnet.get_action(sess, states)
    for batch_id in range(N_BATCH):
      # we need to put truth as the last case in case ss is the final state
      # this way we can compute the V(ss) using the target Qnetwork
      trutru = truths[batch_id] if _ == OBS_SIZE - 1 else None
      env = envs[batch_id]
      s = states[batch_id]
      act = actions[batch_id]
      ss, r = env.step(s, act)
      ret[batch_id].append((s, np.argmax(act), ss, r, trutru, "query"))
      new_states.append(ss)

    states = new_states

  # take a guess in the end
  guess = qnet.get_guess(sess, states)
  guess_con_env = zip(guess, envs)
  guess_reward = [x[1].get_final_reward(x[0])  for x in guess_con_env]
  last_experience = zip(states, guess, states, 
                        guess_reward, truths, ["guess" for _ in range(N_BATCH)])

  for i_batch in range(N_BATCH):
    ret[i_batch].append(last_experience[i_batch])
  return ret
      
# target generation ! !
# experience is many many (s, a, s', r)
# we want to conver them with target into ("query", s, a, target)
# if s is the FINAL state before our prediction, leave it as ("guess", s, a, truth) just learn with supervised using the truth
# if s' is the FINAL state, then we use xentropy with Q_targets last step to get target
# otherwise we use max_{a'}Q(s',a') for the a' as the query step
def gen_target(sess, qnet_target, batch_experience):
  ret = []
  s_s = [x[0] for x in batch_experience]
  ss_s = [x[2] for x in batch_experience]
  all_qs = qnet_target.get_action(sess, s_s)
  guesses = qnet_target.get_guess(sess, ss_s)

  for batch_id, exp in enumerate(batch_experience):
    s, a, ss, r, truth, typ = exp
    # if it is a ultimate state, i.e. a guess, train it supervised and just pass it on
    if typ == "guess":
      ret.append(("guess", s, a, truth))
    if typ == "query":
      # if this is the penultimate state
      # use the Q to find out V(ss) from guess 
      if truth != None:
        tru_1hot = onehot(truth, L)
        target_guess = guesses[batch_id]
        targ = xentropy(tru_1hot, target_guess) + r
        ret.append(("query", s, a, targ))
      # if this is just a normal query state
      # use the Q to find out V(ss) from argmax over actions
      else:
        qaction = all_qs[len(ss)][batch_id] 
        targ = np.max(qaction) + r
        ret.append(("query", s, a, targ))
  return ret











 


















 


















