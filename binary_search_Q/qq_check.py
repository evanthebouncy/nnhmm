from model_q import *

x_x, obs_x, obs_tf, new_ob_x, new_ob_tf = gen_data()

qq1 = Qnetwork("orig")
qq2 = Qnetwork("clone")

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  print "from network orig"
  qs = qq1.get_all_actions(sess, obs_x, obs_tf)
  print qs[7][0][:4]

  print "from network clone"
  qq2.clone_from(sess, qq1)
  qs = qq2.get_all_actions(sess, obs_x, obs_tf)
  print qs[7][0][:4]

  print "faking a change"
  qq1.fake_change(sess)

  print "from network orig after change"
  qs = qq1.get_all_actions(sess, obs_x, obs_tf)
  print qs[7][0][:4]

  print "from network clone after orig changed, this shouldnt change"
  qs = qq2.get_all_actions(sess, obs_x, obs_tf)
  print qs[7][0][:4]

  print "obs size ", OBS_SIZE
  print "action size ", len(qs)

  action1 = qq1.get_action(sess, [[] for _ in range(N_BATCH)])
  print action1
  print show_dim(action1)

envs = get_envs()
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  print "HEYA"
  tracee = gen_trace(sess, qq1, envs)
  print len(tracee), " ", len(tracee[0])
  print "let us look at a particular trace"
  tracee0 = tracee[0]
  for exp in tracee0:
    s, a, ss, r = exp
    print show_state(s), np.argmax(a)











