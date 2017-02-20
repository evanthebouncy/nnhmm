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
  print show_dim(action1)

envs = get_envs()
experiences = Experience(5000)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  print "HEYA"
  tracee = gen_batch_trace(sess, qq1, envs)
  print len(tracee), " ", len(tracee[0])
  print "for a particular trace "
  for xxxx in tracee[0]:
    print "HI HI HI"
    print len(xxxx)
    print xxxx
  print "END"
  print envs[0].X

  print "Add to experience"
  for tr in tracee:
    experiences.add(tr)

  print "A sample of experience "
  print experiences.sample()









