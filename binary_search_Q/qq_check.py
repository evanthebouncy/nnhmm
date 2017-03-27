from model_q import *

x_x, obs_x, obs_tf, new_ob_x, new_ob_tf = gen_data()

qq1 = Qnetwork("orig")
qq2 = Qnetwork("clone")
imply = Inetwork("imply")

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  print "from network orig"
  qs = qq1.get_all_actions(sess, obs_x, obs_tf)
  print qs[2][0][:4]

  print "from network clone"
  qq2.clone_from(sess, qq1)
  qs = qq2.get_all_actions(sess, obs_x, obs_tf)
  print qs[2][0][:4]

  print "faking a change"
  qq1.fake_change(sess)

  print "from network orig after change"
  qs = qq1.get_all_actions(sess, obs_x, obs_tf)
  print qs[2][0][:4]

  print "from network clone after orig changed, this shouldnt change"
  qs = qq2.get_all_actions(sess, obs_x, obs_tf)
  print qs[2][0][:4]

  print "obs size ", OBS_SIZE
  print "action size ", len(qs)

  action1 = qq1.get_action(sess, [[] for _ in range(N_BATCH)])
  print show_dim(action1)

envs = get_envs()
experiences = Experience(5000)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  print "clone qq2"
  qq2.clone_from(sess, qq1)

  print "comparing trace from qq1 and qq2"
  print "HEYA trace from qq1"
  tracee = gen_batch_trace(sess, qq1, envs, 0.0)
  print len(tracee), " ", len(tracee[0])
  print "for a particular trace "
  for xxxx in tracee[0]:
    print "HI HI HI"
    print len(xxxx)
    print xxxx
  print "END"
  print envs[0].X

  print "HEYA trace from qq1"
  tracee2 = gen_batch_trace(sess, qq2, envs, 0.0)
  print len(tracee2), " ", len(tracee2[0])
  print "for a particular trace "
  for xxxx in tracee2[0]:
    print "HI HI HI"
    print len(xxxx)
    print xxxx
  print "END"
  print envs[0].X

  print "Add to experience"
  for tr in tracee:
    experiences.add(tr)

  print "A sample of experience "
  a_sample = experiences.sample()
  # for xx in a_sample:
  #   print len(xx), " ", xx
  # print "END"

  print "generating target from sample"
  target = gen_target(sess, qq2, a_sample, imply)
  # for tg in target:
  #   print tg

  print "preparing the feeder "
  _sb, _a, _am, _abm, _g, _gm = prepare_target_feed(target)
  print show_dim(_a)
  print show_dim(_am)
  print show_dim(_abm)
  print show_dim(_g)
  print show_dim(_gm)

  qq1.learn(sess, (_sb, _a, _am, _abm, _g, _gm))

  print "now qq1 should be bit different from qq2 in the traces"
  print "qq 1"
  tracee = gen_batch_trace(sess, qq1, envs)
  print len(tracee), " ", len(tracee[0])
  print "for a particular trace "
  for xxxx in tracee[0]:
    print "HI HI HI"
    print len(xxxx)
    print xxxx
  print "END"
  print "qq 2"
  tracee = gen_batch_trace(sess, qq2, envs)
  print len(tracee), " ", len(tracee[0])
  print "for a particular trace "
  for xxxx in tracee[0]:
    print "HI HI HI"
    print len(xxxx)
    print xxxx
  print "END"








