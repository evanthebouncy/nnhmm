from model_q import *

qq = Qnetwork("orig")
qq_clone = Qnetwork("clone")

N_EPI = 1000

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  experiences = Experience(1000)
  envs = get_envs(1.0)

  for episode in range(N_EPI):
    print "ON EPISODE ", episode

    # target network cloning once in awhile
    if episode % 1000 == 0:
      print "cloining target"
      qq_clone.clone_from(sess, qq)

#    episilon = 1.0 - float(episode) / N_EPI

#    envs = get_envs(episilon)

    trace = gen_batch_trace(sess, qq, envs)
    print "for this particular trace "
    for xxxx in trace[0]:
      s, a, ss, r, trut, typ = xxxx
      print "HI HI HI"
      print s, " ", a, " ", np.argmin(a), " reward ", r
    print "END"
    print envs[0].X

    print "Add to experience"
    for tr in trace:
      experiences.add(tr)

    print "A sample of experience "
    a_sample = experiences.sample()

    print "generating target from sample"
    target = gen_target(sess, qq_clone, a_sample)

    print "for this particular target "
    for tgg in target:
      print tgg

    print "learning "
    qq.learn(sess, prepare_target_feed(target))

