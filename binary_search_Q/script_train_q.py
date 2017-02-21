from model_q import *

qq = Qnetwork("orig")
qq_clone = Qnetwork("clone")

N_EPI = 10000
N_CLONE = 20

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  experiences = Experience(1000)
#  envs = get_envs(0.0)

  for episode in range(N_EPI):
    print "ON EPISODE ", episode

    # target network cloning once in awhile
    if episode % N_CLONE == 0:
      print "cloining target"
      qq_clone.clone_from(sess, qq)

    episilon = 1.0 - float(episode) / N_EPI
    envs = get_envs(episilon)

    trace = gen_batch_trace(sess, qq, envs)
    print "for this particular trace "
    for xxxx in trace[0]:
      s, a, ss, r, trut, typ = xxxx
      print "HI HI HI"
      print s, " ", a, " ", np.argmax(a), " reward ", r
    print "END"
    print envs[0].X

    print "on this trace correct number "
    lasts = [np.argmax(trrrr[-1][1]) for trrrr in trace]
    lasts_con_tru = zip(lasts, [ee.X for ee in envs])
    print lasts_con_tru
    shitz = [1 if x[0] == x[1] else 0 for x in lasts_con_tru]
    print sum(shitz) 

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

    print "prepared feed "
    # print prepare_target_feed(target)

    print "learning "
    qq.learn(sess, prepare_target_feed(target))

