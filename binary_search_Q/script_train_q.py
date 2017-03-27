from model_q import *

qq = Qnetwork("orig")
qq_clone = Qnetwork("clone")
imply = Inetwork("imply")
# imply = None

N_EPI = 5000000000
N_CLONE = 100











def get_states_batch(envs):
  ret = []
  for e in envs:
    to_add = []
    for ii in range(np.random.randint(1, OBS_SIZE)):
      rand_oo = np.random.randint(L)
      rand_obs = e.query(rand_oo)
      to_add.append((rand_oo, rand_obs))
    ret.append(to_add)
  return ret




with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  experiences = Experience(10000)
#  envs = get_envs(0.0)

  for i in range(5000):
    print "\n\n\nPRE TRAIN ITERAT ", i, " ", 5000
    envs = get_envs()
    
    stackz = get_states_batch(envs)
    imply.learn(sess, stackz)








  for episode in xrange(N_EPI):
    print "-------------------------------------------- ON EPISODE ", episode

    # target network cloning once in awhile
    if episode % N_CLONE == 0:
      print "cloining target"
      qq_clone.clone_from(sess, qq)

    episilon = 1.0 - float(episode) / N_EPI if np.random.random() < 0.5 else 0.0
    print "episilon of random move probability ", episilon
    envs = get_envs()

    trace = gen_batch_trace(sess, qq, envs, episilon)
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
    print "Experience size ", len(experiences.buf)

    print "A sample of experience "
    a_sample = experiences.sample()

    print "generating target from sample"
    # target = gen_target(sess, qq_clone, a_sample)
    target = gen_target(sess, qq_clone, a_sample, imply)

    print "for this particular target "
    for tgg in target:
      print tgg

    print "prepared feed "
    # print prepare_target_feed(target)

    print "learning "
    qq.learn(sess, prepare_target_feed(target))

    print "training prediction from ss in sample"
    imply.learn(sess, [x[2] for x in a_sample])

