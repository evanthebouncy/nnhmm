from model_q import *


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
    
imply = Inetwork("imply")
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  for i in range(10000):
    print "\n\n\nITERAT ", i
    envs = get_envs()

    
    stackz = get_states_batch(envs)
    print stackz

    print "potential before "
    imply.get_potential(sess, stackz)

    imply.learn(sess, stackz)
   
    print "potential after "
    imply.get_potential(sess, stackz)

    stack11 = [((5, [1.0, 0.0]), (8, [0.0, 1.0]), (6, [1.0, 0.0]))]
    stack22 = [((5, [1.0, 0.0]), (8, [0.0, 1.0]), (9, [0.0, 1.0]))]

    pot11 = imply.get_potential(sess, stack11)
    pot22 = imply.get_potential(sess, stack22)
    print "potential of special stackz "
    print pot11
    print pot22
