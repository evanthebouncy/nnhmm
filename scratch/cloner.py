import tensorflow as tf
import numpy as np

def weight_variable(shape, name="W"):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

class FF():
  def __init__(self):

    self.inputt = tf.placeholder(tf.float32, [1, 4])
    self.W = weight_variable([4, 2])

    self.outputt = tf.matmul(self.inputt, self.W)

    # fake a small error
    self.err = tf.reduce_sum(self.W)
    self.optimizer = tf.train.RMSPropOptimizer(0.02)
    self.opt_node = self.optimizer.minimize(self.err)

  def clone_from(self, sess, other):
    copyy = self.W.assign(other.W)
    sess.run(copyy)

## some running stuff


inp = np.random.rand(*[1, 4])

print inp

ff_1 = FF()
ff_2 = FF()

ff_clone = FF()

init = tf.initialize_all_variables()
with tf.Session() as sess:
  sess.run(init)    
  
  out1 = sess.run(ff_1.outputt, 
    feed_dict={ff_1.inputt : inp})
  print "out1", out1

#  out2 = sess.run(ff_2.outputt, 
#    feed_dict={ff_2.inputt : inp})
#  print "out2", out2

  print "clone out 1",
  ff_clone.clone_from(sess, ff_1)
  clone_out1 = sess.run(ff_clone.outputt, 
    feed_dict={ff_clone.inputt : inp})
  print clone_out1

  print "updating weight for orig 1",
  out1 = sess.run(ff_1.opt_node,
    feed_dict={ff_1.inputt : inp})

  out1 = sess.run(ff_1.outputt, 
    feed_dict={ff_1.inputt : inp})
  print "out1 new", out1

  print "clone out 1 again, hopefully no change?",
  clone_out1 = sess.run(ff_clone.outputt, 
    feed_dict={ff_clone.inputt : inp})
  print clone_out1
  
for xxx in tf.all_variables():
  print xxx.name






















