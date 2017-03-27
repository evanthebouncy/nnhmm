import tensorflow as tf
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial), tf.placeholder(tf.float32, shape)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial), tf.placeholder(tf.float32, shape)

class FF():
  def use(self, v, vclone):
    if self.is_clone:
      return vclone
    else:
      return v
  
  def __init__(self, is_clone):
    self.is_clone = is_clone

    self.inputt = tf.placeholder(tf.float32, [10, 4])
    self.W, self.W_clone = weight_variable([4, 2])

    W = self.use(self.W, self.W_clone)

    self.outputt = tf.matmul(self.inputt, W)


## some running stuff


inp = np.random.rand(*[10, 4])

print inp

ff_orig = FF(is_clone=False)
ff_clone = FF(is_clone=True)
init = tf.initialize_all_variables()
with tf.Session() as sess:
  sess.run(init)    
  
  orig_out = sess.run(ff_orig.outputt, 
    feed_dict={ff_orig.inputt : inp})

  orig_W = sess.run(ff_orig.W,
    feed_dict={ff_orig.inputt : inp})

  print orig_W

  print orig_out

  clone_out = sess.run(ff_clone.outputt,
    feed_dict={ff_clone.inputt : inp,
               ff_clone.W_clone : orig_W})

  print clone_out























