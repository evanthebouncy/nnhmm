from model import *
from active_learning import *

# train the model and save checkpt to a file location
def train_model(save_loc):
  # Launch the graph.
  sess = tf.Session()
  sess.run(init)
  saver = tf.train.Saver()


  for i in range(5000001):
    x_x, x_y, obs_x, obs_y, obs_tfs, new_ob_x, new_ob_y, new_ob_tf = get_active_train_batch(sess)
    feed_dic = gen_feed_dict(x_x, x_y, obs_x, obs_y, obs_tfs, new_ob_x, new_ob_y, new_ob_tf)

    # train inversion
    cost_inv_pre = sess.run([cost_inv], feed_dict=feed_dic)[0]
    sess.run([train_inv], feed_dict=feed_dic)
    cost_inv_post = sess.run([cost_inv], feed_dict=feed_dic)[0]
    print "train inv ", cost_inv_pre, " ", cost_inv_post, " ", True if cost_inv_post < cost_inv_pre else False
    # train query prediction
    cost_query_pred_pre = sess.run([cost_query_pred], feed_dict=feed_dic)[0]
    sess.run([train_query_pred], feed_dict=feed_dic)
    cost_query_pred_post = sess.run([cost_query_pred], feed_dict=feed_dic)[0]
    print "train query pred ", cost_query_pred_pre, " ", cost_query_pred_post, " ", True if cost_query_pred_post < cost_query_pred_pre else False

    if i % 100 == 0:
      print "for inversion"
      ran_x_invs = sess.run(x_invs, feed_dict=feed_dic)
      ran_y_invs = sess.run(y_invs, feed_dict=feed_dic)
      print "inverted "
      print ran_x_invs[9][0], np.argmax(ran_x_invs[9][0])
      print ran_y_invs[9][0], np.argmax(ran_y_invs[9][0]) 
      print "true " 
      print x_x[0], np.argmax(x_x[0])
      print x_y[0], np.argmax(x_y[0])

      print "for query prediction"
      print "query loc "
      print new_ob_x[0]
      print new_ob_y[0]
      print "predicted <===> true"
      for haha in zip(sess.run(query_preds, feed_dict=feed_dic)[9], new_ob_tf):
        print haha
      save_path = saver.save(sess, save_loc)
      print("Model saved in file: %s" % save_path)

train_model("model.ckpt")
