from active_learning import *

sess = load_model("model.ckpt")

dist_rand = 0
dist_active = 0
for i in range(1,1000):
  hid_x = gen_X()
  query = mk_query(hid_x)
  rand_inv = get_random_inv(sess, query) 
  acti_inv = get_active_inv(sess, query) 
  print "==========================="
  print hid_x
  print rand_inv
  print acti_inv
#   dist_rand += dist(rand_inv, hid_x)
#   dist_active += dist(acti_inv, hid_x)
# 
#   print "avg dist rand ", dist_rand / i 
#   print "avg dist acti ", dist_active / i 
