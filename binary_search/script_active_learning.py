from active_learning import *

sess = load_model("model.ckpt")


hid_x = gen_X()
query = mk_query(hid_x)
rand_inv = get_random_inv(sess, query) 
acti_inv = get_active_inv(sess, query) 

print "original hidden ", hid_x
print "random sample ", rand_inv
print "active learning ", acti_inv
draw_trace(rand_inv, "rand_inv.png")
draw_trace(acti_inv, "acti_inv.png")

assert 0


dist_rand = 0
dist_active = 0
for i in range(1,1000):
  hid_x = gen_X()
  query = mk_query(hid_x)
  rand_inv = get_random_inv(sess, query) 
  acti_inv = get_active_inv(sess, query) 
  dist_rand += dist(rand_inv, hid_x)
  dist_active += dist(acti_inv, hid_x)

  print "avg dist rand ", dist_rand / i 
  print "avg dist acti ", dist_active / i 
