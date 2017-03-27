from active_learning import *

# disable the random hit
RAND_HIT = 0.0

sess = load_model("model.ckpt")


hid_x = gen_X()
print hid_x
hid_a = gen_A(hid_x)
print hid_a
query = mk_query(hid_x)

# print "random sample"
# a_min, a_max, scores =  get_random_inv_direct(sess, query)
# print a_min, np.argmax(a_min)
# print a_max, np.argmax(a_max)
# # 
print "active learning"
print get_active_inv(sess, query)
# 
# print "SCORES"
# score =  get_all_query_score(sess, [ ((0,1), [1.0, 0.0]) for i in range(5)])
# print "Scoraaaa"
# for x in score:
#   print x, score[x]
#  
