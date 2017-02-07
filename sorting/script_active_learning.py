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
draw_truth(query, "drawings/truth.png")
print "active learning"
thing = get_active_inv(sess, query)
for idx, th in enumerate(thing):
  print "hmm "
  print th[0]
  print th[1]
  print get_score(query, th[1])
  draw_all_preds(th[1], "drawings/{0}.png".format(idx))

# print thing[-1]
# 
# print "obss "
# print [x[0] for x in thing]
# 
# print "SCORES"
# score =  get_all_query_score(sess, [ ((0,1), [1.0, 0.0]) for i in range(5)])
# print "Scoraaaa"
# for x in score:
#   print x, score[x]
#  
