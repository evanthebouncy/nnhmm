from model import *
from draw import *
# ------------- helpers --------------
# some on-policy data generation
def full_output_to_trace(full_out, Img, S):
  obs = [x[0] for x in full_out[1:]]
  return Trace(Img, S, obs)

def is_correct(link_fail, all_preds):
  n_cor, n_inc = 0, 0
  for iii in range(N):
    if np.argmax(link_fail[iii]) == np.argmax(all_preds[iii]):
      n_cor += 1
    else:
      n_inc += 1
  return n_cor, n_inc

def get_belief(link_fail, all_preds):
  ret = []
  for iii in range(N):
    for jjj in range(iii):
      if link_fail[iii][jjj] != 0:
        ret.append((all_preds[iii][jjj][0], (iii,jjj)))
  ret.sort()
  return ret

links_fail, _x, ge_fail = get_instance()
print "links failure: "
print links_fail
qry = mk_query(_x) 

draw_graph(G_V, ge_fail, "drawings/graph_fail.png")
# draw_orig(img, "drawings/link_fail.png")

impnet = Implynet("imp")
sess = tf.Session()
impnet.load_model(sess, "model_imply.ckpt")


trace = impnet.get_active_trace(sess, qry, epi=0.0)
print trace

for i in range(len(trace)):
  trace_prefix = trace[:i]
  all_preds = impnet.get_all_preds(sess, trace_prefix)
  n_cor, n_inc = is_correct(_x, all_preds)
  print i, n_cor, n_inc, float(n_cor) / (n_cor + n_inc)
         
#  draw_allob(all_preds, "drawings/pred_ob{0}.png".format(i), trace_prefix)
  # print "belief: "
  # print get_belief(img, all_preds)
  print "------------------------------------", G_OBS[trace[i][0]], trace[i], "made ob"

# trace_rand = impnet.get_active_trace(sess, qry, epi=1.0)
# for i in range(len(trace_rand)):
#   trace_prefix = trace_rand[:i]
#   all_preds = impnet.get_all_preds(sess, trace_prefix)
#   draw_allob(all_preds, "drawings/rand_ob{0}.png".format(i), trace_prefix)


