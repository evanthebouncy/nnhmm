from model import *
from draw import *
# ------------- helpers --------------
# some on-policy data generation
def full_output_to_trace(full_out, Img, S):
  obs = [x[0] for x in full_out[1:]]
  return Trace(Img, S, obs)

def is_correct(link_fail, all_prds, iii, jjj):
  pred = np.argmax(all_preds[iii][jjj])
  if link_fail[iii][jjj] == 1 and pred > 1:
    return True
  if link_fail[iii][jjj] == 1 and pred == 1:
    return False
  if link_fail[iii][jjj] == -1 and pred > 1:
    return False
  if link_fail[iii][jjj] == -1 and pred == 1:
    return True
  return None

impnet = Implynet("imp")
sess = tf.Session()
impnet.load_model(sess, "model_imply.ckpt")

img, _x, ge_fail = get_instance()
qry = mk_query(_x) 

draw_graph(G_V, ge_fail, "drawings/graph_fail.png")

trace = impnet.get_active_trace(sess, qry, epi=0.0)
print trace
pps = []

draw_orig(img, "drawings/link_fail.png")
for i in range(len(trace)):
  trace_prefix = trace[:i]
  all_preds = impnet.get_all_preds(sess, trace_prefix)
  pps.append(all_preds)
  n_cor, n_inc = 0, 0
  #print "------------"
  for iii in range(N):
    for jjj in range(iii):
      #print iii, jjj, img[iii][jjj], all_preds[iii][jjj]
      if is_correct(img, all_preds, iii, jjj) == True:
        #print "yay"
        n_cor += 1
      if is_correct(img, all_preds, iii, jjj) == False:
        #print "fuck"
        n_inc += 1
  better = float(n_cor) / (n_cor + n_inc) > float(i) / 49
  print i, n_cor, n_inc, float(n_cor) / (n_cor + n_inc), float(i) / 49, better
         
  draw_allob(all_preds, "drawings/pred_ob{0}.png".format(i), trace_prefix)

# trace_rand = impnet.get_active_trace(sess, qry, epi=1.0)
# for i in range(len(trace_rand)):
#   trace_prefix = trace_rand[:i]
#   all_preds = impnet.get_all_preds(sess, trace_prefix)
#   draw_allob(all_preds, "drawings/rand_ob{0}.png".format(i), trace_prefix)


