from model import *
from draw import *
from naive_baseline import *

# ------------- helpers --------------
def pred_acc(preds, qry):                                                              
  num_cor = 0
  for i in range(L):
    for j in range(L):
      if np.argmax(preds[i][j]) == np.argmax(qry((i,j))):
        num_cor += 1 
  return float(num_cor) / L*L     

impnet = Implynet("imp")
sess = tf.Session()
impnet.load_model(sess, "model_imply.ckpt")

img, _x = get_img_class()
qry = mk_query(_x) 

trace = impnet.get_active_trace(sess, qry, epi=0.0)

draw_orig(img, "drawings/orig.png")
for i in range(len(trace)):
  trace_prefix = trace[:i]
  all_preds = impnet.get_all_preds(sess, trace_prefix)
  draw_allob(all_preds, "drawings/pred_ob{0}.png".format(i), trace_prefix)

# trace_rand = impnet.get_active_trace(sess, qry, epi=1.0)
# for i in range(len(trace_rand)):
#   trace_prefix = trace_rand[:i]
#   all_preds = impnet.get_all_preds(sess, trace_prefix)
#   draw_allob(all_preds, "drawings/rand_ob{0}.png".format(i), trace_prefix)

print "we gonna run on 1000 trials and compare the 2 agents"
num_baseline = np.array([0 for _ in range(L*L)])
num_active = np.array([0 for _ in range(L*L)])

for _ in range(1, 1000):
  img, _x = get_img_class()
  qry = mk_query(_x) 
  baseline_trace = baseline_get_trace(qry)
  active_trace = impnet.get_active_trace(sess, qry, epi=0.0, play=False)

  for i in range(len(baseline_trace)):
    trace_prefix = baseline_trace[:i]
    all_preds = baseline_get_all_preds(trace_prefix)
    baseline_acc = pred_acc(all_preds, qry)
    num_baseline[i] += baseline_acc

  for i in range(len(active_trace)):
    trace_prefix = active_trace[:i]
    all_preds = impnet.get_all_preds(sess, trace_prefix)
    acti_acc = pred_acc(all_preds, qry)
    num_active[i] += acti_acc

  print "iteration ", _
  print "baseline ", num_baseline / _
  print "active ", num_active / _



