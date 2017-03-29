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

print "we gonna run on 1000 trials and compare the 4 agents"
num_random = np.array([0.0 for _ in range(L*L)])
num_sink = np.array([0.0 for _ in range(L*L)])
num_greedy = np.array([0.0 for _ in range(L*L)])
num_oc = np.array([0.0 for _ in range(L*L)])

for _ in range(1, 1000):
  img, _x = get_img_class()
  qry = mk_query(_x) 
  
  rand_trace = rrandom_trace(qry)
  sink_trace = baseline_get_trace(qry)
  greedy_trace = impnet.get_active_trace(sess, qry, epi=0.0, play=True)
  oc_trace = impnet.get_active_trace(sess, qry, epi=0.0, play=False)

  for i in range(OBS_SIZE):
    rand_prefix = rand_trace[:i]
    sink_prefix = sink_trace[:i]
    greedy_prefix = greedy_trace[:i]
    oc_prefix = oc_trace[:i]

    rand_all_p = baseline_get_all_preds(rand_prefix)
    sink_all_p = baseline_get_all_preds(sink_prefix)
    greedy_all_p = impnet.get_all_preds(sess, greedy_prefix)
    oc_all_p = impnet.get_all_preds(sess, oc_prefix)

    rand_acc = pred_acc(rand_all_p, qry)
    num_random[i] += rand_acc / 100
    sink_acc = pred_acc(sink_all_p, qry)
    num_sink[i] += sink_acc / 100
    greedy_acc = pred_acc(greedy_all_p, qry)
    num_greedy[i] += greedy_acc / 100
    oc_acc = pred_acc(oc_all_p, qry)
    num_oc[i] += oc_acc / 100

  print "iteration ", _
  print "random ", num_random / _
  print "sink ", num_sink / _
  print "greedy ", num_greedy / _
  print "oc ", num_oc / _



