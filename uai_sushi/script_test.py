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

def get_best_item(preds):
  scores = [0 for i in range(L)]
  for ii in range(L):
    for jj in range(L):
      if len(preds[ii][jj]) == 2:
        if preds[ii][jj][0] > preds[ii][jj][1]:
          scores[ii] += 1
      if len(preds[ii][jj]) == 1:
        scores[ii] += preds[ii][jj]
  return np.argmax(scores)
  

img, _x = get_img_class(test=True, idx=0)
draw_orig(img, "drawings/orig.png")
qry = mk_query(_x) 

impnet = Implynet("imp")
sess = tf.Session()
impnet.load_model(sess, "model_imply.ckpt")


trace = impnet.get_active_trace(sess, qry, epi=0.0)

for i in range(len(trace)):
  trace_prefix = trace[:i]
  all_preds = impnet.get_all_preds(sess, trace_prefix)
  draw_allob(all_preds, "drawings/pred_ob{0}.png".format(i), trace_prefix)
  the_best_pred = get_best_item(all_preds)
  the_best_truth = get_best_item(img)
  # print the_best_pred, the_best_truth

# trace_rand = impnet.get_active_trace(sess, qry, epi=1.0)
# for i in range(len(trace_rand)):
#   trace_prefix = trace_rand[:i]
#   all_preds = impnet.get_all_preds(sess, trace_prefix)
#   draw_allob(all_preds, "drawings/rand_ob{0}.png".format(i), trace_prefix)

print "we gonna run on 2500 test data"
num_active = np.array([0.0 for _ in range(L*L)])
num_best = np.array([0.0 for _ in range(L*L)])

for _ in range(2500):
  img, _x = get_img_class(test=True, idx=_)
  qry = mk_query(_x) 
  active_trace = impnet.get_active_trace(sess, qry, epi=0.0, play=False)

  for i in range(len(active_trace)):
    trace_prefix = active_trace[:i]
    all_preds = impnet.get_all_preds(sess, trace_prefix)
    acti_acc = pred_acc(all_preds, qry)
    num_active[i] += acti_acc

    the_best_pred = get_best_item(all_preds)
    the_best_truth = get_best_item(img)
    if the_best_pred == the_best_truth:
      num_best[i] += 1

  print "H m m ", num_best

  print "iteration ", _
  print "active ", num_active / (_ +1)
  print "best ", num_best / (_ +1)



