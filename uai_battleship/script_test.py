from model import *
from draw import *
# ------------- helpers --------------
# some on-policy data generation
def full_output_to_trace(full_out, Img, S):
  obs = [x[0] for x in full_out[1:]]
  return Trace(Img, S, obs)

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


