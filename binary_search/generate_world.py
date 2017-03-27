import numpy as np

def generate_truth(n_person, k_features):
  ret = dict()
  for p_id in range(n_person):
    feat = [np.random.randint(0,2) for _ in range(k_features)]
    ret[p_id] = feat
  return ret

def check_distinct(truth):
  seen = set()
  for p in truth:
    if tuple(truth[p]) in seen:
      return False
    else:
      seen.add(tuple(truth[p]))
  return True

truth = generate_truth(30, 10)
print truth
print check_distinct(truth)
