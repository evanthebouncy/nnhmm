import random
from copy import copy

def sorta(arrr):
  tracker = []

  def sub_partition(array, start, end, idx_pivot):

      'returns the position where the pivot winds up'

      if not (start <= idx_pivot <= end):
          raise ValueError('idx pivot must be between start and end')

      array[start], array[idx_pivot] = array[idx_pivot], array[start]
      tracker.append(copy(array))
      pivot = array[start]
      i = start + 1
      j = start + 1

      while j <= end:
          tracker.append(copy(array))
          if array[j] <= pivot:
              array[j], array[i] = array[i], array[j]
              i += 1
          j += 1

      array[start], array[i - 1] = array[i - 1], array[start]
      tracker.append(copy(array))
      return i - 1

  def quicksort(array, start=0, end=None):

      if end is None:
          end = len(array) - 1

      if end - start < 1:
          return

      idx_pivot = random.randint(start, end)
      i = sub_partition(array, start, end, idx_pivot)
      #print array, i, idx_pivot
      quicksort(array, start, i - 1)
      quicksort(array, i + 1, end)

  quicksort(arrr)
  return tracker

# trace = sorta([10,1,8,7,6,2,4,3,5,9])
# for tr in trace:
#   print tr
