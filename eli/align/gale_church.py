# -*- coding: utf-8 -*-
# Gale-Church Algorithm
# Original version from GaChalign: http://goo.gl/17t9UG
# Authors: Liling Tan

import math

def norm_cdf(z):
  """ Use the norm distribution functions as of Gale-Church (1993) srcfile. """
  # Equation 26.2.17 from Abramowitz and Stegun (1964:p.932)
  
  t = 1/float(1+0.2316419*z) # t = 1/(1+pz) , z=0.2316419
  probdist = 1 - 0.3989423*math.exp(-z*z/2) * ((0.319381530 * t)+ \
                                         (-0.356563782* math.pow(t,2))+ \
                                         (1.781477937 * math.pow(t,3)) + \
                                         (-1.821255978* math.pow(t,4)) + \
                                         (1.330274429 * math.pow(t,5)))
  return probdist

def norm_logsf(z):
  """ Take log of the survival function for normal distribution. """
  try:
    return math.log(1 - norm_cdf(z))
  except ValueError:
    return float('-inf')

LOG2 = math.log(2)

BEAD_COSTS = {(1, 1): 0, (2, 1): 230, (1, 2): 230, (0, 1): 450, 
              (1, 0): 450, (2, 2): 440 }

def sent_length(sentence):
  """ Returns sentence length without spaces. """
  return sum(1 for c in sentence if c != ' ')


def length_cost(sx, sy, mean_xy, variance_xy):
  """  
  Calculate length cost given 2 sentence. Lower cost = higher prob.
   
  The original Gale-Church (1993:pp. 81) paper considers l2/l1 = 1 hence:
   delta = (l2-l1*c)/math.sqrt(l1*s2)
  
  If l2/l1 != 1 then the following should be considered:
   delta = (l2-l1*c)/math.sqrt((l1+l2*c)/2 * s2)
   substituting c = 1 and c = l2/l1, gives the original cost function.
  """
  lx, ly = sum(sx), sum(sy)
  m = (lx + ly * mean_xy) / 2 
  try:
    delta = (lx - ly * mean_xy) / math.sqrt(m * variance_xy)
  except ZeroDivisionError:
    return float('-inf')
  return - 100 * (LOG2 + norm_logsf(abs(delta)))

def _align(x, y, mean_xy, variance_xy, bead_costs):
  """ 
  The minimization function to choose the sentence pair with 
  cheapest alignment cost. 
  """
  m = {}
  for i in range(len(x) + 1):
    for j in range(len(y) + 1):
      if i == j == 0:
        m[0, 0] = (0, 0, 0)
      else:
        m[i, j] = min((m[i-di, j-dj][0] +
                      length_cost(x[i-di:i], y[j-dj:j], mean_xy, variance_xy) \
                      + bead_cost, di, dj)
                      for (di, dj), bead_cost in BEAD_COSTS.iteritems()
                      if i-di>=0 and j-dj>=0)

  i, j = len(x), len(y)
  while True:
    (c, di, dj) = m[i, j]
    if di == dj == 0:
      break
    yield (i-di, i), (j-dj, j)
    i -= di
    j -= dj


def align(sx, sy, mean_xy, variance_xy, bc):
  """ Main alignment function. """
  cx = map(sent_length,sx); cy = map(sent_length, sy) 
  for (i1, i2), (j1, j2) in \
  reversed(list(_align(cx, cy, mean_xy, variance_xy, bc))):
    yield ' '.join(sx[i1:i2]), ' '.join(sy[j1:j2])
    
