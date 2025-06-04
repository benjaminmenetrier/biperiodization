#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import numpy as np
import os

# Parameter
n = 100
x = np.linspace(0.0, 1.0, n+2)

# Boyd function
L = [1, 1.5, 2, 2.5]
b = np.zeros((n+2, len(L)))
for l in range(0, len(L)):
  b[0, l] = 1.0
  for i in range(0, n):
    b[i+1, l] = 0.5*(1.0+math.erf(L[l]*(1.0-2.0*x[i])/np.sqrt(4.0*x[i]*(1.0-x[i]))))
  b[n+1, l] = 0.0

# Legend
leg = []
for l in range(0, len(L)):
  leg.append("L = " + str(L[l]))

# Plot curves
colors = ["limegreen", "deepskyblue", "gold", "indianred"]
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,3))
for l in range(0, len(L)):
  ax.plot(x, b[:,l], linewidth=3.0, color=colors[l])
ax.axhline(y=0, color="gray", linewidth=0.5)
ax.axhline(y=1, color="gray", linewidth=0.5)
ax.set_xlim(0.0, 1.0)
plt.legend(leg)
plt.savefig('boyd.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop boyd.pdf boyd.pdf')
