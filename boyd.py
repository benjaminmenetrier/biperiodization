#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import numpy as np
import os

# Parameter
n = 100
x = np.linspace(0.01, 0.99, n)

# Boyd function
L = [1, 2, 3]
b = np.zeros((n, len(L)))
for l in range(0, len(L)):
  for i in range(0, n):
    b[i, l] = 0.5*(1.0+math.erf(L[l]*(1.0-2.0*x[i])/np.sqrt(4.0*x[i]*(1.0-x[i]))))

# Legend
leg = []
for l in range(0, len(L)):
  leg.append("L = " + str(L[l]))

# Plot curves
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(4,3))
ax.set_title('Boyd function')
ax.plot(x, b, linewidth=2.0)
ax.axvline(x=0, color="gray")
ax.axhline(y=0, color="gray")
plt.legend(leg)
plt.savefig('boyd.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop boyd.pdf boyd.pdf')
