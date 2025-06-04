#!/usr/bin/env python3

import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter

# Random seed
seed = 10

# (P) dimension
nx = 198

# (E) dimension
nxExt = int(nx/5.0)

# X axis
x = np.linspace(0, 1.0, nx+nxExt)

# Mixing size (have an impact if lower than nxExt or nyExt)
nmix = 12

# Mixing scale
Lmix = 1.5

# Boyd scale
Lboyd = 1.5

# Line width
w = 3.0

# Define smooth random field on P
np.random.seed(seed)
rnd = np.random.random((nx+nxExt))
init1 = gaussian_filter(rnd, sigma=0.04*nx)
init2 = gaussian_filter(rnd, sigma=0.015*nx)
init = np.full(shape=nx+nxExt, fill_value=np.NaN)
for jx in range(nx+nxExt):
  ux = float(jx+1)/float(nx+nxExt+1)
  coef = 0.5*(1.0+math.erf(Lmix*(1.0-2.0*ux)/math.sqrt(4.0*ux*(1.0-ux))));
  init[jx] = coef*init1[jx]+(1.0-coef)*init2[jx]
init[nx:] = np.NaN
for jx in range(nx):
  init[jx] += x[jx]*0.25
init[nx-1] += 0.4*(init[nx-1]-init[nx-2])
init[0:nx] -= np.sum(init[0:nx])/nx
ymin = -0.2
ymax = 0.32

# Shifting value
shift = int((nx+nxExt)/2-(nx+nxExt/2))

# Plot
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(8,4))
ax.plot(x, np.roll(init, shift), 'black', linewidth=w)
ax.plot(x[nx+nxExt+shift], init[0], 'black', marker="o")
ax.plot(x[-shift-1], init[nx-1], 'black', marker="o")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(ymin, ymax)
plt.savefig('detail_init.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop detail_init.pdf detail_init.pdf')

# Fill (E) with a spline
spline = copy.deepcopy(init)
zk = float(nxExt+1)
zkp1 = zk+1.0
zlamb = zk/zkp1
zepsa = ((init[0]-init[nx-1])/zk-init[nx-1]+init[nx-2])*6.0/zkp1
zepsb = (init[1]-init[0]-(init[0]-init[nx-1])/zk)*6.0/zkp1
zmm = 4.0-zlamb*zlamb
zm1 = (2.0*zepsa-zlamb*zepsb)/zmm
zm2 = (2.0*zepsb-zlamb*zepsa)/zmm
za = init[nx-1]
zb = (init[0]-init[nx-1])/zk-(2.0*zm1+zm2)*zk/6.0
zc = 0.5*zm1
zd = (zm2-zm1)/(6.0*zk)
for jx in range(nxExt):
  zj = float(jx+1)
  spline[nx+jx] = za+zj*(zb+zj*(zc+zd*zj))

# Plot
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(8,4))
ax.plot(x, np.roll(spline, shift), 'black', linestyle=(0, (2, 2)), linewidth=w)
ax.plot(x, np.roll(init, shift), 'black', linewidth=w)
ax.plot(x[nx+nxExt+shift], init[0], 'black', marker="o")
ax.plot(x[-shift-1], init[nx-1], 'black', marker="o")
ax.hlines(y=np.max(init[0:nx]), xmin=0.4, xmax=0.52, linewidth=2, color='indianred')
ax.set_xlim(0.0, 1.0)
ax.set_ylim(ymin, ymax)
plt.savefig('detail_spline.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop detail_spline.pdf detail_spline.pdf')

# Fill (E) with symmetric or anti-symmetry value
sym = copy.deepcopy(init)
antisym = copy.deepcopy(init)
for jx in range(nxExt):
   sym[nx+jx] = init[nx-2-jx]
   antisym[nx+jx] = 2.0*init[nx-1]-init[nx-2-jx]
sym[0] = np.NaN
antisym[0] = np.NaN

# Mixing mask
mixing = np.full(shape=nxExt, fill_value=np.NaN)
for jx in range(nxExt):
  ux = float(jx+1)/float(min(nxExt,nmix)+1)
  if ux < 1:
    mixing[jx] = 0.5*(1.0+math.erf(Lmix*(1.0-2.0*ux)/math.sqrt(4.0*ux*(1.0-ux))));
  else:
    mixing[jx] = 0.0

# Mixed value
mixedLeft = copy.deepcopy(init)
for jx in range(nxExt):
  mixedLeft[nx+jx] = mixing[jx]*antisym[nx+jx]+(1.0-mixing[jx])*sym[nx+jx]
mixedLeft[0] = np.NaN

# Plot
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(8,4))
ax.plot(x, np.roll(sym, shift), 'deepskyblue', linewidth=w)
ax.plot(x, np.roll(antisym, shift), 'indianred', linewidth=w)
ax.plot(x, np.roll(mixedLeft, shift), 'dimgray', linestyle=(0, (2, 2)), linewidth=w)
ax.plot(x, np.roll(init, shift), 'black', linewidth=w)
ax.plot(x[nx+nxExt+shift], init[0], 'black', marker="o")
ax.plot(x[-shift-1], init[nx-1], 'black', marker="o")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(ymin, ymax)
plt.savefig('detail_mix_left.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop detail_mix_left.pdf detail_mix_left.pdf')

# Wide mixing mask
mixingWide = np.full(shape=nxExt, fill_value=np.NaN)
for jx in range(nxExt):
  ux = float(jx+1)/float(nxExt+1)
  if ux < 1:
    mixingWide[jx] = 0.5*(1.0+math.erf(Lmix*(1.0-2.0*ux)/math.sqrt(4.0*ux*(1.0-ux))));
  else:
    mixingWide[jx] = 0.0

# Mixed value
mixedWideLeft = copy.deepcopy(init)
for jx in range(nxExt):
  mixedWideLeft[nx+jx] = mixingWide[jx]*antisym[nx+jx]+(1.0-mixingWide[jx])*sym[nx+jx]
mixedWideLeft[0] = np.NaN

# Plot
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(8,4))
ax.plot(x, np.roll(sym, shift), 'deepskyblue', linewidth=w)
ax.plot(x, np.roll(antisym, shift), 'indianred', linewidth=w)
ax.plot(x, np.roll(mixedWideLeft, shift), 'dimgray', linestyle=(0, (2, 2)), linewidth=w)
ax.plot(x, np.roll(init, shift), 'black', linewidth=w)
ax.plot(x[nx+nxExt+shift], init[0], 'black', marker="o")
ax.plot(x[-shift-1], init[nx-1], 'black', marker="o")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(ymin, ymax)
plt.savefig('detail_mix_wide_left.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop detail_mix_wide_left.pdf detail_mix_wide_left.pdf')

# Fill (E) with symmetric or anti-symmetry value
sym = copy.deepcopy(init)
antisym = copy.deepcopy(init)
for jx in range(nxExt):
   sym[nx+nxExt-1-jx] = init[jx+1]
   antisym[nx+nxExt-1-jx] = 2.0*init[0]-init[jx+1]
sym[nx-1] = np.NaN
antisym[nx-1] = np.NaN

# Mixing mask
mixing = np.full(shape=nxExt, fill_value=np.NaN)
for jx in range(nxExt):
  ux = float(jx+1)/float(min(nxExt,nmix)+1)
  if ux < 1:
    mixing[jx] = 0.5*(1.0+math.erf(Lmix*(1.0-2.0*ux)/math.sqrt(4.0*ux*(1.0-ux))));
  else:
    mixing[jx] = 0.0

# Mixed value
mixedRight = copy.deepcopy(init)
for jx in range(nxExt):
  mixedRight[nx+nxExt-1-jx] = mixing[jx]*antisym[nx+nxExt-1-jx]+(1.0-mixing[jx])*sym[nx+nxExt-1-jx]
mixedRight[nx-1] = np.NaN

# Plot
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(8,4))
ax.plot(x, np.roll(sym, shift), 'deepskyblue', linewidth=w)
ax.plot(x, np.roll(antisym, shift), 'indianred', linewidth=w)
ax.plot(x, np.roll(mixedRight, shift), 'dimgray', linestyle=(0, (2, 2)), linewidth=w)
ax.plot(x, np.roll(init, shift), 'black', linewidth=w)
ax.plot(x[nx+nxExt+shift], init[0], 'black', marker="o")
ax.plot(x[-shift-1], init[nx-1], 'black', marker="o")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(ymin, ymax)
plt.savefig('detail_mix_right.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop detail_mix_right.pdf detail_mix_right.pdf')

# Boyd mask
boyd = np.full(shape=nxExt, fill_value=np.NaN)
for jx in range(nxExt):
  ux = float(jx+1)/float(nxExt+1)
  boyd[jx] = 0.5*(1.0+math.erf(Lboyd*(1.0-2.0*ux)/math.sqrt(4.0*ux*(1.0-ux))));

# Final value
final = copy.deepcopy(init)
for jx in range(nxExt):
  final[nx+jx] = boyd[jx]*mixedLeft[nx+jx]+(1.0-boyd[jx])*mixedRight[nx+jx]


# Plot
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(8,4))
ax.plot(x, np.roll(mixedLeft, shift), 'dimgray', linestyle=(0, (2, 2)), linewidth=w)
ax.plot(x, np.roll(mixedRight, shift), 'dimgray', linestyle=(0, (2, 2)), linewidth=w)
ax.plot(x, np.roll(final, shift), 'black', linestyle=(0, (2, 2)), linewidth=w)
ax.plot(x, np.roll(init, shift), 'black', linewidth=w)
ax.plot(x[nx+nxExt+shift], init[0], 'black', marker="o")
ax.plot(x[-shift-1], init[nx-1], 'black', marker="o")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(ymin, ymax)
plt.savefig('detail_final.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop detail_final.pdf detail_mix_final.pdf')
