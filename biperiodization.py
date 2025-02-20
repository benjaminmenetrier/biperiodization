#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter

# Case name
case = "large_adjusted"

# C+I dimensions
nx = 50
ny = 70

# E zone dimensions
nxExt = 20
nyExt = 28

# Mixing scale
nmix = 10
Lmix = 1.0

# Boyd scale
Lboyd = 2.0

# Define smooth random field on C+I
np.random.seed(10)
rnd = np.random.random((nx, ny))
blurred = gaussian_filter(rnd, sigma=2)

# Include in C+I+E+E' domain
init = np.full(shape=(nx+2*nxExt, ny+2*nyExt), fill_value=np.nan)
init[nxExt:nx+nxExt, nyExt:ny+nyExt] = blurred

# Fill E and E' with symmetric or anti-symmetry value
sym = np.full(shape=(nx+2*nxExt, ny+2*nyExt), fill_value=np.nan)
antisym = np.full(shape=(nx+2*nxExt, ny+2*nyExt), fill_value=np.nan)
for jx in range(0, nx+2*nxExt):
  for jy in range(0, ny+2*nyExt):
    if jx >= nxExt and jx <= nx+nxExt-1 and jy >= nyExt and jy <= ny+nyExt-1:
      # Fill interior
      sym[jx, jy] = init[jx, jy]
      antisym[jx, jy] = init[jx, jy] 
    elif jx < nxExt and jy < nyExt:
      # Bottom left corner
      sym[jx, jy] = init[2*nxExt-jx, 2*nyExt-jy]
      antisym[jx, jy] = 2*init[nxExt, nyExt]-init[2*nxExt-jx, 2*nyExt-jy]
    elif jx < nxExt and jy > ny+nyExt-1:
      # Bottom right corner
      sym[jx, jy] = init[2*nxExt-jx, 2*(ny+nyExt-1)-jy]
      antisym[jx, jy] = 2*init[nxExt, ny+nyExt-1]-init[2*nxExt-jx, 2*(ny+nyExt-1)-jy]
    elif jx > nx+nxExt-1 and jy > ny+nyExt-1:
      # Top right corner
      sym[jx, jy] = init[2*(nx+nxExt-1)-jx, 2*(ny+nyExt-1)-jy]
      antisym[jx, jy] = 2*init[nx+nxExt-1, ny+nyExt-1]-init[2*(nx+nxExt-1)-jx, 2*(ny+nyExt-1)-jy]
    elif jx > nx+nxExt-1 and jy < nyExt:
      # Top left corner
      sym[jx, jy] = init[2*(nx+nxExt-1)-jx, 2*nyExt-jy]
      antisym[jx, jy] = 2*init[nx+nxExt-1, nyExt]-init[2*(nx+nxExt-1)-jx, 2*nyExt-jy]
    elif jx < nxExt:
      # Bottom side
      sym[jx, jy] = init[2*nxExt-jx, jy]
      antisym[jx, jy] = 2*init[nxExt, jy]-init[2*nxExt-jx, jy]
    elif jy < nyExt:
      # Left side
      sym[jx, jy] = init[jx, 2*nyExt-jy]
      antisym[jx, jy] = 2*init[jx, nyExt]-init[jx, 2*nyExt-jy]
    elif jx > nx+nxExt-1:
      # Top side
      sym[jx, jy] = init[2*(nx+nxExt-1)-jx, jy]
      antisym[jx, jy] = 2*init[nx+nxExt-1, jy]-init[2*(nx+nxExt-1)-jx, jy]
    elif jy > ny+nyExt-1:
      # Right side
      sym[jx, jy] = init[jx, 2*(ny+nyExt-1)-jy]
      antisym[jx, jy] = 2*init[jx, ny+nyExt-1]-init[jx, 2*(ny+nyExt-1)-jy]

# Mixing mask
mixingX = np.full(shape=nxExt, fill_value=np.nan)
for jx in range(0, nxExt):
  ux = float(jx+1)/float(min(nxExt,nmix)+1)
  if ux < 1:
    mixingX[jx] = 0.5*(1.0+math.erf(Lmix*(1.0-2.0*ux)/math.sqrt(1.0-(2.0*ux-1.0)*(2.0*ux-1.0))));
  else:
    mixingX[jx] = 0.0
mixingY = np.full(shape=nyExt+1, fill_value=np.nan)
for jy in range(0, nyExt):
  uy = float(jy+1)/float(min(nyExt,nmix)+1)
  if uy < 1:
    mixingY[jy] = 0.5*(1.0+math.erf(Lmix*(1.0-2.0*uy)/math.sqrt(1.0-(2.0*uy-1.0)*(2.0*uy-1.0))));
  else:
    mixingY[jy] = 0.0
mixing = np.full(shape=(nx+2*nxExt, ny+2*nyExt), fill_value=np.nan)
for jx in range(0, nx+2*nxExt):
  for jy in range(0, ny+2*nyExt):
    if jx >= nxExt and jx <= nx+nxExt-1 and jy >= nyExt and jy <= ny+nyExt-1:
      # Fill interior
      mixing[jx, jy] = 1
    elif jx < nxExt and jy < nyExt:
      # Bottom left corner
      mixing[jx, jy] = mixingX[nxExt-jx-1]*mixingY[nyExt-jy-1]
    elif jx < nxExt and jy > ny+nyExt-1:
      # Bottom right corner
      mixing[jx, jy] = mixingX[nxExt-jx-1]*mixingY[jy-(ny+nyExt)]
    elif jx > nx+nxExt-1 and jy > ny+nyExt-1:
      # Top right corner
      mixing[jx, jy] = mixingX[jx-(nx+nxExt)]*mixingY[jy-(ny+nyExt)]
    elif jx > nx+nxExt-1 and jy < nyExt:
      # Top left corner
      mixing[jx, jy] = mixingX[jx-(nx+nxExt)]*mixingY[nyExt-jy-1]
    elif jx < nxExt:
      # Bottom side
      mixing[jx, jy] = mixingX[nxExt-jx-1]
    elif jy < nyExt:
      # Left side
      mixing[jx, jy] = mixingY[nyExt-jy-1]
    elif jx > nx+nxExt-1:
      # Top side
      mixing[jx, jy] = mixingX[jx-(nx+nxExt)]
    elif jy > ny+nyExt-1:
      # Right side
      mixing[jx, jy] = mixingY[jy-(ny+nyExt)]

# Mixed extended field
mixed = antisym*mixing+sym*(1.0-mixing)

# Boyd mask
boydX = np.full(shape=nxExt, fill_value=np.nan)
for jx in range(0, nxExt):
  ux = float(jx+1)/float(nxExt+1)
  boydX[jx] = 0.5*(1.0+math.erf(Lboyd*(1.0-2.0*ux)/math.sqrt(1.0-(2.0*ux-1.0)*(2.0*ux-1.0))));
boydY = np.full(shape=nyExt+1, fill_value=np.nan)
for jy in range(0, nyExt):
  uy = float(jy+1)/float(nyExt+1)
  boydY[jy] = 0.5*(1.0+math.erf(Lboyd*(1.0-2.0*uy)/math.sqrt(1.0-(2.0*uy-1.0)*(2.0*uy-1.0))));
boyd = np.full(shape=(nx+2*nxExt, ny+2*nyExt), fill_value=np.nan)
for jx in range(0, nx+2*nxExt):
  for jy in range(0, ny+2*nyExt):
    if jx >= nxExt and jx <= nx+nxExt-1 and jy >= nyExt and jy <= ny+nyExt-1:
      # Fill interior
      boyd[jx, jy] = 1
    elif jx < nxExt and jy < nyExt:
      # Bottom left corner
      boyd[jx, jy] = boydX[nxExt-jx-1]*boydY[nyExt-jy-1]
    elif jx < nxExt and jy > ny+nyExt-1:
      # Bottom right corner
      boyd[jx, jy] = boydX[nxExt-jx-1]*boydY[jy-(ny+nyExt)]
    elif jx > nx+nxExt-1 and jy > ny+nyExt-1:
      # Top right corner
      boyd[jx, jy] = boydX[jx-(nx+nxExt)]*boydY[jy-(ny+nyExt)]
    elif jx > nx+nxExt-1 and jy < nyExt:
      # Top left corner
      boyd[jx, jy] = boydX[jx-(nx+nxExt)]*boydY[nyExt-jy-1]
    elif jx < nxExt:
      # Bottom side
      boyd[jx, jy] = boydX[nxExt-jx-1]
    elif jy < nyExt:
      # Left side
      boyd[jx, jy] = boydY[nyExt-jy-1]
    elif jx > nx+nxExt-1:
      # Top side
      boyd[jx, jy] = boydX[jx-(nx+nxExt)]
    elif jy > ny+nyExt-1:
      # Right side
      boyd[jx, jy] = boydY[jy-(ny+nyExt)]

# Apply Boyd mask on mixed field and fold
folded = mixed*boyd
folded[0:nxExt,:] += folded[nx+nxExt:nx+2*nxExt,:]
folded[nx+nxExt:nx+2*nxExt,:] = folded[0:nxExt,:]
folded[:,0:nyExt] += folded[:,ny+nyExt:ny+2*nyExt]
folded[:,ny+nyExt:ny+2*nyExt] = folded[:,0:nyExt]

# Get min/max values
vmax = np.max(antisym)
vmin = np.min(antisym)
levels = np.linspace(vmin, vmax, 21)
cmap = "turbo"

# Plot smooth random field
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Initial field')
ax.contourf(init, levels=levels, cmap=cmap)
ax.contour(init, levels=levels, colors='k', linewidths=0.5)
ax.plot([nyExt, nyExt], [nxExt, nx+2*nxExt-1], 'k--', linewidth=3.0)
ax.plot([nyExt, ny+2*nyExt-1], [nxExt, nxExt], 'k--', linewidth=3.0)
ax.plot([nyExt, ny+nyExt-1, ny+nyExt-1], [nx+nxExt-1, nx+nxExt-1, nxExt], 'k--', linewidth=3.0)
ax.text(ny+1.4*nyExt, nx+1.3*nxExt, "E", size=20, ha="center", va="center")
ax.text(0.5*ny+nyExt, 0.5*nx+nxExt, "P", size=20, ha="center", va="center")
ax.text(0.6*nyExt, 0.5*nxExt, "E'", size=20, ha="center", va="center")
plt.savefig(case + '_init.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + case + '_init.pdf ' + case + '_init.pdf')

# Plot extended field (anti-symmetric)
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Extended field (anti-symmetric)')
ax.contourf(antisym, levels=levels, cmap=cmap)
ax.contour(antisym, levels=levels, colors='k', linewidths=0.5)
ax.plot([nyExt, nyExt], [nxExt, nx+2*nxExt-1], 'k--', linewidth=3.0)
ax.plot([nyExt, ny+2*nyExt-1], [nxExt, nxExt], 'k--', linewidth=3.0)
ax.plot([nyExt, ny+nyExt-1, ny+nyExt-1], [nx+nxExt-1, nx+nxExt-1, nxExt], 'k--', linewidth=3.0)
ax.text(ny+1.4*nyExt, nx+1.3*nxExt, "E", size=20, ha="center", va="center")
ax.text(0.5*ny+nyExt, 0.5*nx+nxExt, "P", size=20, ha="center", va="center")
ax.text(0.6*nyExt, 0.5*nxExt, "E'", size=20, ha="center", va="center")
plt.savefig(case + '_extended_antisym.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + case + '_extended_antisym.pdf ' + case + '_extended_antisym.pdf')

# Plot extended field (symmetric)
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Extended field (symmetric)')
ax.contourf(sym, levels=levels, cmap=cmap)
ax.contour(sym, levels=levels, colors='k', linewidths=0.5)
ax.plot([nyExt, nyExt], [nxExt, nx+2*nxExt-1], 'k--', linewidth=3.0)
ax.plot([nyExt, ny+2*nyExt-1], [nxExt, nxExt], 'k--', linewidth=3.0)
ax.plot([nyExt, ny+nyExt-1, ny+nyExt-1], [nx+nxExt-1, nx+nxExt-1, nxExt], 'k--', linewidth=3.0)
ax.text(ny+1.4*nyExt, nx+1.3*nxExt, "E", size=20, ha="center", va="center")
ax.text(0.5*ny+nyExt, 0.5*nx+nxExt, "P", size=20, ha="center", va="center")
ax.text(0.6*nyExt, 0.5*nxExt, "E'", size=20, ha="center", va="center")
plt.savefig(case + '_extended_sym.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + case + '_extended_sym.pdf ' + case + '_extended_sym.pdf')

# Plot mixing mask
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Mixing mask')
ax.contourf(mixing, levels=np.linspace(0.0, 1.0, 21), cmap="YlOrBr")
ax.plot([nyExt, nyExt], [nxExt, nx+2*nxExt-1], 'k--', linewidth=3.0)
ax.plot([nyExt, ny+2*nyExt-1], [nxExt, nxExt], 'k--', linewidth=3.0)
ax.plot([nyExt, ny+nyExt-1, ny+nyExt-1], [nx+nxExt-1, nx+nxExt-1, nxExt], 'k--', linewidth=3.0)
ax.text(ny+1.4*nyExt, nx+1.3*nxExt, "E", size=20, ha="center", va="center")
ax.text(0.5*ny+nyExt, 0.5*nx+nxExt, "P", size=20, ha="center", va="center")
ax.text(0.6*nyExt, 0.5*nxExt, "E'", size=20, ha="center", va="center")
plt.savefig(case + '_mixing.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + case + '_mixing.pdf ' + case + '_mixing.pdf')

# Plot extended field (mixed)
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Extended field (mixed)')
ax.contourf(mixed, levels=levels, cmap=cmap)
ax.contour(mixed, levels=levels, colors='k', linewidths=0.5)
ax.plot([nyExt, nyExt], [nxExt, nx+2*nxExt-1], 'k--', linewidth=3.0)
ax.plot([nyExt, ny+2*nyExt-1], [nxExt, nxExt], 'k--', linewidth=3.0)
ax.plot([nyExt, ny+nyExt-1, ny+nyExt-1], [nx+nxExt-1, nx+nxExt-1, nxExt], 'k--', linewidth=3.0)
ax.text(ny+1.4*nyExt, nx+1.3*nxExt, "E", size=20, ha="center", va="center")
ax.text(0.5*ny+nyExt, 0.5*nx+nxExt, "P", size=20, ha="center", va="center")
ax.text(0.6*nyExt, 0.5*nxExt, "E'", size=20, ha="center", va="center")
plt.savefig(case + '_extended_mixed.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + case + '_extended_mixed.pdf ' + case + '_extended_mixed.pdf')

# Plot Boyd mask
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Boyd mask')
ax.contourf(boyd, levels=np.linspace(0.0, 1.0, 21), cmap="YlOrBr")
ax.plot([nyExt, nyExt], [nxExt, nx+2*nxExt-1], 'k--', linewidth=3.0)
ax.plot([nyExt, ny+2*nyExt-1], [nxExt, nxExt], 'k--', linewidth=3.0)
ax.plot([nyExt, ny+nyExt-1, ny+nyExt-1], [nx+nxExt-1, nx+nxExt-1, nxExt], 'k--', linewidth=3.0)
ax.text(ny+1.4*nyExt, nx+1.3*nxExt, "E", size=20, ha="center", va="center")
ax.text(0.5*ny+nyExt, 0.5*nx+nxExt, "P", size=20, ha="center", va="center")
ax.text(0.6*nyExt, 0.5*nxExt, "E'", size=20, ha="center", va="center")
plt.savefig(case + '_boyd.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + case + '_boyd.pdf ' + case + '_boyd.pdf')

# Plot folded field
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Folded field')
ax.contourf(folded, levels=levels, cmap=cmap)
ax.contour(folded, levels=levels, colors='k', linewidths=0.5)
ax.plot([nyExt, nyExt], [nxExt, nx+2*nxExt-1], 'k--', linewidth=3.0)
ax.plot([nyExt, ny+2*nyExt-1], [nxExt, nxExt], 'k--', linewidth=3.0)
ax.plot([nyExt, ny+nyExt-1, ny+nyExt-1], [nx+nxExt-1, nx+nxExt-1, nxExt], 'k--', linewidth=3.0)
ax.text(ny+1.4*nyExt, nx+1.3*nxExt, "E", size=20, ha="center", va="center")
ax.text(0.5*ny+nyExt, 0.5*nx+nxExt, "P", size=20, ha="center", va="center")
ax.text(0.6*nyExt, 0.5*nxExt, "E'", size=20, ha="center", va="center")
plt.savefig(case + '_folded.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + case + '_folded.pdf ' + case + '_folded.pdf')
