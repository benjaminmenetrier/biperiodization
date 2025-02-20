#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter

# Case name (base, narrow, broad, broad_adjusted)
case = "base"

# (P) dimensions
nx = 70
ny = 50

if case == "base":
  # (E) dimensions
  nxExt = 14
  nyExt = 10

  # Mixing size (have an impact if lower than nxExt or nyExt)
  nmix = 100
elif case == "narrow":
  # E zone dimensions
  nxExt = 7
  nyExt = 5

  # Mixing size (have an impact if lower than nxExt or nyExt)
  nmix = 100
elif case == "broad":
  # E zone dimensions
  nxExt = 28
  nyExt = 20

  # Mixing size (have an impact if lower than nxExt or nyExt)
  nmix = 100
elif case == "broad_adjusted":
  # E zone dimensions
  nxExt = 28
  nyExt = 20

  # Mixing size (have an impact if lower than nxExt or nyExt)
  nmix = 10
else:
  print("Wrong case!")
  exit()

# Mixing scale
Lmix = 1.0

# Boyd scale
Lboyd = 2.0

# Define smooth random field on (P)
np.random.seed(10)
rnd = np.random.random((nx, ny))
blurred = gaussian_filter(rnd, sigma=2)

# Include in (P+E+E') domain
init = np.full(shape=(nx+2*nxExt, ny+2*nyExt), fill_value=np.nan)
init[nxExt:nx+nxExt, nyExt:ny+nyExt] = blurred

# Fill (E) and (E') with symmetric or anti-symmetry value
sym = np.full(shape=(nx+2*nxExt, ny+2*nyExt), fill_value=np.nan)
antisym = np.full(shape=(nx+2*nxExt, ny+2*nyExt), fill_value=np.nan)
for jx in range(0, nx+2*nxExt):
  for jy in range(0, ny+2*nyExt):
    if jx >= nxExt and jx <= nx+nxExt-1 and jy >= nyExt and jy <= ny+nyExt-1:
      # Fill interior
      sym[jx, jy] = init[jx, jy]
      antisym[jx, jy] = init[jx, jy] 
    elif jx < nxExt and jy < nyExt:
      # Left-bottom corner
      sym[jx, jy] = init[2*nxExt-jx, 2*nyExt-jy]
      antisym[jx, jy] = 2*init[nxExt, nyExt]-init[2*nxExt-jx, 2*nyExt-jy]
    elif jx < nxExt and jy > ny+nyExt-1:
      # Left-top corner
      sym[jx, jy] = init[2*nxExt-jx, 2*(ny+nyExt-1)-jy]
      antisym[jx, jy] = 2*init[nxExt, ny+nyExt-1]-init[2*nxExt-jx, 2*(ny+nyExt-1)-jy]
    elif jx > nx+nxExt-1 and jy > ny+nyExt-1:
      # Right-top corner
      sym[jx, jy] = init[2*(nx+nxExt-1)-jx, 2*(ny+nyExt-1)-jy]
      antisym[jx, jy] = 2*init[nx+nxExt-1, ny+nyExt-1]-init[2*(nx+nxExt-1)-jx, 2*(ny+nyExt-1)-jy]
    elif jx > nx+nxExt-1 and jy < nyExt:
      # Right-bottom corner
      sym[jx, jy] = init[2*(nx+nxExt-1)-jx, 2*nyExt-jy]
      antisym[jx, jy] = 2*init[nx+nxExt-1, nyExt]-init[2*(nx+nxExt-1)-jx, 2*nyExt-jy]
    elif jx < nxExt:
      # Left side
      sym[jx, jy] = init[2*nxExt-jx, jy]
      antisym[jx, jy] = 2*init[nxExt, jy]-init[2*nxExt-jx, jy]
    elif jx > nx+nxExt-1:
      # Right side
      sym[jx, jy] = init[2*(nx+nxExt-1)-jx, jy]
      antisym[jx, jy] = 2*init[nx+nxExt-1, jy]-init[2*(nx+nxExt-1)-jx, jy]
    elif jy < nyExt:
      # Bottom side
      sym[jx, jy] = init[jx, 2*nyExt-jy]
      antisym[jx, jy] = 2*init[jx, nyExt]-init[jx, 2*nyExt-jy]
    elif jy > ny+nyExt-1:
      # Top side
      sym[jx, jy] = init[jx, 2*(ny+nyExt-1)-jy]
      antisym[jx, jy] = 2*init[jx, ny+nyExt-1]-init[jx, 2*(ny+nyExt-1)-jy]

# Mixing mask components
mixingX = np.full(shape=nxExt, fill_value=np.nan)
for jx in range(0, nxExt):
  ux = float(jx+1)/float(min(nxExt,nmix)+1)
  if ux < 1:
    mixingX[jx] = 0.5*(1.0+math.erf(Lmix*(1.0-2.0*ux)/math.sqrt(4.0*ux*(1.0-ux))));
  else:
    mixingX[jx] = 0.0
mixingY = np.full(shape=nyExt+1, fill_value=np.nan)
for jy in range(0, nyExt):
  uy = float(jy+1)/float(min(nyExt,nmix)+1)
  if uy < 1:
    mixingY[jy] = 0.5*(1.0+math.erf(Lmix*(1.0-2.0*uy)/math.sqrt(4.0*uy*(1.0-uy))));
  else:
    mixingY[jy] = 0.0

# Mixing mask
mixing = np.full(shape=(nx+2*nxExt, ny+2*nyExt), fill_value=np.nan)
for jx in range(0, nx+2*nxExt):
  for jy in range(0, ny+2*nyExt):
    if jx >= nxExt and jx <= nx+nxExt-1 and jy >= nyExt and jy <= ny+nyExt-1:
      # Fill interior
      mixing[jx, jy] = 1
    elif jx < nxExt and jy < nyExt:
      # Left-bottom corner
      mixing[jx, jy] = mixingX[nxExt-jx-1]*mixingY[nyExt-jy-1]
    elif jx < nxExt and jy > ny+nyExt-1:
      # Left-top corner
      mixing[jx, jy] = mixingX[nxExt-jx-1]*mixingY[jy-(ny+nyExt)]
    elif jx > nx+nxExt-1 and jy > ny+nyExt-1:
      # Right-top corner
      mixing[jx, jy] = mixingX[jx-(nx+nxExt)]*mixingY[jy-(ny+nyExt)]
    elif jx > nx+nxExt-1 and jy < nyExt:
      # Right-bottom corner
      mixing[jx, jy] = mixingX[jx-(nx+nxExt)]*mixingY[nyExt-jy-1]
    elif jx < nxExt:
      # Left side
      mixing[jx, jy] = mixingX[nxExt-jx-1]
    elif jx > nx+nxExt-1:
      # Right side
      mixing[jx, jy] = mixingX[jx-(nx+nxExt)]
    elif jy < nyExt:
      # Bottom side
      mixing[jx, jy] = mixingY[nyExt-jy-1]
    elif jy > ny+nyExt-1:
      # Top side
      mixing[jx, jy] = mixingY[jy-(ny+nyExt)]

# Mixed extended field
mixed = antisym*mixing+sym*(1.0-mixing)

# Boyd mask components
boydX = np.full(shape=nxExt, fill_value=np.nan)
for jx in range(0, nxExt):
  ux = float(jx+1)/float(nxExt+1)
  boydX[jx] = 0.5*(1.0+math.erf(Lboyd*(1.0-2.0*ux)/math.sqrt(4.0*ux*(1.0-ux))));
boydY = np.full(shape=nyExt+1, fill_value=np.nan)
for jy in range(0, nyExt):
  uy = float(jy+1)/float(nyExt+1)
  boydY[jy] = 0.5*(1.0+math.erf(Lboyd*(1.0-2.0*uy)/math.sqrt(4.0*uy*(1.0-uy))));

# Boyd mask
boyd = np.full(shape=(nx+2*nxExt, ny+2*nyExt), fill_value=np.nan)
for jx in range(0, nx+2*nxExt):
  for jy in range(0, ny+2*nyExt):
    if jx >= nxExt and jx <= nx+nxExt-1 and jy >= nyExt and jy <= ny+nyExt-1:
      # Fill interior
      boyd[jx, jy] = 1
    elif jx < nxExt and jy < nyExt:
      # Left-bottom corner
      boyd[jx, jy] = boydX[nxExt-jx-1]*boydY[nyExt-jy-1]
    elif jx < nxExt and jy > ny+nyExt-1:
      # Left-top corner
      boyd[jx, jy] = boydX[nxExt-jx-1]*boydY[jy-(ny+nyExt)]
    elif jx > nx+nxExt-1 and jy > ny+nyExt-1:
      # Right-top corner
      boyd[jx, jy] = boydX[jx-(nx+nxExt)]*boydY[jy-(ny+nyExt)]
    elif jx > nx+nxExt-1 and jy < nyExt:
      # Right-bottom corner
      boyd[jx, jy] = boydX[jx-(nx+nxExt)]*boydY[nyExt-jy-1]
    elif jx < nxExt:
      # Left side
      boyd[jx, jy] = boydX[nxExt-jx-1]
    elif jx > nx+nxExt-1:
      # Right side
      boyd[jx, jy] = boydX[jx-(nx+nxExt)]
    elif jy < nyExt:
      # Bottom side
      boyd[jx, jy] = boydY[nyExt-jy-1]
    elif jy > ny+nyExt-1:
      # Top side
      boyd[jx, jy] = boydY[jy-(ny+nyExt)]

# Apply Boyd mask on mixed field and fold
folded = mixed*boyd
folded[0:nxExt,:] += folded[nx+nxExt:nx+2*nxExt,:]
folded[nx+nxExt:nx+2*nxExt,:] = folded[0:nxExt,:]
folded[:,0:nyExt] += folded[:,ny+nyExt:ny+2*nyExt]
folded[:,ny+nyExt:ny+2*nyExt] = folded[:,0:nyExt]

# All steps in a single one, for (P+E) only
single = np.full(shape=(nx+nxExt, ny+nyExt), fill_value=np.nan)
for jx in range(0, nx+nxExt):
  for jy in range(0, ny+nyExt):
    if jx <= nx-1 and jy <= ny-1:
      # Fill interior
      single[jx, jy] = blurred[jx, jy]
    elif jx > nx-1 and jy > ny-1:
      single[jx, jy] = 0.0

      # Left-bottom corner
      s_sym = blurred[nxExt-(jx-nx), nyExt-(jy-ny)]
      s_antisym = 2*blurred[0, 0]-blurred[nxExt-(jx-nx), nyExt-(jy-ny)]
      s_mixing = mixingX[nxExt-(jx-nx)-1]*mixingY[nyExt-(jy-ny)-1]
      s_boyd = boydX[nxExt-(jx-nx)-1]*boydY[nyExt-(jy-ny)-1]
      single[jx, jy] += (s_antisym*s_mixing+s_sym*(1.0-s_mixing))*s_boyd

      # Left-top corner
      s_sym = blurred[nxExt-(jx-nx), 2*(ny-1)-jy]
      s_antisym = 2*blurred[0, ny-1]-blurred[nxExt-(jx-nx), 2*(ny-1)-jy]
      s_mixing = mixingX[nxExt-(jx-nx)-1]*mixingY[jy-ny]
      s_boyd = boydX[nxExt-(jx-nx)-1]*boydY[jy-ny]
      single[jx, jy] += (s_antisym*s_mixing+s_sym*(1.0-s_mixing))*s_boyd

      # Right-top corner
      s_sym = blurred[2*(nx-1)-jx, 2*(ny-1)-jy]
      s_antisym = 2*blurred[nx-1, ny-1]-blurred[2*(nx-1)-jx, 2*(ny-1)-jy]
      s_mixing = mixingX[jx-nx]*mixingY[jy-ny]
      s_boyd = boydX[jx-nx]*boydY[jy-ny]
      single[jx, jy] += (s_antisym*s_mixing+s_sym*(1.0-s_mixing))*s_boyd

      # Right-bottom corner
      s_sym = blurred[2*(nx-1)-jx, nyExt-(jy-ny)]
      s_antisym = 2*blurred[nx-1, 0]-blurred[2*(nx-1)-jx, nyExt-(jy-ny)]
      s_mixing = mixingX[jx-nx]*mixingY[nyExt-(jy-ny)-1]
      s_boyd = boydX[jx-nx]*boydY[nyExt-(jy-ny)-1]
      single[jx, jy] += (s_antisym*s_mixing+s_sym*(1.0-s_mixing))*s_boyd
    elif jx > nx-1:
      single[jx, jy] = 0.0 

      # Left side
      s_sym = blurred[nxExt-(jx-nx), jy]
      s_antisym = 2*blurred[0, jy]-blurred[nxExt-(jx-nx), jy]
      s_mixing = mixingX[nxExt-(jx-nx)-1]
      s_boyd = boydX[nxExt-(jx-nx)-1]
      single[jx, jy] += (s_antisym*s_mixing+s_sym*(1.0-s_mixing))*s_boyd

      # Right side
      s_sym = blurred[2*(nx-1)-jx, jy]
      s_antisym = 2*blurred[nx-1, jy]-blurred[2*(nx-1)-jx, jy]
      s_mixing = mixingX[jx-nx]
      s_boyd = boydX[jx-nx]
      single[jx, jy] += (s_antisym*s_mixing+s_sym*(1.0-s_mixing))*s_boyd
    elif jy > ny-1:
      single[jx, jy] = 0.0

      # Bottom side
      s_sym = blurred[jx, nyExt-(jy-ny)]
      s_antisym = 2*blurred[jx, 0]-blurred[jx, nyExt-(jy-ny)]
      s_mixing = mixingY[nyExt-(jy-ny)-1]
      s_boyd = boydY[nyExt-(jy-ny)-1]
      single[jx, jy] += (s_antisym*s_mixing+s_sym*(1.0-s_mixing))*s_boyd

      # Top side
      s_sym = blurred[jx, 2*(ny-1)-jy]
      s_antisym = 2*blurred[jx, ny-1]-blurred[jx, 2*(ny-1)-jy]
      s_mixing = mixingY[jy-ny]
      s_boyd = boydY[jy-ny]
      single[jx, jy] += (s_antisym*s_mixing+s_sym*(1.0-s_mixing))*s_boyd

# Difference for check
print("Single step check: " + str(np.max(np.abs(folded[nxExt:2*nxExt+nx, nyExt:2*nyExt+ny] - single))))

# Get min/max values
vmax = np.max(blurred)*1.3
vmin = np.min(blurred)*0.6
levels = np.linspace(vmin, vmax, 31)
cmap = "turbo"

# Plot smooth random field
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Initial field')
ax.contourf(np.transpose(init), levels=levels, cmap=cmap)
ax.contour(np.transpose(init), levels=levels, colors='k', linewidths=0.5)
ax.plot([nxExt, nxExt], [nyExt, ny+2*nyExt-1], 'k--', linewidth=3.0)
ax.plot([nxExt, nx+2*nxExt-1], [nyExt, nyExt], 'k--', linewidth=3.0)
ax.plot([nxExt, nx+nxExt-1, nx+nxExt-1], [ny+nyExt-1, ny+nyExt-1, nyExt], 'k--', linewidth=3.0)
ax.text(nx+1.4*nxExt, ny+1.3*nyExt, "E", size=20, ha="center", va="center")
ax.text(0.5*nx+nxExt, 0.5*ny+nyExt, "P", size=20, ha="center", va="center")
ax.text(0.6*nxExt, 0.5*nyExt, "E'", size=20, ha="center", va="center")
plt.savefig(case + '_init.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + case + '_init.pdf ' + case + '_init.pdf')

# Plot extended field (anti-symmetric)
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Extended field (anti-symmetric)')
ax.contourf(np.transpose(antisym), levels=levels, cmap=cmap)
ax.contour(np.transpose(antisym), levels=levels, colors='k', linewidths=0.5)
ax.plot([nxExt, nxExt], [nyExt, ny+2*nyExt-1], 'k--', linewidth=3.0)
ax.plot([nxExt, nx+2*nxExt-1], [nyExt, nyExt], 'k--', linewidth=3.0)
ax.plot([nxExt, nx+nxExt-1, nx+nxExt-1], [ny+nyExt-1, ny+nyExt-1, nyExt], 'k--', linewidth=3.0)
ax.text(nx+1.4*nxExt, ny+1.3*nyExt, "E", size=20, ha="center", va="center")
ax.text(0.5*nx+nxExt, 0.5*ny+nyExt, "P", size=20, ha="center", va="center")
ax.text(0.6*nxExt, 0.5*nyExt, "E'", size=20, ha="center", va="center")
plt.savefig(case + '_extended_antisym.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + case + '_extended_antisym.pdf ' + case + '_extended_antisym.pdf')

# Plot extended field (symmetric)
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Extended field (symmetric)')
ax.contourf(np.transpose(sym), levels=levels, cmap=cmap)
ax.contour(np.transpose(sym), levels=levels, colors='k', linewidths=0.5)
ax.plot([nxExt, nxExt], [nyExt, ny+2*nyExt-1], 'k--', linewidth=3.0)
ax.plot([nxExt, nx+2*nxExt-1], [nyExt, nyExt], 'k--', linewidth=3.0)
ax.plot([nxExt, nx+nxExt-1, nx+nxExt-1], [ny+nyExt-1, ny+nyExt-1, nyExt], 'k--', linewidth=3.0)
ax.text(nx+1.4*nxExt, ny+1.3*nyExt, "E", size=20, ha="center", va="center")
ax.text(0.5*nx+nxExt, 0.5*ny+nyExt, "P", size=20, ha="center", va="center")
ax.text(0.6*nxExt, 0.5*nyExt, "E'", size=20, ha="center", va="center")
plt.savefig(case + '_extended_sym.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + case + '_extended_sym.pdf ' + case + '_extended_sym.pdf')

# Plot mixing mask
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Mixing mask')
ax.contourf(np.transpose(mixing), levels=np.linspace(0.0, 1.0, 21), cmap="YlOrBr")
ax.plot([nxExt, nxExt], [nyExt, ny+2*nyExt-1], 'k--', linewidth=3.0)
ax.plot([nxExt, nx+2*nxExt-1], [nyExt, nyExt], 'k--', linewidth=3.0)
ax.plot([nxExt, nx+nxExt-1, nx+nxExt-1], [ny+nyExt-1, ny+nyExt-1, nyExt], 'k--', linewidth=3.0)
ax.text(nx+1.4*nxExt, ny+1.3*nyExt, "E", size=20, ha="center", va="center")
ax.text(0.5*nx+nxExt, 0.5*ny+nyExt, "P", size=20, ha="center", va="center")
ax.text(0.6*nxExt, 0.5*nyExt, "E'", size=20, ha="center", va="center")
plt.savefig(case + '_mixing.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + case + '_mixing.pdf ' + case + '_mixing.pdf')

# Plot extended field (mixed)
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Extended field (mixed)')
ax.contourf(np.transpose(mixed), levels=levels, cmap=cmap)
ax.contour(np.transpose(mixed), levels=levels, colors='k', linewidths=0.5)
ax.plot([nxExt, nxExt], [nyExt, ny+2*nyExt-1], 'k--', linewidth=3.0)
ax.plot([nxExt, nx+2*nxExt-1], [nyExt, nyExt], 'k--', linewidth=3.0)
ax.plot([nxExt, nx+nxExt-1, nx+nxExt-1], [ny+nyExt-1, ny+nyExt-1, nyExt], 'k--', linewidth=3.0)
ax.text(nx+1.4*nxExt, ny+1.3*nyExt, "E", size=20, ha="center", va="center")
ax.text(0.5*nx+nxExt, 0.5*ny+nyExt, "P", size=20, ha="center", va="center")
ax.text(0.6*nxExt, 0.5*nyExt, "E'", size=20, ha="center", va="center")
plt.savefig(case + '_extended_mixed.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + case + '_extended_mixed.pdf ' + case + '_extended_mixed.pdf')

# Plot Boyd mask
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Boyd mask')
ax.contourf(np.transpose(boyd), levels=np.linspace(0.0, 1.0, 21), cmap="YlOrBr")
ax.plot([nxExt, nxExt], [nyExt, ny+2*nyExt-1], 'k--', linewidth=3.0)
ax.plot([nxExt, nx+2*nxExt-1], [nyExt, nyExt], 'k--', linewidth=3.0)
ax.plot([nxExt, nx+nxExt-1, nx+nxExt-1], [ny+nyExt-1, ny+nyExt-1, nyExt], 'k--', linewidth=3.0)
ax.text(nx+1.4*nxExt, ny+1.3*nyExt, "E", size=20, ha="center", va="center")
ax.text(0.5*nx+nxExt, 0.5*ny+nyExt, "P", size=20, ha="center", va="center")
ax.text(0.6*nxExt, 0.5*nyExt, "E'", size=20, ha="center", va="center")
plt.savefig(case + '_boyd.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + case + '_boyd.pdf ' + case + '_boyd.pdf')

# Plot folded field
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Folded field')
ax.contourf(np.transpose(folded), levels=levels, cmap=cmap)
ax.contour(np.transpose(folded), levels=levels, colors='k', linewidths=0.5)
ax.plot([nxExt, nxExt], [nyExt, ny+2*nyExt-1], 'k--', linewidth=3.0)
ax.plot([nxExt, nx+2*nxExt-1], [nyExt, nyExt], 'k--', linewidth=3.0)
ax.plot([nxExt, nx+nxExt-1, nx+nxExt-1], [ny+nyExt-1, ny+nyExt-1, nyExt], 'k--', linewidth=3.0)
ax.text(nx+1.4*nxExt, ny+1.3*nyExt, "E", size=20, ha="center", va="center")
ax.text(0.5*nx+nxExt, 0.5*ny+nyExt, "P", size=20, ha="center", va="center")
ax.text(0.6*nxExt, 0.5*nyExt, "E'", size=20, ha="center", va="center")
plt.savefig(case + '_folded.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + case + '_folded.pdf ' + case + '_folded.pdf')

# Plot single step field
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Single step')
ax.contourf(np.transpose(single), levels=levels, cmap=cmap)
ax.contour(np.transpose(single), levels=levels, colors='k', linewidths=0.5)
ax.plot([0, 0], [0, ny+nyExt-1], 'k--', linewidth=3.0)
ax.plot([0, nx+nxExt-1], [0, 0], 'k--', linewidth=3.0)
ax.plot([0, nx-1, nx-1], [ny-1, ny-1, 0], 'k--', linewidth=3.0)
ax.text(nx+0.4*nxExt, ny+0.3*nyExt, "E", size=20, ha="center", va="center")
ax.text(0.5*nx, 0.5*ny, "P", size=20, ha="center", va="center")
plt.savefig(case + '_single.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + case + '_single.pdf ' + case + '_single.pdf')
