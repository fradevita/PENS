import numpy as np
import matplotlib.pyplot as plt



N = 100

fig, ax = plt.subplots()

############################################################################
# Physcial domain
############################################################################
ax.set_xlim([-1.5,6.5])
ax.set_ylim([-1.5,6.5])
# Left line
ax.plot(0.*np.ones(N), np.linspace(0.0,5.0,N), '-k', linewidth = 3)
# Right line
ax.plot(5.*np.ones(N), np.linspace(0.0,5.0,N), '-k', linewidth = 3)
# Bottom line
ax.plot(np.linspace(0.0,5.0,N), 0.*np.ones(N), '-k', linewidth = 3)
# Top line
ax.plot(np.linspace(0.0,5.0,N), 5.*np.ones(N), '-k', linewidth = 3)

############################################################################
# Central cell
############################################################################
# Left line
ax.plot(2.*np.ones(N), np.linspace(1.8,3.2,N), '-k')
# Right line
ax.plot(3.*np.ones(N), np.linspace(1.8,3.2,N), '-k')
# Bottom line
ax.plot(np.linspace(1.8,3.2,N), 2.*np.ones(N), '-k')
# Top line
ax.plot(np.linspace(1.8,3.2,N), 3.*np.ones(N), '-k')
# Cell center point
ax.plot(2.5, 2.5, 'ob', ms = 10)
ax.text(2.42, 2.625, r'$p$', fontsize = 14, color = 'blue')
ax.text(2.25, 2.25, r'$(i,j)$', fontsize = 14, color = 'black')
# x-Face point
ax.plot(3.0, 2.5, 'sr', ms = 10)
ax.text(3.15, 2.45, r'$u$', fontsize = 14, color = 'red')
# y-Face point
ax.plot(2.5, 3.0, '^g', ms = 10)
ax.text(2.45, 3.15, r'$v$', fontsize = 14, color = 'green')

############################################################################
# Lower left cell
############################################################################
# Left line
ax.plot(0.*np.ones(N), np.linspace(0.0,1.2,N), '-k')
# Right line
ax.plot(1.*np.ones(N), np.linspace(0.0,1.2,N), '-k')
# Bottom line
ax.plot(np.linspace(0.0,1.2,N), 0.*np.ones(N), '-k')
# Top line
ax.plot(np.linspace(0.0,1.2,N), 1.*np.ones(N), '-k')

# ghost cell lines
ax.plot(-1.*np.ones(N), np.linspace(0.0,1.0,N), '--k')
ax.plot(np.linspace(-1.0,0.0,N), 0.*np.ones(N), '--k')
ax.plot(np.linspace(-1.0,0.0,N), 1.*np.ones(N), '--k')
ax.plot(0.*np.ones(N), np.linspace(-1.0,0.0,N), '--k')
ax.plot(1.*np.ones(N), np.linspace(-1.0,0.0,N), '--k')
ax.plot(np.linspace(0.0,1.0,N), -1.*np.ones(N), '--k')

# Cell center point
ax.plot(0.5, 0.5, 'ob', ms = 10)
ax.text(0.25, 0.25, r'$(1,1)$', fontsize = 14, color = 'black')
# Cell center point ghost node
ax.plot(-0.5,  0.5, 'ob', ms = 10, fillstyle = 'none')
ax.plot( 0.5, -0.5, 'ob', ms = 10, fillstyle = 'none')
# x-Face point
ax.plot(1.0, 0.5, 'sr', ms = 10)
# x-Face point ghost node
ax.plot(0.0, 0.5, 'sr', ms = 10, fillstyle = 'none')
ax.plot(1.0, -0.5, 'sr', ms = 10, fillstyle = 'none')
# y-Face point
ax.plot(0.5, 1.0, '^g', ms = 10)
# y-Face point ghost node
ax.plot(0.5, 0.0, '^g', ms = 10, fillstyle = 'none')
ax.plot(-0.5, 1.0, '^g', ms = 10, fillstyle = 'none')

############################################################################
# Lower right cell
############################################################################
# Left line
ax.plot(4.*np.ones(N), np.linspace(0.0,1.2,N), '-k')
# Right line
ax.plot(5.*np.ones(N), np.linspace(0.0,1.2,N), '-k')
# Bottom line
ax.plot(np.linspace(3.8,5.0,N), 0.*np.ones(N), '-k')
# Top line
ax.plot(np.linspace(3.8,5.0,N), 1.*np.ones(N), '-k')

# ghost cell lines
ax.plot(6.*np.ones(N), np.linspace(0.0,1.0,N), '--k')
ax.plot(np.linspace(5.0,6.0,N), 0.*np.ones(N), '--k')
ax.plot(np.linspace(5.0,6.0,N), 1.*np.ones(N), '--k')
ax.plot(4.*np.ones(N), np.linspace(-1.0,0.0,N), '--k')
ax.plot(5.*np.ones(N), np.linspace(-1.0,0.0,N), '--k')
ax.plot(np.linspace(4.0,5.0,N), -1.*np.ones(N), '--k')

# Cell center point
ax.plot(4.5, 0.5, 'ob', ms = 10)
ax.text(4.2, 0.25, r'$(n_x,1)$', fontsize = 14, color = 'black')
# Cell center point ghost node
ax.plot(5.5, 0.5, 'ob', ms = 10, fillstyle = 'none')
ax.plot(4.5, -0.5, 'ob', ms = 10, fillstyle = 'none')
# x-Face point
ax.plot(5.0, 0.5, 'sr', ms = 10)
# x-Face point ghost node
ax.plot(6.0, 0.5, 'sr', ms = 10, fillstyle = 'none')
ax.plot(5.0, -0.5, 'sr', ms = 10, fillstyle = 'none')
# y-Face point
ax.plot(4.5, 1.0, '^g', ms = 10)
# y-Face point ghost node
ax.plot(4.5, 0.0, '^g', ms = 10, fillstyle = 'none')
ax.plot(5.5, 1.0, '^g', ms = 10, fillstyle = 'none')

############################################################################
# Upper left cell
############################################################################
# Left line
ax.plot(0.*np.ones(N), np.linspace(3.8,5.0,N), '-k')
# Right line
ax.plot(1.*np.ones(N), np.linspace(3.8,5.0,N), '-k')
# Bottom line
ax.plot(np.linspace(0.0,1.2,N), 4.*np.ones(N), '-k')
# Top line
ax.plot(np.linspace(0.0,1.2,N), 5.*np.ones(N), '-k')

# ghost cell lines
ax.plot(-1.*np.ones(N), np.linspace(4.0,5.0,N), '--k')
ax.plot(np.linspace(-1.0,0.0,N), 4.*np.ones(N), '--k')
ax.plot(np.linspace(-1.0,0.0,N), 5.*np.ones(N), '--k')
ax.plot(0.*np.ones(N), np.linspace(5.0,6.0,N), '--k')
ax.plot(1.*np.ones(N), np.linspace(5.0,6.0,N), '--k')
ax.plot(np.linspace(0.0,1.0,N), 6.*np.ones(N), '--k')

# Cell center point
ax.plot(0.5, 4.5, 'ob', ms = 10)
ax.text(0.2, 4.25, r'$(1,n_y)$', fontsize = 14, color = 'black')
# Cell center point ghost node
ax.plot(-0.5, 4.5, 'ob', ms = 10, fillstyle = 'none')
ax.plot(0.5, 5.5, 'ob', ms = 10, fillstyle = 'none')
# x-Face point
ax.plot(1.0, 4.5, 'sr', ms = 10)
# x-Face point ghost node
ax.plot(0.0, 4.5, 'sr', ms = 10, fillstyle = 'none')
ax.plot(1.0, 5.5, 'sr', ms = 10, fillstyle = 'none')
# y-Face point
ax.plot(0.5, 5.0, '^g', ms = 10)
# y-Face point ghost node
ax.plot(-0.5, 5.0, '^g', ms = 10, fillstyle = 'none')
ax.plot(0.5, 6.0, '^g', ms = 10, fillstyle = 'none')

############################################################################
# Upper right cell
############################################################################
# Left line
ax.plot(4.*np.ones(N), np.linspace(3.8,5.0,N), '-k')
# Right line
ax.plot(5.*np.ones(N), np.linspace(3.8,5.0,N), '-k')
# Bottom line
ax.plot(np.linspace(3.8,5.0,N), 4.*np.ones(N), '-k')
# Top line
ax.plot(np.linspace(3.8,5.0,N), 5.*np.ones(N), '-k')

# ghost cell lines
ax.plot(6.*np.ones(N), np.linspace(4.0,5.0,N), '--k')
ax.plot(np.linspace(5.0,6.0,N), 4.*np.ones(N), '--k')
ax.plot(np.linspace(5.0,6.0,N), 5.*np.ones(N), '--k')
ax.plot(4.*np.ones(N), np.linspace(5.0,6.0,N), '--k')
ax.plot(5.*np.ones(N), np.linspace(5.0,6.0,N), '--k')
ax.plot(np.linspace(4.0,5.0,N), 6.*np.ones(N), '--k')

# Cell center point
ax.plot(4.5, 4.5, 'ob', ms = 10)
ax.text(4.2, 4.25, r'$(n_x,n_y)$', fontsize = 14, color = 'black')
# Cell center point ghost node
ax.plot(5.5, 4.5, 'ob', ms = 10, fillstyle = 'none')
ax.plot(4.5, 5.5, 'ob', ms = 10, fillstyle = 'none')
# x-Face point
ax.plot(5.0, 4.5, 'sr', ms = 10)
# x-Face point ghost node
ax.plot(6.0, 4.5, 'sr', ms = 10, fillstyle = 'none')
ax.plot(5.0, 5.5, 'sr', ms = 10, fillstyle = 'none')
# y-Face point
ax.plot(4.5, 5.0, '^g', ms = 10)
# y-Face point ghost node
ax.plot(4.5, 6.0, '^g', ms = 10, fillstyle = 'none')
ax.plot(5.5, 5.0, '^g', ms = 10, fillstyle = 'none')

############################################################################
# Cosmetics
############################################################################
# Set plot to square size
ax.set_aspect('equal')

# Remove all ticks 
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
# Show plot
fig.canvas.manager.full_screen_toggle() # toggle fullscreen mode
plt.tight_layout()
plt.axis('off')
plt.show()
plt.close()