
# coding: utf-8

# In[1]:


(159.806000, 18.600000)(181.822006, 38.628601)(203.800003, 58.657101)(225.699997, 78.685699)(247.483994, 98.714302)(269.113007, 118.742996)(290.550995, 138.770996)(311.757996, 158.800003)(332.697998, 178.828995)(353.334015, 198.856995)(373.630005, 218.886002)(393.549011, 238.914001)(413.057007, 258.942993)(432.118988, 278.971008)(450.703003, 299.000000)(468.773987, 319.028992)(486.302002, 339.057007)(503.256012, 359.085999)(519.604980, 379.114014)(535.320984, 399.143005)(550.377014, 419.170990)(564.745972, 439.200012)(578.401978, 459.229004)(591.320984, 479.256989)(603.481995, 499.286011)(614.861023, 519.314026)(625.440979, 539.343018)(635.200989, 559.370972)(644.124023, 579.400024)(652.195984, 599.429016)(659.401001, 619.456970)(665.726990, 639.486023)(671.164001, 659.513977)(675.700012, 679.543030)(679.328979, 699.570984)(682.044006, 719.599976)(683.840027, 739.629028)(684.713989, 759.656982)(684.664978, 779.685974)(683.692017, 799.713989)(681.796997, 819.742981)(678.984009, 839.770996)(675.257996, 859.799988)(670.625000, 879.828979)(665.093018, 899.856995)(658.671997, 919.885986)(651.374023, 939.914001)(643.210999, 959.942993)(634.197021, 979.971008)(624.348022, 1000.000000)


# In[21]:


"""
test_ares_database.py
Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Jul  7 09:56:00 PDT 2017
Description: 
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
from scipy import linalg as la
from mpl_toolkits.mplot3d import Axes3D
import sys, csv 
import astropy as ap
from astropy.io import fits

def fromcsv(filename):
    print 'Reading', filename
    d = csv.reader(open(filename,'r'), delimiter=',')
    x = np.array(list(d)[18:-1], dtype=np.float)
    #print d[0]
    return x[:,0]/1e6, x[:,1]   
    
def lin(db,ph): # Converting the magnitude and phase of the measurements into complex returnloss in voltage ratio.
    return 10**(db/20.) * np.exp(2j*np.pi*ph/360.)

#fq,db = fromcsv('/Users/Nipanjana/WORKAREA1/Code_Area/Project_HYPERION/S11_DB.csv') 
#fq,ph = fromcsv('/Users/Nipanjana/WORKAREA1/Code_Area/Project_HYPERION/S11_PH.csv')

#d = lin(db,ph) 

# Open the file
f = h5py.File('/Users/Nipanjana/WORKAREA1/CODE_Area/Project_HYPERION/ares_fcoll_4d.hdf5', 'r')

# Grab the 21-cm signals and the corresponding redshifts
all_signals = f['blobs']['dTb'].value
all_z = f['blobs']['dTb'].attrs.get('ivar')[0]
fqs = 1420. / (1 + all_z)
fqs = fqs[::-1]
# Grab tau and reionization histories to excise garbage models
tau = f['blobs']['tau_e'].value
QHII = f['blobs']['cgm_h_2'].value
Qf = QHII[:,7] # ionized fraction at z=5.7 (should be >= 0.95)

# Models where tau is reasonable
tau_ok = np.logical_and(tau >= 0.04, tau <= 0.09)

# Models where reionization is over when it should be
eor_over = Qf >= 0.95

# Create a filtered set of models. We lose a lot :(
ok = np.logical_and(tau_ok, eor_over)
ok_signals = 0.001*all_signals[ok,:]

# Grab 500 random models and plot them
for i in np.random.randint(0, ok.sum(), size=1500):
    plt.plot(1420. / (1.  + all_z), ok_signals[i], linewidth=2.5, color='b', alpha=0.02)
    

print 'shape of the T21 datacube:', ok_signals.shape

plt.xlim(30,150)
plt.ylim(-0.3,0.3)
#plt.xlabel('Freq (GHz)')
plt.xlabel('Frequency (GHz)')
plt.ylabel('$T_{21}$')
plt.grid()
#plt.legend(loc='lower right')
plt.show()

##########################
#ipython 
#print U.shape, V.shape, s.shape 

# calculate the covariance matrix
R = np.cov(ok_signals, rowvar=False)
U_eor,S_eor,V_eor = np.linalg.svd(R)
plt.figure()
plt.semilogy(S_eor)
plt.grid()
plt.show()


anttemplist1 = ap.io.fits.open("Anttemp-OVRO-230mins-230freqs-50to100MHz.fits") 
anttemplist2 = ap.io.fits.open("Anttemp-OVRO-230mins-230freqs-101to150MHz.fits") 
anttempdata1 = anttemplist1[0].data
anttempdata2 = anttemplist2[0].data
anttemplist1.close
anttemplist2.close

fg_data= np.hstack((anttempdata1,anttempdata2))

print 'Shape of the fits  datacube:', anttempdata1.shape, anttempdata1.shape, fgdata.shape


#fg_poly = np.polyfit(np.log10(fqs), np.log10(fg_data[0]), deg=5)
#fg_mdl = 10**np.polyval(fg_poly, np.log10(fqs))
#fg_resid = fg_data[0]-fg_mdl
##plt.plot(fqs, fg_data[0], '.')
#plt.plot(fqs, fg_mdl, 'r')
#plt.show()

plt.plot(fg_data[0], '.')
plt.show()

C = np.cov(fg_data, rowvar=False)
print C.shape
#U,S,V = np.linalg.svd(C)
#plt.figure()
#plt.semilogy(S)
#plt.grid()
#plt.show()











# In[22]:


C = np.cov(fg_data, rowvar=False)
print C.shape
U,S,V = np.linalg.svd(C)
plt.figure()
plt.semilogy(S)
plt.xlim(0,10)
plt.grid()
plt.show()




# In[19]:


print fg_data.shape


# In[29]:


eor_vals = np.dot(U,ok_signals.T)
print eor_vals.shape


# In[108]:


plt.figure()
plt.semilogy(S)
plt.semilogy(np.abs(eor_vals[...,0]))
plt.xlim(0,20)
plt.ylim(1e-4,1e2)
plt.grid()
plt.show()


# In[37]:


fq,db = fromcsv('/Users/Nipanjana/WORKAREA1/Code_Area/Project_HYPERION/S11_DB.csv') 
fq,ph = fromcsv('/Users/Nipanjana/WORKAREA1/Code_Area/Project_HYPERION/S11_PH.csv')

d = lin(db,ph)


# In[44]:


total_signal = fg_data[0]+ok_signals[0]
plt.plot(fg_data[0])
plt.plot(ok_signals[0])
print ok_signals.shape
print fg_data.shape
plt.plot(total_signal)
plt.show()


# In[55]:


total_signal_copy = total_signal.copy()
total_signal_copy.shape = (-1,1)
total_vals = np.dot(V,total_signal_copy)


# In[87]:


filter_total_vals = total_vals.copy()
print filter_total_vals.shape

filter_total_vals[0:3,0] = 0
filter_total_signal = np.dot(filter_total_vals.T,V)


# In[104]:


eor_copy = ok_signals[20].copy()
eor_copy.shape = (-1,1)
eor_vals = np.dot(V,eor_copy)
filter_eor_vals = eor_vals.copy()
print eor_vals.shape
filter_eor_vals[0:3,0] = 0
filter_eor_signal = np.dot(filter_eor_vals.T,V)


# In[105]:


plt.plot(filter_total_signal.flatten())
#plt.plot(fg_data[0])
plt.plot(ok_signals[0])
plt.plot(filter_eor_signal.flatten(),'.')
#plt.plot(total_signal)
plt.show()


# # This is a writeup of what we learned
# 
# We learned that the instrument modifies the eigenbasis $\hat e$ of foregrounds, and they better not project onto $\hat e_{\rm sig}$.
