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
    
def interpolation(fqs,fq,d):
	d_fqs = np.interp(fqs, fq, d)
	return d_fqs
	


#=============================================================================================#
# Open the file containing ARES models and read the 21cm models:
#=============================================================================================#
f = h5py.File('/Users/Nipanjana/WORKAREA1/CODE_Area/Project_HYPERION/ares_fcoll_4d.hdf5', 'r')

# Grab the 21-cm signals and the corresponding redshifts
all_signals = f['blobs']['dTb'].value
all_z = f['blobs']['dTb'].attrs.get('ivar')[0]
fqs = 1420. / (1 + all_z)
fqs = fqs[::-1]
#for i in range(0, 459):
#				if fqs[i]>150:
#					print i, fqs[i]
#valid = np.where(np.logical_and(fqs < 150, fqs > 50))


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

ok_signals_copy=np.zeros((8211, 190))
fqs_copy = np.zeros(190)
for i in range(0, 8210):
			for j in range(0, 190):
				ok_signals_copy[i][j] = ok_signals[i][235+j]
				fqs_copy[j] = fqs[235+j] 
			

print 'ok_signals_copy.shape', ok_signals_copy.shape
print 'fqs_copy.shape', fqs_copy.shape



# Grab 500 random models and plot them
#for i in np.random.randint(0, ok.sum(), size=1500):
plt.xlim(50,150)
plt.ylim(-0.3,0.3)
#plt.xlabel('Freq (GHz)')
plt.xlabel('Frequency (GHz)')
plt.ylabel('$T_{21}(K)$')
plt.grid()
for i in range(0, 20):
	    plt.plot(1420. / (1.  + all_z), ok_signals[i], '.', alpha=0.5, label = "Model")

#plt.legend(loc='lower right')
plt.show()

#Calculate the covariance matrix for EoR models
#R = np.cov(ok_signals, rowvar=False)
#U_eor,S_eor,V_eor = np.linalg.svd(R)
'''
plt.figure()
plt.semilogy(S_eor, '.')
plt.grid()
plt.show()
'''

#=============================================================================================#
# Open the file containing foreground models and read power spectrum:
#=============================================================================================#
'''
anttemplist1 = ap.io.fits.open("Anttemp-OVRO-230mins-230freqs-50to100MHz.fits") 
anttemplist2 = ap.io.fits.open("Anttemp-OVRO-230mins-230freqs-101to150MHz.fits") 
anttempdata1 = anttemplist1[0].data
anttempdata2 = anttemplist2[0].data
anttemplist1.close
anttemplist2.close

fg_data= np.hstack((anttempdata1,anttempdata2))
fg_data_copy=np.zeros((230, 190))
for i in range(0, 230):
			for j in range(0, 189):
				fg_data_copy[i][j] = fg_data[i][235+j]
			

print 'Shape of the foreground  datacube:', anttempdata1.shape, anttempdata1.shape, fg_data.shape, fg_data_copy.shape
plt.plot(fg_data_copy[0], '.')
plt.show()
'''

anttemplist1 = ap.io.fits.open("Anttemp-OVRO-1mins_4hours.fits") 
anttempdata1 = anttemplist1[0].data
anttemplist1.close
fg_data= np.array(anttempdata1)
print 'fg_data.shape=',fg_data.shape

#fg_data = fg_data[valid]
#=============================================================================================#
# Decompose the foreground data into principle components and project EoR dataset 
#onto the foreground principle components
#=============================================================================================#
C = np.cov(fg_data, rowvar=False)
U,S,V = np.linalg.svd(C)

print 'C.Shape = ', C.shape 
print 'U.Shape = ', U.shape
print 'S.Shape = ', U.shape
print 'V.Shape = ', U.shape

fg_vals = np.dot(fg_data,V.T)
eor_vals = np.dot(ok_signals_copy, V.T) #Projecting the eor signals onto the foreground eigen modes. 

plt.figure()
plt.xlabel("Eigen Values")
plt.ylabel("Amplitude")
#plt.xlim(0,10)
plt.legend(loc='upper right')
plt.grid()
#plt.semilogy(S, 'o', color = 'blue', linewidth = 2.5, label = '')
#plt.semilogy(S, linewidth = 2.5, color = 'black', label = 'Foreground Eigen mode')

#for i in np.random.randint(0, 400, size=1):
for i in range(0, 500):
		if i ==0:
			plt.semilogy(np.abs(fg_vals[i]),'o',color = 'red', label = 'EoR models projected on to the Foreground Eigen modes')
			plt.semilogy(np.abs(eor_vals[i]),'o',color = 'red', label = 'EoR models projected on to the Foreground Eigen modes')
		else:
			plt.semilogy(np.abs(eor_vals[i]),'o')
		#plt.pause(0.5)	
plt.show()

#=============================================================================================#
# Take total data and filter out the most dominant Eigen modes.
#=============================================================================================#

total_signal =np.zeros((50,len(fg_data[0])))
total_signal_copy =np.zeros((50,len(fg_data[0])))

#Take 20 EoR model and add them onto the same foreground mode. 
#Remove the first three Eigen mode and look at the distorion in the residual data that contains only the models.

for i in range(0, 50):
	#for j in range(0, 460):
	total_signal[i] = np.add(fg_data[0],ok_signals_copy[i])
#	plt.plot(total_signal[i])
	
#plt.show()

print total_signal.shape
print fg_data[0].shape
print ok_signals[0].shape

for i in range(0, 50):
		total_signal_copy = total_signal[i].copy()
		total_signal_copy.shape = (-1,1)
		total_vals = np.dot(V,total_signal_copy)
		filter_total_vals = total_vals.copy()
		#print filter_total_vals.shape
		filter_total_vals[0:3,0] = 0
		filter_total_signal = np.dot(filter_total_vals.T,V)
		#plt.plot(filter_total_signal.flatten(), label = 'Residuals after filtering the Foreground Eigen mode')
		#plt.plot(ok_signals[i], label = 'EoR model')
		#plt.pause(0.01)
		#plt.cla()

#plt.show()

#=============================================================================================#
# Take total data, multiply by instrument response and filter out the most dominant Eigen modes.
#=============================================================================================#

beam = np.loadtxt('beam.txt')
beam_avg  = (beam.T).sum(axis = 1)
print 'beam_avg.shape', beam_avg.shape

fq,db = fromcsv('/Users/Nipanjana/WORKAREA1/Code_Area/Project_HYPERION/S11_DB.csv') 
fq,ph = fromcsv('/Users/Nipanjana/WORKAREA1/Code_Area/Project_HYPERION/S11_PH.csv')
d = lin(db,ph)  # Returns linear and complex voltage ratio 
db_fqs = np.interp(fqs_copy, fq, db) # Interpolate instrument data at frequencies where ARES models available
ph_fqs = np.interp(fqs_copy, fq, ph)# Interpolate instrument data at frequencies  where ARES models available
d_fqs = lin(db_fqs, ph_fqs)
inst = (1-np.abs(d_fqs)**2.0)
print d.shape, d_fqs.shape, inst.shape

#for j in range(0, 190):
#			print fqs_copy[j], db_fqs[j]
#			inst[j] = 5.0
			

plt.plot(fqs_copy,db_fqs, 'o')
#plt.plot(fqs,20*np.log10(np.abs(d_fqs)), '.', label= 'interpolated')
#plt.plot(fq,20*np.log10(np.abs(d)), '.', label = 'measured')
#plt.plot(fqs,inst)
#plt.xlim(50,150)
plt.show()


fg_data_inst = np.multiply(fg_data,inst)
fg_data_inst_vals = np.dot(fg_data_inst, V.T)
ok_signal_copy_inst = np.multiply(ok_signals_copy,inst)
eor_inst_vals =np.dot(ok_signals_copy, V.T)

#plt.plot(fqs,total_signal[0], '.', label ='Total Signal' )
#plt.plot(fqs, total_signal_inst[0], '.', label = 'Total Signal * Instrument')
#plt.xlim(60,150)
#plt.show()


C_inst = np.cov(fg_data_inst, rowvar=False)
U_inst,S_inst,V_inst = np.linalg.svd(C_inst)

#print 'C_inst.Shape = ', C.shape 
#print 'U_inst.Shape = ', U.shape
#print 'S_inst.Shape = ', U.shape
#print 'V_inst.Shape = ', U.shape

plt.figure()
#plt.semilogy(S, 'o', color = 'blue', linewidth = 2.5, label = '')
#plt.semilogy(S, linewidth = 2.5, color = 'black', label = 'Foreground Eigen mode')
#plt.semilogy(S_inst, 'o', color = 'yellow', linewidth = 2.5, label = 'Foreground*Instrument')
plt.semilogy(np.abs(fg_vals[i]),'o',color = 'red', label = 'Foreground Principle Components')
#plt.semilogy(np.abs(fg_data_inst_vals[i]),'o',color = 'brown', label = 'Forground*Instrument response on Foreground Principle Components')
for i in range(0,5):
		if i==0:
			plt.semilogy(np.abs(eor_vals[i]),'o', label = 'EoR models projected on the Foreground Principle Components')
			plt.semilogy(np.abs(eor_inst_vals[i]),'o', label = 'EoR*Instrument models projected on the Foreground Principle Components')
#			plt.semilogy(np.abs(fg_data_inst_vals[i]),'o',color = 'blue', label = 'Inctrument response on to the Foreground Eigen modes')
		else:
			plt.semilogy(np.abs(eor_vals[i]),'o')
			plt.semilogy(np.abs(eor_inst_vals[i]),'o')
#			plt.semilogy(np.abs(fg_data_inst_vals[i]),'o')
		#plt.pause(0.5)	
plt.xlabel("Eigen Values")
plt.ylabel("Amplitude")
#plt.xlim(0,100)
plt.legend(loc='upper right')
plt.grid()
plt.show()

#for i in range(0, 50):
#		total_signal_inst_copy = total_signal_inst[i].copy()
#		total_signal_inst_copy.shape = (-1,1)
#		total_inst_vals = np.dot(V,total_signal_inst_copy)
#		filter_total_inst_vals = total__inst_vals.copy()
#		print filter_total_inst_vals.shape
#		filter_total_inst_vals[0:3,0] = 0
#		filter_total_inst_signal = np.dot(filter_total_vals.T,V)
		
		#plt.plot(filter_total_signal.flatten(), label = 'Residuals after filtering the Foreground Eigen mode')
		#plt.plot(ok_signals[i], label = 'EoR model')
		#plt.pause(0.01)
		#plt.cla()

#plt.show()






#for i in range(0, 460):
#		print fqs[i],total_signal[0][i], inst[i], total_signal_inst[i]
		#print fq[i], fqs[i]

#fg_poly = np.polyfit(np.log10(fqs), np.log10(fg_data[0]), deg=5)
#fg_mdl = 10**np.polyval(fg_poly, np.log10(fqs))
#fg_resid = fg_data[0]-fg_mdl
##plt.plot(fqs, fg_data[0], '.')
#plt.plot(fqs, fg_mdl, 'r')
#plt.show()
#plt.plot(fg_data[0], '.')
#plt.show()

'''
print eor_vals.shape
label = ['%d' % m for m in range(10)]
plt.figure()

plt.semilogy(S)
#for i in np.random.randint(0, 400, size=10):
 #			plt.semilogy(np.abs(eor_vals[...,i]))
plt.semilogy(np.abs(eor_vals[...,20]))
#plt.semilogy(np.abs(eor_vals[...,2]))
#plt.semilogy(np.abs(eor_vals[...,3]))
#plt.semilogy(np.abs(eor_vals[...,5]))
#plt.semilogy(np.abs(eor_vals), alpha=0.01)
plt.xlim(0,20)
#plt.ylim(1e-4,1e2)
plt.grid()
plt.show()
'''