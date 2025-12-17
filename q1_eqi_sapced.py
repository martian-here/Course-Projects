import numpy as np
import matplotlib.pyplot as plt

# Lagrange poynomial L_i
def L(i,a,x,N):
	b = 1.0
	for j in range(N):
		if (j != i):
			b = b*(a-x[j])/(x[i]-x[j])
	return b
	
xx = np.linspace(-1,1,50)
ff_true = np.sin(np.pi*xx/2)

N_values = np.array([1,2,4,8,16,32,64])
l1Error = 0*N_values
l2Error = 0*N_values
lInftyError = 0*N_values

for itr in range(np.size(N_values)):
	N = N_values[itr]
	x = np.linspace(-1,1,N)
	f = np.sin(np.pi*x/2)
	ff = 0*xx
	for i in range(50):
		for j in range(N):
			ff[i] = ff[i] + L(j,xx[i],x,N)*f[j]
	
	l1 = np.sum(np.abs(ff-ff_true))/np.sum(np.abs(ff_true))
	l1Error[itr] = np.log10(l1)
	l2 = np.sqrt(np.sum((ff-ff_true)**2)/np.sum((ff_true)**2))
	l2Error[itr] = np.log10(l2)
	l3 = np.max(np.abs(ff-ff_true))/np.max(np.abs(ff_true))
	lInftyError[itr] = np.log10(l3)
	
plt.plot(xx,ff)
plt.plot(xx,ff_true,'--')
plt.grid()

plt.figure()
plt.plot(N_values, l1Error,'k', label="$L_1$ error")
plt.plot(N_values, l2Error,'--', label="$L_2$ error")
plt.plot(N_values, lInftyError,'-.', label="$L_{\infty}$ error")
plt.xlabel("N")
plt.ylabel("log$_{10}$ Error")
plt.legend()
plt.grid()

plt.show()
