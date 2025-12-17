import numpy as np
import matplotlib.pyplot as plt

# Lagrange poynomial L_i(a)
def L(i,a,x):
	b = 1.0
	for j in range(np.size(x)):
		if (j != i):
			b = b*(a-x[j])/(x[i]-x[j])
	return b
	
def dL(i,a,x):   #Lagrange poynomial derivative (d/dx)L_i(a)
	c = 0.0
	for k in range(np.size(x)):
		b = 1.0
		if(k != i):
			for j in range(np.size(x)):
				if (j != i and j != k):
					b = b*(a-x[j])/(x[i]-x[j])
			c = c + b/(x[i]-x[k])
	return c
					
	
xx = np.linspace(-1,1,50)

df_true = 0.5*np.pi*np.cos(np.pi*xx/2)
df = 0*xx

N_values = np.array([1,2,4,8,16,32,62])
l1Error = 0*N_values
l2Error = 0*N_values
lInftyError = 0*N_values

for itr in range(np.size(N_values)):
	N = N_values[itr]
	x = np.linspace(-1,1,N)
	f = np.sin(np.pi*x/2)
	df = 0*xx
	for i in range(50):
		for j in range(N):
			df[i] = df[i] + dL(j,xx[i],x)*f[j]
		
	l1Error[itr] = np.log10( np.sum(np.abs(df-df_true))/np.sum(np.abs(df_true)) )
	l2Error[itr] = np.log10( np.sqrt(np.sum((df-df_true)**2))/np.sum((df_true)**2) )
	lInftyError[itr] = np.log10( np.max(np.abs(df-df_true))/np.max(np.abs(df_true)) )

plt.figure()	
plt.plot(xx,df_true)	
plt.plot(xx,df,'--')
plt.grid()

plt.figure()
plt.plot(N_values, l1Error, label="$L_1$ error")
plt.plot(N_values, l2Error,'--', label="$L_2$ error")
plt.plot(N_values, lInftyError,'-.', label="$L_{\infty}$ error")
plt.xlabel("N")
plt.ylabel("log$_{10}$ Error")
plt.legend()
plt.grid()


plt.show()
