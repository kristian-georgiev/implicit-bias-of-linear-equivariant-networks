import numpy as np
import math
import cvxpy as cp

pi = math.pi

def get_dn_irreps(n=8):
	out = np.zeros((n,n)).astype(np.complex128)
	out[:2,:] = 1.
	out[1,n//2:] = -1.
	rots = np.arange(n//2)

	if n%4 ==0:
		a = np.arange(0,n,2)
		out[2,a] = 1
		out[2,a+1] = -1
		out[3,:] = 1
		out[3,a[:n//4]+1] = -1
		out[3,a[n//4:]] = -1
		left = (n-4)//4
		curr = 4
	else:
		left = (n-2)//4
		curr = 2
	for rho in range(left):
		out[rho*4+curr,:n//2] = np.exp(rots*pi*2j/(n/2)) 
		out[rho*4+curr+3,:n//2] = np.exp(-rots*pi*2j/(n/2)) 
		out[rho*4+curr+1,n//2:] = np.exp(rots*pi*2j/(n/2)) 
		out[rho*4+curr+2,n//2:] = np.exp(-rots*pi*2j/(n/2)) 
	#out[4:,:] *= np.sqrt(2)
	#out = out/np.sqrt(n)
	return out

def perform_Fm(n,x):
	F = get_dn_irreps(n)
	print(F, type(F))
	Fx = F@x
	out = []
	out.append(Fx[0,:].reshape(-1,1,1))
	out.append(Fx[1,:].reshape(-1,1,1))
	if n%4 ==0:
		out.append(Fx[2,:].reshape(-1,1,1))
		out.append(Fx[3,:].reshape(-1,1,1))
		left = (n-4)//4
		curr = 4
	else:
		left = (n-2)//4
		curr = 2
	for rho in range(left):
		out.append(Fx[curr+rho*4:curr + rho*4 +4,:].T.reshape(-1,2,2))
	return out

import pprint
def opt(n,data):
	F = get_dn_irreps(n)
	F = F[:, [0, 2, 1, 3, 4, 6, 5, 7]] #so it's in line with our fourier basis matrix...

	x = cp.Variable(n,complex = True)
	A = F@x

	params = []
	multipliers = []
	# params.append(cp.Parameter((1,1), complex = True))
	params.append(cp.reshape(A[0],(1,1)))
	multipliers.append(1.)
	# params.append(cp.Parameter((1,1), complex = True))
	params.append(cp.reshape(A[1],(1,1)))
	multipliers.append(1.)
	if n%4 == 0:
		# params.append(cp.Parameter((1,1),complex = True))
		# params[-1][0,0].value = A[2]
		# params.append(cp.Parameter((1,1),complex = True))
		# params[-1][0,0].value = A[3]
		params.append(cp.reshape(A[2],(1,1)))
		params.append(cp.reshape(A[3],(1,1)))
		multipliers.append(1.)
		multipliers.append(1.)
		left = (n-4)//4
		curr = 4
	else:
		left = (n-2)//4
		curr = 2

	for i in range(left):
		params.append(cp.reshape(A[curr+i*4:curr+i*4+4],(2,2)))
		multipliers.append(2.)

	value = 0.
	for m,v in zip(multipliers,params):
		value += cp.norm(v,'nuc')*m

	constraints = [cp.real(data@x) >= 1]
	prob = cp.Problem(cp.Minimize(value), constraints)
	prob.solve(solver=cp.SCS)
	return x.value



if __name__ == '__main__': 
	A = np.round(get_dn_irreps(8))
	B = perform_Fm(8,A[-6:,:].T.conj())
	print(B)
	X = np.random.randn(3,8).astype(np.complex128)

	print()
	for i in X:
		print(i)
	print(type(X))
	v = opt(8,X)
	print(v)