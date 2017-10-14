# Convolutional Network
import csv
import time
import numpy
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib.ticker import LinearLocator, FormatStrFormatter

import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from numpy import linalg as LA
import matlab.engine
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.layers.convolutional import AveragePooling1D
eng = matlab.engine.start_matlab()

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
nox = 61
notrain = 10000
notest = 2000
data = numpy.loadtxt("laplaceTrain10KI01.csv", delimiter=",")
# split into input (X) and output (Y) variables
X_train = data[:,0:notrain]
X_train = numpy.transpose(X_train)

# dimensions: no_train_ex, no_features, 1
X_train = X_train.reshape(notrain, nox, 1).astype('float32')
y_train = data[:,notrain:(2*notrain)]
y_train = numpy.transpose(y_train)


data2 = numpy.loadtxt("laplaceTest2000.csv", delimiter=",")
# split into input (X) and output (Y) variables
X_test = data2[:,0:notest]
X_test = numpy.transpose(X_test)

# dimensions: no_test_ex, no_features, 1
X_test = X_test.reshape(notest, nox, 1).astype('float32')
y_test = data2[:,notest:(notest*2)]
y_test = numpy.transpose(y_test)

def L3(y_true, y_pred):
	diff = K.abs(y_true-y_pred)
	return K.mean(K.pow(diff,3))

def sin(x):
	return K.sin(x)

def cos(x):
	return K.cos(x)

# Get x-values for plotting
X = data[0:nox,(2*notrain)]
#X = X.reshape(1,61)
def baseline_model():
	model = Sequential()
	model.add(Conv1D(250, 5, activation='sigmoid',input_shape=(nox,1)))
	model.add(AveragePooling1D(2))
	model.add(Conv1D(100, 3, activation='sigmoid'))
	model.add(AveragePooling1D(2))
	model.add(Dropout(0.1))
	model.add(Flatten())
	model.add(Dense(nox, activation='linear'))
	model.compile(loss='mean_squared_error', optimizer='adam') #metrics=['mean_squared_error'])
	return model

# build the model
#model = KerasRegressor(build_fn=baseline_model(),verbose=0)

model = baseline_model()

start = time.time()
#stop = 10
#val=[None]*stop
#for q in range(0,stop):

start = time.time()
# Fit the model
model.fit(X_train, y_train, epochs=100, batch_size=20,verbose=0)
end = time.time()
print("Time for training:")
print(end-start)


# Predict values for finer grid
#out = model.predict(X_test)
#print(out.shape)
# Evaluate model
#scores = model.evaluate(X_test, y_test)
#print("Validation loss is:")
#print(scores)
#val[q] = scores

##################### Create Jacobian ###################################
E = numpy.identity(nox)

# Predict unit vectors:
J12 = model.predict(E.reshape(nox, nox, 1).astype('float32'))
J12  = numpy.transpose(J12)
g=9.81
zero = numpy.zeros((nox,nox))

jac1 = numpy.hstack((zero, J12))
jac2 = numpy.hstack((-g*E,zero))
jacobian = numpy.vstack((jac1,jac2))

eigvals = LA.eigvals(jacobian)
#print(eigvals)

fig,ax = plt.subplots()
ax.scatter(eigvals.real, eigvals.imag)
ax.set_xlabel('Real')
ax.set_ylabel('Imaginary')
ax.set_title('Stability test')
plt.show()

##########################################################################


'''
	plt.figure(1)

	plt.subplot(211)
	plt.plot(X,y_test[1,:],'k',X,out[1,:],'b')

	plt.subplot(212)
	plt.plot(X,y_test[10,:],'k',X,out[10,:],'b')
	plt.show()
		
	if q == stop:
		break

end = time.time()
print("Total time is:")
print(end - start)
print("Average time is:")
print((end - start)/stop)
print("Average validation is:")
print(numpy.mean(val))
'''

# Predict
# Initialize
tid=0.0
g = 9.81
init_param = eng.initParam(tid,nargout=9)
E = numpy.float64(init_param[0])
E = E.reshape(nox,1)
P = numpy.float64(init_param[1])
Tmax = numpy.float64(init_param[2])
Twave = numpy.float64(init_param[3])
Np1D = numpy.float64(init_param[4])
Nk = numpy.float64(init_param[5])	
rk4a = numpy.float64(init_param[6])
rk4b = numpy.float64(init_param[7])
rk4c = numpy.float64(init_param[8])
dt = Twave/100
Nsteps = int(round((Tmax*Twave)/dt))
dt = (Tmax*Twave)/Nsteps
resE=E*0
resP=P*0


resultw = []
resultp = []
resulte = []
print(Nsteps)
print(Tmax)
print(dt)
start = time.time()
#loop
for tstep in range(0,Nsteps):
	for INTRK in range(0,5):
		P = P.reshape(1,nox,1)
		E = E.reshape(1,nox,1)
		RKtime = tid + dt*rk4c[0][INTRK]

		resE = rk4a[0][INTRK]*resE
		resP = rk4a[0][INTRK]*resP

		E1 = matlab.double(E.tolist())
		P1 = matlab.double(P.tolist())

		W = model.predict(P)
		#W = numpy.float64(eng.LaplaceSEM2D(E1,P1))
		W = numpy.transpose(W)
		
		rhsE = W
		rhsP = -g*E

		P = P.reshape(nox,1)
		E = E.reshape(nox,1)
		
		resE = resE + dt*rhsE
		resP = resP + dt*rhsP

		E = E + rk4b[0][INTRK]*resE		
		P = P + rk4b[0][INTRK]*resP
		
		#plt.figure(1)

		#plt.plot(X,W,'r',X,P,'b',X,E,'g')
		#plt.show()
	
	resultw.append(W)
	resultp.append(P)
	resulte.append(E)	
	tid = tid + dt
	
end = time.time()
di = end-start
print("Time for predicting:")
print(di)

resw = numpy.asarray(resultw)
resp = numpy.asarray(resultp)
rese = numpy.asarray(resulte)
W = resw.reshape(Nsteps,nox)
P = resp.reshape(Nsteps,nox)
E = rese.reshape(Nsteps,nox)

print(W.shape)
print(E.shape)
print(P.shape)
print(y_test.shape)

Y = numpy.arange(0, (tid), dt)
X, Y = numpy.meshgrid(X,Y)


print(X.shape)
print(Y.shape)

#print(tid)

file1 = open('outdata01.csv','wt')
file2 = open('outdata02.csv','wt')
file3 = open('outdata03.csv','wt')
file4 = open('outdata04.csv','wt')
writer1 = csv.writer(file1)
writer2 = csv.writer(file2)
writer3 = csv.writer(file3)
writer4 = csv.writer(file4)
for irow in range(0,Nsteps):
	writer1.writerow(W[irow,:])
	writer2.writerow(P[irow,:])
	writer3.writerow(E[irow,:])
	writer4.writerow(y_test[irow,:])
file1.close()
file2.close()
file3.close()
file4.close()


fejl = numpy.absolute(W-y_test)
#print(fejl.shape)
frofejl = LA.norm(fejl)
inffejl = fejl.max()
print(inffejl)
print(frofejl)

fig01 = plt.figure(1)
ax01 = fig01.gca(projection='3d')
ax01.set_xlabel('X')
ax01.set_ylabel('Time')
ax01.set_zlabel('Error')
# Plot the surface.

surf01 = ax01.plot_surface(X, Y, fejl, cmap=cm.coolwarm,linewidth=0, antialiased=False)


# Add a color bar which maps values to colors.

fig01.colorbar(surf01, shrink=0.5, aspect=5)


plt.show()





fig02 = plt.figure(2)
ax02 = fig02.gca(projection='3d')


ax02.set_xlabel('X')
ax02.set_ylabel('Time')
ax02.set_zlabel('Vertical velocity - true')
surf02 = ax02.plot_surface(X, Y, y_test, cmap=cm.coolwarm,linewidth=0, antialiased=False)

fig02.colorbar(surf02, shrink=0.5, aspect=5)


plt.show()


fig02 = plt.figure(2)
ax02 = fig02.gca(projection='3d')


ax02.set_xlabel('X')
ax02.set_ylabel('Time')
ax02.set_zlabel('Vertical velocity')
surf02 = ax02.plot_surface(X, Y, W, cmap=cm.coolwarm,linewidth=0, antialiased=False)

fig02.colorbar(surf02, shrink=0.5, aspect=5)


plt.show()


fig3 = plt.figure(3)
ax3 = fig3.gca(projection='3d')


ax3.set_xlabel('X')
ax3.set_ylabel('Time')
ax3.set_zlabel('Eta')
surf3 = ax3.plot_surface(X, Y, E, cmap=cm.coolwarm,linewidth=0, antialiased=False)

fig3.colorbar(surf3, shrink=0.5, aspect=5)


plt.show()

fig4 = plt.figure(3)
ax4 = fig4.gca(projection='3d')


ax4.set_xlabel('X')
ax4.set_ylabel('Time')
ax4.set_zlabel('Potential')
surf4 = ax4.plot_surface(X, Y, P, cmap=cm.coolwarm,linewidth=0, antialiased=False)

fig4.colorbar(surf4, shrink=0.5, aspect=5)


plt.show()