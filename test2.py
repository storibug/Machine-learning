import numpy as np

#Input array
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

#Output
y=np.array([[1],[1],[0]])

#Sigmoid Function
def tanh (x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_tanh(x):
    return 1-(tanh(x)*tanh(x))

#Variable initialization
epoch=3000 #Setting training iterations
lr=0.1 #Settig learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3#number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer
#weight and bias initialization
print("Epoch= ",epoch,"\nlr= ",lr,"\nhiddenlayer_neurons= ",hiddenlayer_neurons)


wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
print("\nwh=",wh,"\n\nbh=",bh,"\n\nwout",wout,"\n\nbout",bout)
for i in range(epoch):

#Forward Propogation
    hidden_layer_input1=np.dot(X,wh)
    hidden_layer_input=hidden_layer_input1 + bh
    hiddenlayer_activations = tanh(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    output = tanh(output_layer_input)

#Backpropagation
    E = y-output
    slope_output_layer = derivatives_tanh(output)
    slope_hidden_layer = derivatives_tanh(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
print("\n\nOUTPUT:\n")
print (output)
print("\n\nI=")
print(i)
