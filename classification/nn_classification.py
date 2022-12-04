#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
np.random.seed(42)
NUM_FEATS = 90


# In[3]:


def one_hot(target):
    m = target.shape[0]
    result = np.zeros((m, 4))    
    for idx, value in enumerate(target):
        if value == "Very Old":
            result[idx] = np.array([1, 0, 0, 0])
        elif value == "Old":
            result[idx] = np.array([0, 1, 0, 0])
        elif value == "Recent":
            result[idx] = np.array([0, 0, 1, 0])
        elif value == "New":
            result[idx] = np.array([0, 0, 0, 1])
#     print(result.shape)
    return result



def read_data():
    dev = pd.read_csv('dev.csv')
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    train_target = np.array(train.iloc[:,0])
    train_target = one_hot(train_target)
    dev_target = np.array(dev.iloc[:,0])
    dev_target = one_hot(dev_target)

    train_input = np.array(train.iloc[:, 1:])
    dev_input = np.array(dev.iloc[:, 1:])
    test_input = np.array(test)
            
    train_input = (train_input - train_input.mean(axis = 0)) / train_input.std(axis = 0)
    dev_input = (dev_input - dev_input.mean(axis = 0)) / dev_input.std(axis = 0)
    test_input = (test_input - test_input.mean(axis = 0)) / test_input.std(axis = 0)

    return train_input, train_target, dev_input, dev_target, test_input


# In[4]:


def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return np.where(z > 0, 1.0, 0.0)

def softmax(z):
    z = z.T
    z = np.exp(z - np.max(z, axis = 0)) / np.sum(np.exp(z - np.max(z, axis = 0)), axis = 0)
    return z.T
def softmax_prime(z):    
    return np.diagflat(z) - np.dot(z, z.T)


# In[5]:


def loss_mse(y, y_hat):
    return np.sum((y-y_hat)**2)/(2*y.shape[0])

def loss_regularization(weights, biases):
    return np.sum([np.linalg.norm(w) for w in weights])

def loss_fn(y, y_hat, weights, biases, lamda):
    return loss_mse(y, y_hat) + lamda * loss_regularization(weights, biases)

def rmse(y, y_hat):
    return (np.sum((y-y_hat)**2)/(2*y.shape[0]))**0.5

def cross_entropy_loss(y, y_hat):
    eps = np.finfo(float).eps
    return -(1/y.shape[0]) * np.sum(np.sum(y*np.log(y_hat+eps), axis = 1))

def loss_fn_classification(y, y_hat, weights, biases, lamda):
    return cross_entropy_loss(y, y_hat) + lamda * loss_regularization(weights, biases)


# In[6]:


class AdamOptimizer:
    def __init__(self, weights, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.alpha = alpha
        self.m = [np.zeros(w.shape) for w in weights]
        self.v = [np.zeros(w.shape) for w in weights]

    def step(self, weights, gradients):
        self.t = self.t + 1
        for idx, gradient in enumerate(gradients):            
            self.m[idx] = self.beta1*self.m[idx] + (1 - self.beta1)*gradient
            self.v[idx] = self.beta2*self.v[idx] + (1 - self.beta2)*(gradient**2)
            m_hat = self.m[idx]/(1 - self.beta1**self.t)
            v_hat = self.v[idx]/(1 - self.beta2**self.t)
            weights[idx] = weights[idx] - self.alpha*(m_hat/(np.sqrt(v_hat) - self.epsilon))
        return weights



class Optimizer(object):

    def __init__(self, learning_rate, batch_size, weights, biases):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.adam_weight = AdamOptimizer(weights, learning_rate)
        self.adam_bias = AdamOptimizer(biases, learning_rate)
        self.drop_out = 0.3
#         self.drop_out_list = linear_drop_out_distribution()
        
    def adam_step(self,weights, biases, delta_weights, delta_biases):
        new_weights = self.adam_weight.step(weights, delta_weights)
        new_biases = self.adam_bias.step(biases, delta_biases)
        return new_weights, new_biases
    
#     def linear_drop_out_distribution(self):
#         prob_dis = np.linspace(0, self.drop_out, self.max_epoch//10)
#         prob_dis = np.repeat(drop_out_list, 10)
#         return prob_dis
    
#     def drop_out_step(self, weights, biases, delta_weights, delta_biases, epoch_iteration, batch_iteration):
#         drop_out = np.random.binomial(1, 1 - self.drop_out_list[epoch_iteration])
#         lr = self.learning_rate
#         new_weights = [weight -  drop_out* lr * delta_of_weight for weight, delta_of_weight in zip(weights, delta_weights)]
#         new_biases = [bias - drop_out * lr * delta_of_bias for bias, delta_of_bias in zip(biases, delta_biases)]
#         return new_weights, new_biases

    def step(self, weights, biases, delta_weights, delta_biases):
        new_weights = [weight - self.learning_rate * delta_of_weight for weight, delta_of_weight in zip(weights, delta_weights)]
        new_biases = [bias - self.learning_rate * delta_of_bias for bias, delta_of_bias in zip(biases, delta_biases)]
        return new_weights, new_biases


# In[7]:


class Net(object):
    
    
    def __init__(self, num_layers, num_units):
        self.num_layers = num_layers
        self.num_units = num_units

        self.biases = []
        self.weights = []
        self.h_states = []
        self.a_states = []
        for i in range(num_layers):
            if i==0:
                # Input layer
                self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
            else:
                # Hidden layer
                self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))

            self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

        # Output layer
        self.biases.append(np.random.uniform(-1, 1, size=(4, 1)))
        self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 4)))



    def __call__(self, X):
        return self.forward(X)
    
    def derivative_of_activation(self, x):
        #using relu for now
        return relu_prime(x)
    
    def forward(self, X):
        a = X
        last_layer = len(self.weights) - 1
        self.h_states = []
        self.a_states = [] 
        for layer, (w, b) in enumerate(zip(self.weights, self.biases)):
            if layer==0:
                self.h_states.append(a) # For input layer, both h and a are same
            else:
                self.h_states.append(h)    
            
            self.a_states.append(a)
            h = np.dot(a, w) + b.T
            
#             h = a @ w + b.T
            
            if layer < last_layer:
                a = relu(h)
            else: # No activation for the output layer
                a = softmax(h)
#             print("net", a)
        return a

    def backward(self, X, y, lamda):
        #total batch size
        batch_size = np.shape(y)[0]
        
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_W = [np.zeros(w.shape) for w in self.weights]
        

        last_layer = len(self.weights) - 1 
        pred = self.forward(X);        
        
        #finding the final layer loss gradient wrt to weight
        loss_gradient_weight = np.dot(self.a_states[-1].T, pred - y)
        loss_gradient_bias = np.dot((pred-y).T, np.ones([batch_size, 1]))
        
        dl_da = np.dot(self.weights[-1], (pred - y).T)
        
        del_W[-1] = (1./batch_size) * (loss_gradient_weight + lamda * self.weights[-1])
        del_b[-1] = (1./batch_size) * loss_gradient_bias
        
        
        
        for layer in range(last_layer-1, -1, -1):
            da_dz = self.derivative_of_activation(self.h_states[layer+1].T)
            dl_dz = np.multiply(dl_da, da_dz)
            
            dl_dw = np.dot(self.a_states[layer].T, dl_dz.T)
            dl_db = np.dot(dl_dz, np.ones([batch_size, 1]))
            
            del_W[layer] = (1./batch_size) * (dl_dw + lamda * self.weights[layer])
            del_b[layer] = (1./batch_size) * dl_db
        
            dl_da = np.dot(self.weights[layer], dl_dz)
        
        return del_W, del_b


# In[8]:


def train(net, optimizer, lamda, batch_size, max_epochs,
    train_input, train_target,
    dev_input, dev_target):
    
    m = train_input.shape[0]
    epoch_losses = []
    dev_losses = []
    for e in range(max_epochs):
        epoch_loss = 0
#         if e%100 == 0:
#             optimizer = Optimizer(optimizer.learning_rate, batch_size, net.weights, net.biases)
        for i in range(0, m, batch_size):
            batch_input = train_input[i:i+batch_size]
            batch_target = train_target[i:i+batch_size]
            pred = net(batch_input)
            #print("below is the prediction", pred, np.sum(pred))

            # Compute gradients of loss w.r.t. weights and biases
            dW, db = net.backward(batch_input, batch_target, lamda)


            # Get updated weights based on current weights and gradients
#             weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)
            
            #adam optimizer
            weights_updated, biases_updated = optimizer.adam_step(net.weights, net.biases, dW, db)

            # Update model's weights and biases
            net.weights = weights_updated
            net.biases = biases_updated

            # Compute loss for the batch
            batch_loss = loss_fn_classification(batch_target, pred, net.weights, net.biases, lamda)
            epoch_loss += batch_loss

            #print(e, i, rmse(batch_target, pred), batch_loss)

        epoch_loss = cross_entropy_loss(train_target, net(train_input))
        epoch_losses.append(epoch_loss)
        print(e, epoch_loss)
        
        dev_loss = cross_entropy_loss(dev_target, net(dev_input))
        dev_losses.append(dev_loss)

    # After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
    no_of_correct_classification = 0
    dev_pred = net(dev_input)

    for predict, target in zip(dev_pred, dev_target):
        print("this is the predicted target and the actual target: ", np.argmax(predict), np.argmax(target))

#     dev_rmse = rmse(dev_target, dev_pred)
    dev_loss_final = cross_entropy_loss(dev_target, dev_pred)
    
    
    print('Cross entr on dev data: {:.5f}'.format(dev_loss_final))
    return epoch_losses, dev_losses, dev_loss_final 


# In[9]:


import csv
def get_test_data_predictions(net, inputs):
    pred = net(inputs)
    dic = {0 : "Very Old", 1:"Old", 2:"Recent", 3:"New"}
    with open('part2.csv', 'w') as output_file:
        csvwriter = csv.writer(output_file)
        csvwriter.writerow(("Id", "Predictions"))
        for idx, predict in enumerate(pred):
            result = dic[np.argmax(predict)]
            csvwriter.writerow((idx+1, result))
    


# In[10]:


def main():
    max_epochs = 128
    batch_size = 32
    learning_rate = 0.0001
    num_layers = 2

    num_units = 8
    lamda = 0.1 #egularization Parameter

    train_input, train_target, dev_input, dev_target, test_input = read_data()
    net = Net(num_layers, num_units)

    optimizer = Optimizer(learning_rate, batch_size, net.weights, net.biases)

    history = train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target)
    get_test_data_predictions(net, test_input)

if __name__ == '__main__':
    main()


# In[ ]:




