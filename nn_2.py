#!/usr/bin/env python
# coding: utf-8

# In[37]:


import sys
import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt


# In[38]:


np.random.seed(42)
NUM_FEATS = 90


# In[39]:


def read_data():
    train_input = []
    train_target = []
    dev_input = []
    dev_target = []
    test_input = []

    with open('train.csv', 'r') as training_file:
        training_read = csv.reader(training_file)
        training_read.__next__()
        for line in training_read:
            line = [int(line[i]) if i == 0 else float(line[i]) for i in range(len(line))]
            train_target.append(line[0])
            train_input.append(line[1:])

    print(len(train_input), len(train_input[0]))
    print(len(train_target))
    train_input = np.array(train_input).reshape(len(train_input), 90)
    
    #data standardization
#     train_input = np.array([(t_input-np.mean(t_input))/(np.variance(t_input)) for t_input in train_input])
    
    #data min max scaling
    train_input = np.array([(t_input-np.min(t_input))/(np.max(t_input)-np.min(t_input)) for t_input in train_input])
    
    train_target = np.array(train_target).reshape(len(train_target), 1)



    with open('dev.csv', 'r') as dev_file:
        dev_read = csv.reader(dev_file)
        dev_read.__next__()
        for line in dev_read:
            line = [int(line[i]) if i == 0 else float(line[i]) for i in range(len(line))]
            dev_target.append(line[0])
            dev_input.append(line[1:])
    dev_input = np.array(dev_input).reshape(len(dev_input), 90)
    
    #stadardization
#     dev_input = np.array([(d_input-np.mean(d_input))/(np.std(d_input)) for d_input in dev_input])
    
    #min-max scaling
    dev_input = np.array([(d_input-np.min(d_input))/(np.max(d_input)-np.min(d_input)) for d_input in dev_input])
    
    dev_target = np.array(dev_target).reshape(len(dev_target),1)
    
    with open('test.csv', 'r') as test_file:
        test_read = csv.reader(test_file)
        test_read.__next__()
        for line in test_read:
            line = [float(line[i]) for i in range(len(line))]
            test_input.append(line)
        test_input = np.array(test_input).reshape(len(test_input), 90)
        
        #standardization
#         test_input = np.array([(t_input-np.min(t_input))/(np.max(t_input)-np.min(t_input)) for t_input in test_input])

        #min max scaling
        test_input = np.array([(t_input-np.min(t_input))/(np.max(t_input)-np.min(t_input)) for t_input in test_input])



	#return train_input, train_target, dev_input, dev_target, test_input, test_target  #for mnist data
    return train_input, train_target, dev_input, dev_target, test_input


# In[40]:


def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return np.where(z > 0, 1.0, 0.0)


# In[41]:


def loss_mse(y, y_hat):
    return np.sum((y-y_hat)**2)/(2*y.shape[0])

def loss_regularization(weights, biases):
    return np.sum([np.linalg.norm(w) for w in weights])

def loss_rmse(y, y_hat):
    loss = loss_mse(y, y_hat)
    return np.sqrt(loss)

def loss_fn(y, y_hat, weights, biases, lamda):
    return loss_mse(y, y_hat) + lamda * loss_regularization(weights, biases)

def rmse(y, y_hat):
    return (np.sum((y-y_hat)**2)/(2*y.shape[0]))**0.5



class AdamOptimizer:
    def __init__(self, weights, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.alpha = alpha
        self.m = [np.zeros(w.shape) for w in weights]
        self.v = [np.zeros(w.shape) for w in weights]
        self.theta = weights

    def step(self, gradients):
        self.t = self.t + 1
        for idx, gradient in enumerate(gradients):            
            self.m[idx] = self.beta1*self.m[idx] + (1 - self.beta1)*gradient
            self.v[idx] = self.beta2*self.v[idx] + (1 - self.beta2)*(gradient**2)
            m_hat = self.m[idx]/(1 - self.beta1**self.t)
            v_hat = self.v[idx]/(1 - self.beta2**self.t)
            self.theta[idx] = self.theta[idx] - self.alpha*(m_hat/(np.sqrt(v_hat) - self.epsilon))
        return self.theta





class Optimizer(object):

    def __init__(self, learning_rate, batch_size, weights, biases):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.adam_weight = AdamOptimizer(weights, learning_rate)
        self.adam_bias = AdamOptimizer(biases, learning_rate)
        
    def adam_step(self, delta_weights, delta_biases):
        new_weights = self.adam_weight.step(delta_weights)
        new_biases = self.adam_bias.step(delta_biases)
        return new_weights, new_biases

    def step(self, weights, biases, delta_weights, delta_biases):
        new_weights = [weight - self.learning_rate * delta_of_weight for weight, delta_of_weight in zip(weights, delta_weights)]
        new_biases = [bias - self.learning_rate * delta_of_bias for bias, delta_of_bias in zip(biases, delta_biases)]
        return new_weights, new_biases



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
                self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))# * np.sqrt(2 / NUM_FEATS))
            else:
                # Hidden layer
                self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))# * np.sqrt(2 / num_units))

            self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1))) #* np.sqrt(2 / num_units))

        # Output layer
        self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))# * np.sqrt(2))
        self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))# * np.sqrt(2 / num_units))

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
                a = h
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
        loss_gradient_weight = np.dot(self.a_states[-1].T, (pred - y))
        loss_gradient_bias = np.dot(np.ones([1, batch_size]), pred - y)
        
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




def train(
    net, optimizer, lamda, batch_size, max_epochs,
    train_input, train_target,
    dev_input, dev_target
):
    m = train_input.shape[0]
    epoch_losses = []
    dev_losses = []
    for e in range(max_epochs):
        learning_rate = 0.00001
        epoch_loss = 0.
        r_sqr = 0
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
            weights_updated, biases_updated = optimizer.adam_step(dW, db)

            # Update model's weights and biases
            net.weights = weights_updated
            net.biases = biases_updated

            # Compute loss for the batch
            batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
            epoch_loss += batch_loss

            #print(e, i, rmse(batch_target, pred), batch_loss)

        epoch_loss = loss_mse(train_target, net(train_input))
        epoch_losses.append(epoch_loss)
        print(e, epoch_loss)
        
        dev_loss = rmse(dev_target, net(dev_input))
        dev_losses.append(dev_loss)

    # After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
    no_of_correct_classification = 0
    dev_pred = net(dev_input)

    for predict, target in zip(dev_pred, dev_target):
        print("this is the predicted target and the actual target: ", predict, target)

    dev_rmse = rmse(dev_target, dev_pred)
    
    print('RMSE on dev data: {:.5f}'.format(dev_rmse))
    return epoch_losses, dev_losses, dev_rmse 




def driver(max_epochs, batch_size, learning_rate, num_layers, num_units, train_input,
          train_target, dev_input, dev_target, lamda):
    
    net = Net(num_layers, num_units)
    optimizer = Optimizer(learning_rate, batch_size, net.weights, net.biases)
    history = train(net, optimizer, lamda, batch_size, max_epochs,train_input, train_target,dev_input, dev_target)
    
    return net, history







def get_test_data_predictions(net, inputs):
    pred = net(inputs)
    with open('part2.csv', 'w') as output_file:
        csvwriter = csv.writer(output_file)
        csvwriter.writerow(("Id", "Predictions"))
        for idx, predict in enumerate(pred):
            csvwriter.writerow((idx+1, predict[0]))
    

def k_fold(data, k):
    m = len(data)
    batch_size = m//k
    idx_list = np.arange(0, k * batch_size, batch_size)
#     print(idx_list)
    validation_list = []
    train_list  =[]
    for idx in idx_list:
        validation_list.append(data[idx:idx+batch_size, :])
#         print(validation_list)
        train_list.append(np.concatenate( (data[:idx, :], data[idx+batch_size:, :]), axis = 0))
    return validation_list, train_list



def k_fold_cross_validation_for_tuning():
    
    max_epoch_list = [50, 100]
    # batch_size_list = [128, 256]
    batch_size_list = [256]
    learning_rate_list = [0.001]
    # learning_rate_list = [0.001, 0.01]
    num_layers_list = [2, 3]
    num_units_list = [32]
    lamda = 0


    train_input, train_target, dev_input, dev_target, test_input = read_data()
    train_list, val_list = k_fold(train_input, 5)
    train_target_list, val_target_list = k_fold(train_target, 5)

    best_models = []
    histories = []

    min_dev_loss = 10000000


    dev_loss = 0.1
    model_list = []
    best_parameter = {}

    for max_epoch in max_epoch_list:
        for batch_size in batch_size_list:
            for learning_rate in learning_rate_list:
                for num_layers in num_layers_list:
                    for num_units in num_units_list:
                        best_models = []
                        model_list = []
                        dev_loss = 0
                        for k in range(5):
                            model, history = driver(max_epoch, batch_size, learning_rate, num_layers, num_units, 
                                                    train_list[k], train_target_list[k], 
                                                    val_list[k], val_target_list[k], lamda)
                            dev_loss += history[2]
                            model_list.append(model)
                        dev_loss = dev_loss/5
                        print("one combination finished with avg loss: ", dev_loss)
                        if(min_dev_loss > dev_loss):
                            print("models updated")
                            best_models = model_list
                            best_parameter["max_epoch"] = max_epoch
                            best_parameter["batch_size"] = batch_size
                            best_parameter["learning_rate"] = learning_rate
                            best_parameter["num_layers"] = num_layers
                            best_parameter["num_units"] = num_units                        

    #find the avg of best model

    weight = np.mean([x.weights for x in best_models], axis=0)
    bias = np.mean([x.biases for x in best_models], axis=0)


    net = Net(best_parameter["num_layers"], best_parameter["num_units"])

    net.weights = weight
    net.biases = bias

    dev_pred = net(dev_input)
    for predict, target in zip(dev_pred, dev_target):
        print("this is the predicted target and the actual target: ", predict, target)
    dev_rmse = rmse(dev_target, dev_pred)
    print('RMSE on dev data: {:.5f}'.format(dev_rmse))





# In[49]:


def main():
                            
    k_fold_cross_validation_for_tuning()
    
if __name__=='__main__':
    main()

