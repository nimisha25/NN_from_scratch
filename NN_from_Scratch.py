import torchvision as thv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

def create_dataset(train, val) :
    dict_train = defaultdict(list)
    dict_val = defaultdict(list)
    for x,y in zip(train.data, train.targets):
        dict_train[int(y)].append(x)
    
    for x,y in zip(val.data, val.targets):
        dict_val[int(y)].append(x)
    # print(dict_train.keys(), dict_val.keys())

    X_train, y_train = [], []
    X_val, y_val = [], []

    for key in dict_train:
        num_samples = len(dict_train[key])
        half = num_samples//2
        X_train.extend(dict_train[key][:half])
        y_train.extend([key]*half)
    
    for key in dict_val:
        num_samples = len(dict_val[key])
        half = num_samples//2
        X_val.extend(dict_val[key][:half])
        y_val.extend([key]*half)

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    X_train = X_train.reshape(X_train.shape[0], -1)  # Reshape to (num_samples, 784)
    X_val = X_val.reshape(X_val.shape[0], -1)  # Reshape to (num_samples, 784)

    return X_train/255.0, y_train, X_val/255.0, y_val

def plot_random_images(X, y, num_images=5) :
    # Select 'num_images' random indices from the dataset
    random_indices = random.sample(range(X.shape[0]), num_images)
    
    # Create a plot for the images
    plt.figure(figsize=(10, 10))
    
    for i, idx in enumerate(random_indices):
        plt.subplot(1, num_images, i+1)  # Create subplots
        plt.imshow(X[idx].reshape(28, 28), cmap='gray')  # Reshape to 28x28 and display
        plt.title(f"Label: {y[idx]}")
        plt.axis('off')  # Hide axes

    plt.show()

class linear_t:
    # h_l is the output of the previous layer (after activation).
    # h_(l+1) = W^T h_l + b gives the raw output for the current layer.
    # Activation function is applied to h_(l+1) to add non-linearity.
    # The result is passed to the next layer.
    # dim: w = cxa,  b=cx1, h_l = txa, hl_plus1 = txc, b = cx1
    def __init__(self, input_size, output_size):
        self.w = np.random.randn(output_size, input_size) * np.sqrt(2.0 / (input_size + output_size))
        # self.b = np.random.normal(size=(10,1)).astype(np.float32)
        # self.w = np.random.normal(size=(10,784)).astype(np.float32)
        self.b = np.zeros((output_size,1))
        self.b = self.b+0.0001
        # frobenius_norm_w = np.linalg.norm(self.w, ord='fro')
        # self.w = self.w/frobenius_norm_w  #to stabilise the gradients
        # frobenius_norm_b = np.linalg.norm(self.b, ord='fro')
        # self.b = self.b/frobenius_norm_b
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.h_l = None
        print("weights: ", self.w[0][:10], 'bias :', self.b)
        
    def forward(self, h_l) :
        # print("shape of b: ", self.b.shape, "shape of h_l: ", h_l.shape, 'shape of w: ', self.w.shape)
        h_lplus1 = np.matmul(h_l, np.transpose(self.w)) + self.b.T #same as np.transpose(b)
        self.h_l = h_l
        return h_lplus1

    def backward(self, dh_lplus1) :
        # dh_l = #dl/dh_lplus1 x dh_lplus1/dh_l = grad_hplus1 x W
        dh_l = np.matmul(dh_lplus1, self.w)    # During backpropagation, we revert to the original w
        dw = np.matmul(np.transpose(dh_lplus1), self.h_l)
        db = np.matmul(dh_lplus1.T, np.ones([self.h_l.shape[0], 1]))
        self.dw, self.db = dw, db
        return dh_l
    
    def zero_grad(self) :
        self.dw, self.db = 0.0*self.dw, 0.0*self.db
    
    def backward_check(self):
        # We now compute the estimate of the derivative using finite differences, w = [c,a]
        i, j = 3, 2
        k = i
        epsilon = np.random.normal()  # Perturbation size
        e = np.zeros_like(self.w)  # Create a zero matrix of the same shape as self.w
        e[i, j] = epsilon  # Perturbation in the first element for simplicity
        
        # Compute the perturbed gradients
        grad_plus = np.matmul(self.h_l, (self.w + e).T)[0,k]  # Forward pass with perturbed weights
        grad_minus = np.matmul(self.h_l, (self.w - e).T)[0,k]  # Forward pass with perturbed weights
        
        # Compute the finite difference approximation of the gradient for w
        dw_c = (grad_plus - grad_minus) / (2 * epsilon)
        
        # Create a dummy dh_plus1_c for the backward check, batch_size = 1, num_classes = self.w.shape[0]
        dh_plus1_c = np.zeros((1, self.w.shape[0]))  # shape: [1, num_classes]
        dh_plus1_c[0, k] = 1  

        backprop_gradients = self.backward(dh_plus1_c)
        
        # Print the results and compare them
        print("Finite difference gradient: ", dw_c)
        print("Backpropagation gradient: ", self.dw[i,j])

class relu_t :
    def __init__(self) :
        self.h_l = 0.0

    def forward(self, h_l) :
        h_lplus1 = np.maximum(0.0, h_l)
        self.h_l = h_l
        return h_lplus1
        
    def backward(self, dh_lplus1) :
        mask = (self.h_l > 0).astype(int)
        dh_l = np.multiply(dh_lplus1, mask)
        return dh_l
    
    def zero_grad(self) :
        self.h_l = self.h_l*0.0

class softmax_cross_entropy_t:
    #y is a 1D array of true labels of the images in the batch
    # h_l is the output of the previous layer (logits), each element represents the score for kth class for the ith 
    # sample in the batch, shape: [t,c]
    def __init__(self) :
        self.h_l = 0.0
        self.h_lplus1 = 0.0
        self.y = 0.0
    def forward(self, h_l, y):  
        self.h_l = h_l
        self.y = y
        # Stabilize logits by subtracting the maximum logit for each row
        h_l_stable = h_l - np.max(h_l, axis=1, keepdims=True)

        # Compute softmax probabilities
        # exp = np.exp(h_l_stable)
        # total = np.sum(exp, axis=1, keepdims=True)
        # # print('total ', total)
        # self.h_lplus1 = exp/total
    
        self.h_lplus1 = self.softmax(h_l)
        log_prob = -np.log(self.h_lplus1[range(h_l.shape[0]), y.astype(int)])
        # Compute average loss over the batch
        ell = np.sum(log_prob)/h_l.shape[0]

        # Compute classification error (fraction of incorrect predictions)
        y_pred = np.argmax(self.h_lplus1, axis=1)
        error = np.mean(y_pred != y)

        return ell, error
    
    def backward(self) :        # ??????? check 
        # Create a copy of the softmax probabilities
        
        dh_l = self.h_lplus1.copy()

        # Subtract 1 for the correct class in the probabilities
        dh_l[np.arange(self.h_l.shape[0]), self.y] -= 1

        # Normalize by the batch size
        dh_l /= self.h_l.shape[0]

        return dh_l
    
    def softmax(self, z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis] # necessary step to do broadcasting
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis] # dito
        return e_x / div
    
    def zero_grad(self) :
        self.y, self.h_l, self.h_lplus1 = self.y*0.0, self.h_l*0.0, self.h_lplus1*0.0

def smooth_curve(data, smoothing_factor=0.99):
    """Apply exponential moving average for smoothing."""
    smoothed_data = []
    last = data[0]  # Initialize with first value
    for point in data:
        last = last * smoothing_factor + (1 - smoothing_factor) * point
        smoothed_data.append(last)
    return smoothed_data

def plot_loss_error(training_loss, training_error, val_loss=None, val_error=None):
    """
    visualization for loss & error with smoothing & separate plots.
    """
    plt.figure(figsize=(12, 10))

    # Loss Plot
    plt.subplot(2, 1, 1)
    plt.plot(smooth_curve(training_loss), label="Training Loss", color='blue')
    if val_loss:
        plt.plot(smooth_curve(val_loss), label="Validation Loss", color='red', linestyle='dashed')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss vs. Iterations")
    plt.legend()

    # Error Plot
    plt.subplot(2, 1, 2)
    plt.plot(smooth_curve(training_error), label="Training Error", color='orange')
    if val_error:
        plt.plot(smooth_curve(val_error), label="Validation Error", color='green', linestyle='dashed')
    plt.xlabel("Iterations")
    plt.ylabel("Error Rate")
    plt.title("Error vs. Iterations")
    plt.legend()

    plt.tight_layout()
    plt.show()

def validate(l1, l2,l3,l4, X_val, y_val) :
    total_loss, total_error = 0,0
    num_batches = 0
    for i in range (0, X_val.shape[0], 32) :
        x, y = X_val[i:i+32], y_val[i:i+32]
        # forward pass
        h1 = l1.forward(x)
        h2 = l2.forward(h1)
        h3 = l3.forward(h2)
        ell, error = l4.forward(h3, y)
        total_error += error * len(y)
        total_loss += ell * len(y)
        num_batches += len(y)
    # plot_error_loss(error_l, loss_l)
    return total_loss/num_batches, total_error/num_batches

if __name__ == '__main__' :
    train = thv.datasets.MNIST('./', download=True , train=True)
    val = thv.datasets.MNIST('./', download=True , train=False)
    # print(train.data.shape , len(train.targets))
    # print(type(train.targets[0]))
    X_train, y_train, X_val, y_val = create_dataset(train, val)
    print(X_train.shape, X_val.shape)
    print(X_train[0][:10], y_train[0])
    # plot_random_images(X_train, y_train,5)
    # l1 = linear_t()
    #print(l1.b[:10], l1.w[0][0])
    # ll = linear_t()
    # h_l = np.random.randn(1,10)
    # h_lplus1 = ll.forward(h_l)
    # print('Output from forward:', h_l)
    # ll.backward_check()

    # initialise all the layers
    l1, l2, l3, l4 = linear_t(784,128), relu_t(), linear_t(128,10), softmax_cross_entropy_t()
    net = [l1, l2, l3, l4]
    lr = 0.1
    batch_size = 32
    patience = 5000

    training_loss, training_error = [], []
    val_loss_list, val_error_list = [], []
    best_val_loss = float('inf')  # Initialize best validation loss
    best_weights = None
    patience_counter = 0

    for t in range(50000) :
        indices = np.random.choice(len(X_train), batch_size, replace=False)
        X, y = X_train[indices], y_train[indices]

        # Zero gradient buffer
        for l in net :
            l.zero_grad()
        
        # Forward pass
        h1 = l1.forward(X)
        h2 = l2.forward(h1)
        h3 = l3.forward(h2)
        ell, error = l4.forward(h3, y)

        # Backward pass
        dh3 = l4.backward()
        dh2 = l3.backward(dh3)
        dh1 = l2.backward(dh2)
        dx = l1.backward(dh1)

        # Update weights using SGD
        l1.w -= lr * l1.dw
        l1.b -= lr * l1.db
        l3.w -= lr * l3.dw
        l3.b -= lr * l3.db

        # Store training loss & error
        training_error.append(error)
        training_loss.append(ell)

        # Check validation loss every 100 iterations
        if t % 100 == 0:
            val_loss, val_error = validate(l1, l2, l3, l4, X_val, y_val)
            val_loss_list.append(val_loss)
            val_error_list.append(val_error)

            # Check for improvement in validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = (l1.w.copy(), l1.b.copy())  # Save best weights
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 100  

            print(f"Iteration {t}, Loss: {ell:.4f}, Error: {error:.4f}, Val Loss: {val_loss:.4f}, Val Error: {val_error:.4f}, Patience: {patience_counter}")

        if patience_counter >= patience:
            print(f" Early stopping triggered at iteration {t}!")
            break

    # Restore best model weights
    l1.w, l1.b = best_weights

    print("Final Validation Results:", validate(l1, l2, l3, l4, X_val, y_val))

    # Plot the loss & error curves
    plot_loss_error(training_loss, training_error)



