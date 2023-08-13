import numpy as np


def standardize(input_x):
    means=[]
    for i in range(2):
        means.append(np.mean(input_x[:,i]))
    # print(means)
    stds=[]
    for i in range(2):
        stds.append(np.std(input_x[:,i]))
    # print(stds)
    for i in range(2):
        x=input_x[:,i]
        x=(x-means[i])/stds[i]
        # print(x)
        input_x[:,i]=x
    # print(input_x)
    return input_x

def normalize(input_x):
    maxv=[]
    minv=[]
    for  i in range(2):
       minv.append(np.min(input_x[:,i]))
       maxv.append(np.max(input_x[:,i]))

    for i in range(2):
        x=input_x[:,i]
        x=(x-minv[i])/(maxv[i]-minv[i])
        input_x[:,i]=x
    print(maxv)
    print(minv)
    print(input_x)
    return input_x    

class LogisticRegression:
    def __init__(self):
        """
        Initialize `self.weights` properly. 
        Recall that for binary classification we only need 1 set of weights (hence `num_classes=1`).
        We have given the default zero intialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 1 # single set of weights needed
        self.d = 2 # input space is 2D. easier to visualize
        self.weights = np.zeros((self.d+1, self.num_classes))
    

    
    def preprocess(self, input_x):
        """
        Preprocess the input any way you seem fit.
        """
        # print(input_x)
        return standardize(input_x)

    def sigmoid(self, x):


        ones_column=np.ones((x.shape[0], 1))
        new_array = np.hstack((ones_column,x))
        # print(new_array)
        z=new_array.dot(self.weights)

        z=-z
        funz=1/(1+np.exp(z))
        # print("In sigmoid")
        # print("weights :",self.weights)
        # print("x :",new_array)
        # print("sig(z) :",funz)
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        return funz 
        

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        p=self.sigmoid(input_x)
        cost=-np.sum(input_y*np.log(p)+(1-input_y)*np.log(1-p))
        return cost

    def calculate_gradient(self, input_x, input_y):
        #Gradient of logistic loss dJ/dw is (p-y)*x where p is sig(w.x)
        ones_column=np.ones((input_x.shape[0], 1))
        new_array = np.hstack((ones_column,input_x))
        temp=self.sigmoid(input_x)-input_y
        # print(temp)
        gradient=np.transpose(new_array).dot(temp)
        # print(gradient)

        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        return gradient

    def update_weights(self, grad, learning_rate, momentum):
        
        
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """
        self.weights=self.weights-learning_rate*grad
        pass

    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,) 
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        ones_column=np.ones((input_x.shape[0], 1))
        new_array = np.hstack((ones_column,input_x))
        z=new_array.dot(self.weights)
        return z

class LinearClassifier:
    def __init__(self):
        """
        Initialize `self.weights` properly. 
        We have given the default zero intialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 3 # 3 classes
        self.d = 4 # 4 dimensional features
        self.weights = np.zeros((self.d+1, self.num_classes))
    
    def preprocess(self, train_x):
        """
        Preprocess the input any way you seem fit.
        """
        return train_x

    def sigmoid(self, x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        pass

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        pass

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        pass

    def update_weights(self, grad, learning_rate, momentum):
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """
        pass

    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,) 
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        pass

lr=LogisticRegression()
import os
data=np.load(r'C:\IITB\FML\HW1\cs725-hw\hw1\data\binary\train_x.npy')
y=np.load(r'C:\IITB\FML\HW1\cs725-hw\hw1\data\binary\train_y.npy')
y=y.reshape((150,1))
print(data.shape)
print(y.shape)
# data=np.array([[1,2],[3,4],[5,9],[8,7]])
# y=np.array([0,0,1,1]).reshape((4,1))
data=lr.preprocess(data)
for i in range(500):
    print("Epoch :",(i+1))
    print("weights :\n",lr.weights)
    loss=lr.calculate_loss(data,y)
    print("loss :\n",loss)
    grad=lr.calculate_gradient(data,y)
    lr.update_weights(grad,0.3,1)
    print("---------------------------------------------------------------")


