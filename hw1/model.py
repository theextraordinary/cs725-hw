import numpy as np


def standardize(input_x,d):
    means=[]
    for i in range(d):
        means.append(np.mean(input_x[:,i]))
    # print(means)
    stds=[]
    for i in range(d):
        stds.append(np.std(input_x[:,i]))
    # print(stds)
    for i in range(d):
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
        # self.velocity=np.zeros((self.d+1,self.num_classes))
    

    
    def preprocess(self, input_x):
        """
        Preprocess the input any way you seem fit.
        """
        # print(input_x)
        return standardize(input_x)
    
    def hofx(self,x):
        x=np.hstack((np.ones((x.shape[0],1)),x))
        # print("In h(x):\n\n x shape:\n",x.shape)
        # print("w shape:\n",self.weights.shape)
        z=np.dot(x,self.weights)
        # print("z shape:\n",z.shape)
        return z


    def sigmoid(self, z):
        # print("In sigmoid:\n\n")
        g=1/(1+np.exp(-z))
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        return g
        

    def calculate_loss(self, input_x, input_y):
        input_y=input_y.reshape((input_y.shape[0],1))
        # print("In loss:\n\n")
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        p=self.sigmoid(self.hofx(input_x))
        cost=-1/input_x.shape[0]*(np.dot(np.transpose(input_y),np.log(p))+np.dot(np.transpose(1-input_y),np.log(1-p)))
        # print("Cost:",cost)
        return cost[0][0]

    def calculate_gradient(self, input_x, input_y):
        input_y=input_y.reshape((input_y.shape[0],1))
        # print("In gradient:\n\n")
        #Gradient of logistic loss dJ/dw is (p-y)*x where p is sig(w.x)
        p=self.sigmoid(self.hofx(input_x))
        # print("p shape",p.shape)
        # print("shape y",input_y.shape)
        temp=p-input_y
        x=np.hstack((np.ones((input_x.shape[0],1)),input_x))
        # print("x shape",x.shape)
        # print("temp",temp.shape)
        gradient=np.dot(np.transpose(x),temp)
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        return (1/input_x.shape[0])*gradient

    def update_weights(self, grad, learning_rate, momentum):
        # print("In update:\n\n")
        
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """
        # self.velocity=momentum*self.velocity+(1-momentum)*grad
        # self.weights=self.weights-learning_rate*self.velocity
        self.weights=self.weights-learning_rate*grad*1/150
        pass

    def get_prediction(self, input_x):
        # print("In prediction:\n\n")
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,) 
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        z=self.hofx(input_x)
        g=self.sigmoid(z)
        y=[]
        for i in g:
            if i>=0.5:
                y.append(1)
            else:
                y.append(0)
        y=np.array(y)
        return y

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
        return standardize(train_x,self.d)

    def hofx(self,x):
        x=np.hstack((np.ones((x.shape[0],1)),x))
        # print("In h(x):\n\n x shape:\n",x.shape)
        # print("w shape:\n",self.weights.shape)
        z=np.dot(x,self.weights)
        # print("z shape:\n",z.shape)
        return z


    def sigmoid(self, z):
        # print("In sigmoid:\n\n")
        g=1/(1+np.exp(-z))
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        return g
        
    def process_y(self,classs,input_y):
        new_y=[]
        for i in input_y:
            if i!=classs:
                new_y.append(0)
            else:
                new_y.append(1)
        return np.array(new_y)

    def calculate_loss(self, input_x, input_y):

        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        input_y0=self.process_y(0,input_y)
        input_y1=self.process_y(1,input_y)
        input_y2=self.process_y(2,input_y)
        m=input_x.shape[0]
        p=self.sigmoid(self.hofx(input_x))
        loss0=-1/m*(np.dot(np.transpose(input_y0),np.log(p[:,0]))+np.dot(np.transpose(1-input_y0),np.log(1-p[:,0])))
        loss1=-1/m*(np.dot(np.transpose(input_y1),np.log(p[:,1]))+np.dot(np.transpose(1-input_y1),np.log(1-p[:,1])))
        loss2=-1/m*(np.dot(np.transpose(input_y2),np.log(p[:,2]))+np.dot(np.transpose(1-input_y2),np.log(1-p[:,2])))

        return loss0+loss1+loss2/3

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        p=self.sigmoid(self.hofx(input_x))
        input_y0=self.process_y(0,input_y).reshape((input_y.shape[0],1))
        input_y1=self.process_y(1,input_y).reshape((input_y.shape[0],1))
        input_y2=self.process_y(2,input_y).reshape((input_y.shape[0],1))
        input_y=np.hstack((input_y0,input_y1,input_y2))
        # print("Input y: ",input_y)
        temp=p-input_y
        x=np.hstack((np.ones((input_x.shape[0],1)),input_x))
        gradient=np.dot(np.transpose(x),temp)
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
        y=self.sigmoid(self.hofx(input_x))
        # print("Y ",y)
        new_y=[]
        for i in y:
            new_y.append(np.argmax(i))
        
        return np.array(new_y)


# data=np.load(r'C:\IITB\FML\HW1\cs725-hw\hw1\data\iris\train_x.npy')
# y=np.load(r'C:\IITB\FML\HW1\cs725-hw\hw1\data\iris\train_y.npy')
# print(data,y)
