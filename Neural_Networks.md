### Neural Networks Homework
#### November 2017

##### Reading: https://deeplearning4j.org/neuralnet-overview
#### Neural Network Definition
* Set of algorithms modeled after the human brain that are designed to recognize patterns.
* They can be thought of as a clustering and classification layer on top of the data you store and manage.

#### Neural Network Elements
* Deep learning is the name we use for 'stacked neural networks;' that is, networks composed of several layers.
    * Each layer is made of nodes.
* *Node*: A place where computation happens, loosely patterned on a neuron in the human brain, which fires when it encounters sufficient stimuli.

#### Key Concepts of Deep Neural Networks
* Deep-learning networks are distinguished from the more commonplace single-hidden-layer neural networks by their depth (i.e. the number of node layers).
    * Traditional machine learning relies on shallow nets, composed of one input and one output layer, and at most one hidden layer in between.
    * More than three layers (including input and output) qualifies as 'deep' learning.
        * So deep is a strictly defined, technical term that means more than one hidden layer.
* In deep-learning networks, each layer of nodes trains on a distinct set of features based on the previous layer's output.
    * This is known as *feature hierarchy*, and it is a hierarchy of increasing complexity and abstraction.
* Nets are capable of discovering latent structures within unlabeled, unstructured data, which is the vast majority of the data in the world.
    * Another word for unstructured data is *raw media*.
        * Examples: pictures, texts, video, audio.
* Deep-learning networks perform *automatic feature extraction*, without human intervention, unlike most traditional machine-learning algorithms.
* Deep-learning networks end in an output layer: a logistic, or softmax, classifier that assigns a likelihood to a particular outcome or label.

#### Example: Feedforward Networks
* Three key functions of neural networks:
    1) Scoring input
    2) Calculating loss
    3) Applying an update to the model to begin the three-step process over again.
* A neural network is a corrective feedback loop.
* The name for one commonly used optimization function that adjusts weights according to the error they caused is called 'gradient descent.'
    * Gradient is another word for slope, and slope in its typical form on an x-y graph, represents how two variables relate to each other.

##### Reading: https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/nn.pdf
#### Carnegie Mellon's C.S. lecture: "Neural Networks: A Simple Problem"

##### Reading: https://iamtrask.github.io/2015/07/12/basic-python-network/
#### A Neural Network in 11 lines of Python (Part 1)

~~~
E = .5*(yp - y[0])**2

def loss(X, y):
    errors = []
    w0_list = []
    for i in np.linspace(0,1,200):
        w0, w1 = i,i
        X_lsq = np.vstack([np.ones(len(X)), X]).T
        yp = X_lsq[:,0]*w0 + X_lsq[:,1]*w1
        E = .5*(yp - y)**2
        errors.append(E)
        w0_list.append(i)
    return w0_list[np.argmin(errors)]
~~~
print(loss(X,y))

~~~
import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset


# output dataset            

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print "Output After Training:"
print l1
