import numpy as np

from deeplearning.layers import *
from deeplearning.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {
            'W1': np.random.normal(0, scale=weight_scale, size=(input_dim, hidden_dim)),
            'b1': np.zeros((hidden_dim,)),
            'W2': np.random.normal(0, scale=weight_scale, size=(hidden_dim, num_classes)),
            'b2': np.zeros((num_classes,))
        }
        self.reg = reg

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        hidden, cacheHidden = affine_relu_forward(X, self.params["W1"], self.params["b1"])
        output, cacheOutput = affine_forward(hidden, self.params["W2"], self.params["b2"])
        scores = output

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, dx = softmax_loss(scores, y)
        
        loss = loss + (self.reg / 2) * (np.sum(np.square(self.params["W1"])) + np.sum(np.square(self.params["W2"])))
        
        
        dOut, dW2, db2 = affine_backward(dx, cacheOutput)
        dInput, dW1, db1 = affine_relu_backward(dOut, cacheHidden)
        grads = {
            "W1":dW1 + self.reg*self.params["W1"],
            "b1":db1,
            "W2":dW2 + self.reg*self.params["W2"],
            "b2":db2
        }
        
        
        
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        
        sizes = [input_dim] + hidden_dims + [num_classes]
        
        self.params = {k: v for i in range(0, len(sizes)-1) \
                       for k,v in [\
                                   [f"W{i+1}", np.random.normal(0, scale=weight_scale, \
                                                                size=(sizes[i], sizes[i+1])).astype(self.dtype)], \
                                   [f"b{i+1}", np.zeros((sizes[i+1],)).astype(self.dtype)]
                                  ]}
        if self.use_batchnorm:
            self.params.update({k: v for i in range(0, len(sizes)-2) \
                            for k,v in [\
                                        [f"gamma{i+1}", np.ones((sizes[i+1],))],\
                                        [f"beta{i+1}", np.zeros((sizes[i+1],))]
                                       ]})

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode
        
        forwardPassCache = []
        for i in range(0, self.num_layers - 1):
            #print(i)
            if i == 0:
                inputToLayer = X
            else:
                inputToLayer = forwardPassCache[-1][0]
                
            
            if self.use_batchnorm:
                #self.bn_params[i].update(self.bn_params[i-1 if i else i])
                forwardPassCache.append(affine_batch_relu_forward(inputToLayer, self.params[f"W{i+1}"],\
                                        self.params[f"b{i+1}"], self.params[f"gamma{i+1}"],\
                                        self.params[f"beta{i+1}"], self.bn_params[i]))
            else:
                forwardPassCache.append(affine_relu_forward(inputToLayer, \
                                                        self.params[f"W{i+1}"], self.params[f"b{i+1}"]))
                
            if self.use_dropout:
                out, cache = dropout_forward(forwardPassCache[-1][0], self.dropout_param)
                forwardPassCache[-1] = (out, forwardPassCache[-1][1] + (cache,)) 
         
        
        forwardPassCache.append(affine_forward(forwardPassCache[-1][0], \
                                            self.params[f"W{self.num_layers}"], self.params[f"b{self.num_layers}"]))
        scores = forwardPassCache[-1][0]
        ############################################################################
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores
        
        loss, dx = softmax_loss(scores, y)
        L2_regularization = sum([np.sum(np.square(self.params[f"W{i+1}"])) for i in range(0, self.num_layers)])
        
        #print(forwardPassCache)
        
        loss = loss + (self.reg / 2) * L2_regularization
        backwardPassCache = []
        toIter = list(range(0, self.num_layers))
        toIter.reverse()
        for i in toIter:
            if i == self.num_layers-1:
                dOut = dx
                backwardPassCache.append(affine_backward(dOut, forwardPassCache[i][1]))
            else:
                dOut = backwardPassCache[-1][0]
                
                if self.use_dropout:
                    dOut = dropout_backward(dOut, forwardPassCache[i][1][-1])
                    forwardPassCache[i] = (forwardPassCache[i][0], forwardPassCache[i][1][:-1])
                
                if self.use_batchnorm:
                    backwardPassCache.append(affine_batch_relu_backward(dOut, forwardPassCache[i][1]))
                else:
                    backwardPassCache.append(affine_relu_backward(dOut, forwardPassCache[i][1]))
                
            
            
        backwardPassCache.reverse()
        grads = {k:v for i in range(0, self.num_layers) \
                 for k,v in [\
                             [f"W{i+1}", backwardPassCache[i][1] + self.reg*self.params[f"W{i+1}"]],\
                             [f"b{i+1}", backwardPassCache[i][2]]\
                            ]}
        if self.use_batchnorm:
            grads.update({k:v for i in range(0, self.num_layers - 1) \
                     for k,v in [\
                             [f"gamma{i+1}", backwardPassCache[i][3]],\
                             [f"beta{i+1}", backwardPassCache[i][4]]\
                            ]})
        
        ############################################################################
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        ############################################################################

        return loss, grads
