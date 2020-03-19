import numpy as np
import copy

from deeplearning.layers import *
from deeplearning.fast_layers import *
from deeplearning.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.use_dropout = (dropout > 0)
        C, H, W = input_dim
        
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
        
        conv_stride = 1
        pad = (filter_size - 1)//2
        pool_stride = 2
        
        Hout = int( ( (((H+2*pad-filter_size)/conv_stride) + 1) - 2)/2 + 1 )
        Wout = int( ( (((W+2*pad-filter_size)/conv_stride) + 1) - 2)/2 + 1 )

        self.params = {
            "W1": np.random.normal(0, scale=weight_scale, size=(num_filters,C,filter_size,filter_size )).astype(self.dtype),
            "b1": np.zeros((num_filters,)).astype(self.dtype),
            "W2": np.random.normal(0, scale=weight_scale, size=(Hout*Wout*num_filters,hidden_dim)).astype(self.dtype),
            "b2": np.zeros((hidden_dim,)).astype(self.dtype),
            "W3": np.random.normal(0, scale=weight_scale, size=(hidden_dim, num_classes)).astype(self.dtype),
            "b3": np.zeros((num_classes,)).astype(self.dtype)
        }
        
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        convOut, convCache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        affineReluOut, affineReluCache = affine_relu_forward(convOut, W2, b2)
        scores, scoresCache = affine_forward(affineReluOut, W3, b3)
        

        if y is None:
            return scores

        loss, dx = softmax_loss(scores, y)
        
        regularization_loss = np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))
        loss = loss + (self.reg / 2) * regularization_loss
        
        
        dscore, dw3, db3 = affine_backward(dx, scoresCache)
        drelu, dw2, db2 = affine_relu_backward(dscore, affineReluCache)
        dInput, dw1, db1 = conv_relu_pool_backward(drelu, convCache)
        
        grads = {
            "W1": dw1 + (self.reg)*W1,
            "b1": db1,
            "W2": dw2 + (self.reg)*W2,
            "b2": db2,
            "W3": dw3 + (self.reg)*W3,
            "b3": db3
        }

        return loss, grads


class MultiLayerConvolutionNetwork(object):
    
    
    def __init__(self, filter_params, pool_params, hidden_dims, input_dim=(3, 32, 32), num_classes=10, 
                 weight_scale=1e-3, reg=0.0, dropout=0, seed=None, snapshots=0, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - filter_params:
            List of tuples for convolutional layers
            [ (num_filters, filter_size, pad, stride) ]
        - pool_params:
            List of tuples for pooling layers subject to len(filter_params) == len(pool_params)
            If you want conv-conv-pool, the pool_params should be  [ None, pool_param_2 ]
            If you want conv-pool-conv-pool, the pool_params should be [ pool_param_1, pool_param_2 ]
            
            pool_param_2 format should be as follows:
            [ (poolH, poolW, stride) ]
        - hidden_dims:
            List of sizes for hidden dimenson.
        """
        self.params = {}
        self.convParams = {}
        self.poolParams = {}
        
        self.reg = reg
        self.dtype = dtype
        self.num_classes = num_classes
        
        self.use_dropout = (dropout > 0)
        
        self.enable_snapshot = (snapshots > 0)
        if self.enable_snapshot:
            self.num_snapshots = snapshots
            self.snapshots = []
        
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
                
        C, H, W = input_dim
        
        self.num_layers = len(filter_params) + len(hidden_dims) + 1
        Cout = C
        Hout = H
        Wout = W
        prev_i = None
        for filter_param, pool_param, i in zip(filter_params, pool_params, list(range(1, self.num_layers - len(hidden_dims) + 1))):
            num_filters, filter_size, pad, conv_stride = filter_param
            
            if pool_param:
                poolH, poolW, stride = pool_param
                self.poolParams[f"W{i}"] = {'pool_height': poolH, 'pool_width': poolW, 'stride': stride}
                
            
            self.params[f"W{i}"] = np.random.normal(0, scale=weight_scale, size=(num_filters, Cout, 
                                                                                       filter_size, filter_size)).astype(self.dtype) 
            self.params[f"b{i}"] = np.zeros((num_filters,)).astype(self.dtype)
            self.convParams[f"W{i}"] = {'stride': conv_stride, 'pad': pad}
            
            Cout = num_filters
            Hout = int(((Hout+2*pad-filter_size)/conv_stride) + 1) 
            Wout = int(((Wout+2*pad-filter_size)/conv_stride) + 1)
            
            if pool_param:
                Hout = int((Hout - poolH)/stride + 1)
                Wout = int((Wout-poolW)/stride + 1)
                
            prev_i = i
                
            
        prev_dim = Cout*Hout*Wout   
        hidden_dims.append(num_classes)
        for hidden_dim, i in zip(hidden_dims, list(range(prev_i + 1, self.num_layers + 2))):
            self.params[f"W{i}"] = np.random.normal(0, scale=weight_scale, size=(prev_dim, hidden_dim)).astype(self.dtype)
            self.params[f"b{i}"] = np.zeros((hidden_dim,)).astype(self.dtype)
                
            prev_dim = hidden_dim   
            
    
    def appendSnapshot(self):
        self.snapshots.append(copy.deepcopy(self.params))
    
    def forwardPass(self, X, mode, snapshot=None):
        paramsToUse = self.params if snapshot is None else snapshot
        
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        
        forwardPassCache = []
        for i in range(0, self.num_layers - 1):

            inputToLayer = X if i == 0 else forwardPassCache[-1][0]
             
            W,b, gamma, beta= f"W{i+1}", f"b{i+1}", f"gamma{i+1}", f"beta{i+1}"
            
            if W in self.convParams:
                convParams = self.convParams[W]
                if W in self.poolParams:
                    poolParams = self.poolParams[W]
                    forwardPassCache.append(conv_relu_pool_forward(inputToLayer, paramsToUse[W], paramsToUse[b], 
                                                                   convParams, poolParams))
                else:
                    forwardPassCache.append(conv_relu_forward(inputToLayer, paramsToUse[W], paramsToUse[b],
                                                             convParams))
            else:                
                forwardPassCache.append(affine_relu_forward(inputToLayer, paramsToUse[W], paramsToUse[b]))
                
                if self.use_dropout:
                    out, cache = dropout_forward(forwardPassCache[-1][0], self.dropout_param)
                    forwardPassCache[-1] = (out, forwardPassCache[-1][1] + (cache,)) 
                
        
        forwardPassCache.append(affine_forward(forwardPassCache[-1][0], \
                                            paramsToUse[f"W{self.num_layers}"], paramsToUse[f"b{self.num_layers}"]))
        
        return forwardPassCache
    
    def backwardPass(self, dx, forwardPassCache):
        
        backwardPassCache = []
        toIter = list(range(0, self.num_layers))
        toIter.reverse()
        
        for i in toIter:
            if i == self.num_layers-1:
                dOut = dx
                backwardPassCache.append(affine_backward(dOut, forwardPassCache[i][1]))
            else:
                dOut = backwardPassCache[-1][0]
                
                W,b= f"W{i+1}", f"b{i+1}"
                
                if W in self.convParams:
                    if W in self.poolParams:

                        backwardPassCache.append(conv_relu_pool_backward(dOut, forwardPassCache[i][1]))
                    else:

                        backwardPassCache.append(conv_relu_backward(dOut, forwardPassCache[i][1]))
                else:
                    if self.use_dropout:
                        dOut = dropout_backward(dOut, forwardPassCache[i][1][-1])
                        forwardPassCache[i] = (forwardPassCache[i][0], forwardPassCache[i][1][:-1])
                    
                    backwardPassCache.append(affine_relu_backward(dOut, forwardPassCache[i][1]))
                    
        return backwardPassCache
    
    def loss(self, X, y=None, withSnapshot=False):
        
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        
        ## FORWARD PASS
        if withSnapshot:
            scores = None
            for param in self.snapshots:
                forwardPassCache = self.forwardPass(X, mode, snapshot=param)
                if scores is not None:
                    scores += forwardPassCache[-1][0]
                else:
                    scores = forwardPassCache[-1][0]
                    
            scores = scores / len(self.snapshots)
        else:
            forwardPassCache = self.forwardPass(X, mode, snapshot=None)
            scores = forwardPassCache[-1][0]

        if mode == 'test' or withSnapshot:
            return scores
        
        ## LOSS
        loss, dx = softmax_loss(scores, y)
        L2_regularization = sum([np.sum(np.square(self.params[f"W{i+1}"])) for i in range(0, self.num_layers)])
        loss = loss + (self.reg / 2) * L2_regularization

        ## BACKWARD PASS        
        backwardPassCache = self.backwardPass(dx, forwardPassCache)
        backwardPassCache.reverse()
        
        ## GRADIENTS
        grads = {k:v for i in range(0, self.num_layers) \
                 for k,v in [\
                             [f"W{i+1}", backwardPassCache[i][1] + self.reg*self.params[f"W{i+1}"]],\
                             [f"b{i+1}", backwardPassCache[i][2]]\
                            ]}

        return loss, grads