import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """

    N = x.shape[0]
    M = b.shape[0]

    x_reshape = x.reshape(N, -1)
    x_bias = np.concatenate((x_reshape, np.ones((N,1))), axis=1)
    
    w_bias = np.concatenate((w, b.reshape(M,1).T), axis=0)

    out = np.dot(x_bias, w_bias)
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
        
    dL/dw = dL/dout * dout/dw
    
    """
    x, w, b = cache
    N = x.shape[0]
    M = b.shape[0]
    db = np.matmul(dout.T, np.ones((N,1))).reshape(M,)
    dw = np.matmul(x.reshape(N, -1).T, dout)
    dx = np.matmul(dout, w.T).reshape(x.shape)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(x,0)
    
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache
    
    dx = np.multiply(dout, (x > 0).astype(int))
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        mean_centered = np.add(x, -1*x.mean(axis=0))
        variance = (1/N) * (mean_centered**2).sum(axis=0)
        normalized_batch = (x - x.mean(axis=0)) / np.sqrt(x.var(axis=0) + eps)
        
        out = gamma*normalized_batch + beta
        cache = (x, normalized_batch, x.mean(axis=0), variance, gamma, eps)
        
        running_mean = (momentum) * running_mean + (1-momentum)*x.mean(axis=0)
        running_var = (momentum) * running_var + (1-momentum)*variance
        
    elif mode == 'test':
        scale = gamma / (np.sqrt(running_var  + eps))
        out = x * scale + (beta - running_mean * scale)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    x, normalized_batch, mean, variance, gamma, eps = cache
    N,D = x.shape
    d_norm = dout * gamma
    
    dvar_1 = (d_norm*(x - mean)).sum(axis=0) #2b
    d_var = -0.5 * dvar_1 * ( (variance+eps)**-1.5 ) #3b
    
    dmean_1 = (d_norm * ( -1 / np.sqrt(variance+eps) ) ).sum(axis=0)
    dmean_2 = d_var * (-1 / N) * ( ( 2 * (x - mean) ) ).sum(axis=0)
    d_mean = dmean_1 + dmean_2
   
    dx_1 = d_norm / (np.sqrt(variance+eps))
    dx_2 = d_var * 2 * (x - mean) / N
    dx_3 = d_mean / N
    
    dx = (dx_1 + dx_2) + dx_3
    dbeta = dout.sum(axis=0)   
    dgamma = (dout*normalized_batch).sum(axis=0)
                                                                                     
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    x, normalized_batch, mean, variance, gamma, eps = cache
    N,D = x.shape
    
    d_norm = dout * gamma    
    dbeta = dout.sum(axis=0)   
    dgamma = (dout*normalized_batch).sum(axis=0)
    
    dx_numer = N*d_norm - d_norm.sum(axis=0) - normalized_batch * (d_norm*normalized_batch).sum(axis=0)
    dx_denom = N*np.sqrt((variance+eps))
    
    dx = dx_numer / dx_denom
                                                                                     
    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = (np.random.rand(*x.shape)) > p

    if mode == 'train':
        out = x * mask
    elif mode == 'test':
        out = x 

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout*mask
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    
    Hout = int(1 + (H + 2*pad - HH) / stride)
    Wout = int(1 + (W + 2*pad - WW) / stride)
    out = np.empty((N,F,Hout,Wout))
    
    padded_x = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    for h in range(Hout):
        for wi in range(Wout):
            toConvolute = padded_x[:, :, h*stride : h*stride+HH, wi*stride : wi*stride+WW]
            for f in range(F):
                out[:, f, h, wi] = np.sum(toConvolute*w[f], axis=(1,2,3)) + b[f]
    
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x,w,b,conv_param = cache
    
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    
    Hout = int(1 + (H + 2*pad - HH) / stride)
    Wout = int(1 + (W + 2*pad - WW) / stride)
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    
    padded_dx = np.pad(dx, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    padded_x = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    for h in range(Hout):
        for wi in range(Wout):
            for n in range(N):
                padded_dx[n,:,h*stride : h*stride+HH, wi*stride : wi*stride+WW] += \
                    (w*dout[n, :, h, wi][:,None,None,None]).sum(axis=0)
            for f in range(F):
                dw[f,:,:,:] += (padded_x[:, :, h*stride : h*stride+HH, wi*stride : wi*stride+WW]*dout[:,f,h,wi][:,None,None,None]).sum(axis=0)
                
    dx = padded_dx[:,:,pad:-pad, pad:-pad]
    db = dout.sum(axis=(0, 2, 3))
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    N,C,H,W = x.shape
    poolH = pool_param['pool_height']
    poolW = pool_param['pool_width']
    stride = pool_param['stride']
    
    Hout = int(1 + (H-poolH) / stride)
    Wout = int(1 + (W-poolW)/stride)
    out = np.empty((N,C,Hout,Wout))
    for h in range(Hout):
        for w in range(Wout):
            out[:,:,h,w] = x[:,:, h*stride:h*stride+poolH, w*stride:w*stride+poolW].max(axis=(2,3))
    
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    _,_,H,W = x.shape
    poolH = pool_param['pool_height']
    poolW = pool_param['pool_width']
    stride = pool_param['stride']
    
    Hout = int(1 + (H-poolH) / stride)
    Wout = int(1 + (W-poolW)/stride)
    
    dx = np.zeros(x.shape)
    
    for h in range(Hout):
        for w in range(Wout):
            x_mask = x[:,:, h*stride:h*stride+poolH, w*stride:w*stride+poolW]
            Xmask = (x_mask == x_mask.max(axis=(2,3))[:,:,None,None]).astype(np.float32)
            
            dx[:,:, h*stride:h*stride+poolH, w*stride:w*stride+poolW] += dout[:,:,h,w][:,:,None,None]*Xmask 
        
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (C,) giving running mean of features
      - running_var Array of shape (C,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N,C,H,W = x.shape
    toBatch = np.transpose(x, (1,0,2,3)).reshape(C, N*H*W).T
    out, cache = batchnorm_forward(toBatch, gamma, beta, bn_param)
    out = np.transpose(out.T.reshape(C, N, H, W), (1, 0, 2, 3))

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    N,C,H,W = dout.shape
    toBatch = np.transpose(dout, (1,0,2,3)).reshape(C, N*H*W).T
    dx, dgamma, dbeta = batchnorm_backward(toBatch, cache)
    dx = np.transpose(dx.T.reshape(C,N,H,W), (1,0,2,3))

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
