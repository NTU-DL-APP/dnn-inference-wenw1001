import numpy as np
import json

# === Activation functions ===
def relu(x):
    # TODO: Implement the Rectified Linear Unit
    return np.maximum(0, x)

def softmax(x):
    # TODO: Implement the SoftMax function
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 防止 overflow
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b


# === Conv2D ===
def conv2d(x, W, b, stride=1, padding=0):
    """
    x: Input image, shape (N, H, W, C)
    W: Filters, shape (KH, KW, C, F)
    b: Biases, shape (F,)
    stride: int
    padding: int
    Return: Output after convolution, shape (N, H_out, W_out, F)
    """
    N, H, W_, C = x.shape
    KH, KW, _, F = W.shape

    # Zero padding
    x_padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')

    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W_ + 2 * padding - KW) // stride + 1
    out = np.zeros((N, H_out, W_out, F))

    for n in range(N):
        for h in range(H_out):
            for w in range(W_out):
                for f in range(F):
                    h_start = h * stride
                    h_end = h_start + KH
                    w_start = w * stride
                    w_end = w_start + KW
                    region = x_padded[n, h_start:h_end, w_start:w_end, :]  # (KH, KW, C)
                    out[n, h, w, f] = np.sum(region * W[:, :, :, f]) + b[f]

    return out
    
# === MaxPooling2D ===
def maxpool2d(x, pool_size=2, stride=2):
    """
    x: Input image, shape (N, H, W, C)
    Return: Downsampled output
    """
    N, H, W, C = x.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    out = np.zeros((N, H_out, W_out, C))

    for n in range(N):
        for h in range(H_out):
            for w in range(W_out):
                for c in range(C):
                    h_start = h * stride
                    h_end = h_start + pool_size
                    w_start = w * stride
                    w_end = w_start + pool_size
                    region = x[n, h_start:h_end, w_start:w_end, c]
                    out[n, h, w, c] = np.max(region)

    return out

# === Dropout ===
def dropout(x, rate=0.5, training=True):
    """
    x: Input tensor
    rate: Dropout rate (e.g., 0.5 means drop 50%)
    training: Apply dropout only during training
    """
    if not training or rate == 0.0:
        return x
    mask = (np.random.rand(*x.shape) >= rate).astype(x.dtype)
    return (x * mask) / (1.0 - rate)

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
# def nn_forward_h5(model_arch, weights, data):
#     x = data
#     for layer in model_arch:
#         lname = layer['name']
#         ltype = layer['type']
#         cfg = layer['config']
#         wnames = layer['weights']

#         if ltype == "Flatten":
#             x = flatten(x)
#         elif ltype == "Dense":
#             W = weights[wnames[0]]
#             b = weights[wnames[1]]
#             x = dense(x, W, b)
#             if cfg.get("activation") == "relu":
#                 x = relu(x)
#             elif cfg.get("activation") == "softmax":
#                 x = softmax(x)

#     return x

def nn_forward_h5(model_arch, weights, data, training=False):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Conv2D":
            W = weights[wnames[0]]  # (KH, KW, C, F)
            b = weights[wnames[1]]  # (F,)
            stride = cfg.get("strides", (1, 1))[0]
            padding = cfg.get("padding", "valid")
            pad = 0
            if padding == "same":
                pad = W.shape[0] // 2
            x = conv2d(x, W, b, stride=stride, padding=pad)

        elif ltype == "MaxPooling2D":
            pool_size = cfg.get("pool_size", (2, 2))[0]
            stride = cfg.get("strides", pool_size)
            x = maxpool2d(x, pool_size=pool_size, stride=stride)

        elif ltype == "Dropout":
            rate = cfg.get("rate", 0.5)
            x = dropout(x, rate=rate, training=training)

        elif ltype == "Flatten":
            x = flatten(x)

        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            activation = cfg.get("activation")
            if activation == "relu":
                x = relu(x)
            elif activation == "softmax":
                x = softmax(x)

    return x



# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)