import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def ponyfy(I, r):
    bsize, a, b, c = tf.get_shape(I).as_list()
    X = tf.reshape(I, (bsize, a/r, b/r, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, r, r
    X = tf.split(1, a, X)  # a, [bsize, b, r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
    X = tf.split(1, b, X)  # b, [bsize, a*r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def ponynet(X, r, color=False):
    if color:
        Xc = tf.split(3, 3, X)
        X = tf.concat([ponyfy(x, r) for x in Xc])
    else:
        X = ponyfy(X, r)
    return X

if __name__ == "__main__":
    with tf.Session() as sess:
        x = np.arange(2*16*16).reshape(2, 8, 8, 4)
        X = tf.Variable("float32", x, name="X")
        Y = ponynet(X, 2)
        y = Y.eval()
    plt.imshow(y[0, :, :, 0], interpolation="none")
    plt.show()
