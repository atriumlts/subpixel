import tensorflow as tf


def ponyfy(I, r):
    bsize, a, b, c = tf.get_shape(I).as_list()
    X = tf.reshape(I, (bsize, a/r, b/r, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, r, r
    X = tf.split(1, a, X)  # a, [bsize, b, r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
    X = tf.split(1, b, X)  # b, [bsize, a*r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def ponynet(X, r, batch_size):
    Xc = tf.split(3, 3, X)
    X = tf.concat([ponyfy(x, r) for x in Xc])
    return X
