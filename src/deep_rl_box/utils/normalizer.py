"""Components for normalize tensor."""
import tensorflow as tf
import numpy as np


class TensorFlowRunningMeanStd:
    """For RND networks"""

    def __init__(self, shape=()):
        self.mean = tf.Variable(tf.zeros(shape, dtype=tf.float32), trainable=False)
        self.var = tf.Variable(tf.ones(shape, dtype=tf.float32), trainable=False)
        self.count = tf.Variable(0, trainable=False)

        self.deltas = []
        self.min_size = 10

    def update(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        batch_mean = tf.reduce_mean(x, axis=0)
        batch_var = tf.math.reduce_variance(x, axis=0)

        # update count and moments
        n = x.shape[0]
        new_count = self.count + n
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * tf.cast(n, tf.float32) / tf.cast(new_count, tf.float32)
        m_a = self.var * tf.cast(self.count, tf.float32)
        m_b = batch_var * tf.cast(n, tf.float32)
        M2 = m_a + m_b + tf.square(delta) * tf.cast(self.count * n, tf.float32) / tf.cast(new_count, tf.float32)
        new_var = M2 / tf.cast(new_count, tf.float32)

        self.mean.assign(new_mean)
        self.var.assign(new_var)
        self.count.assign(new_count)

    def update_single(self, x):
        self.deltas.append(x)

        if len(self.deltas) >= self.min_size:
            batched_x = tf.concat(self.deltas, axis=0)
            self.update(batched_x)
            self.deltas.clear()

    def normalize(self, x) -> tf.Tensor:
        return (tf.convert_to_tensor(x, dtype=tf.float32) - self.mean) / tf.sqrt(self.var + 1e-8)


class RunningMeanStd:
    """For RND networks"""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = 0

        self.deltas = []
        self.min_size = 10

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)

        # update count and moments
        n = x.shape[0]
        self.count += n
        delta = batch_mean - self.mean
        self.mean += delta * n / self.count
        m_a = self.var * (self.count - n)
        m_b = batch_var * n
        M2 = m_a + m_b + np.square(delta) * (self.count - n) * n / self.count
        self.var = M2 / self.count

    def update_single(self, x):
        self.deltas.append(x)

        if len(self.deltas) >= self.min_size:
            batched_x = np.stack(self.deltas, axis=0)
            self.update(batched_x)

            del self.deltas[:]

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
