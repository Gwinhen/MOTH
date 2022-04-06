# coding: utf-8

import tensorflow as tf
import numpy as np


class LinfPGDAttack:
    def __init__(self, x_input, y_input, logits, epsilon, k, a, clip_max,
                 random_start, loss_func):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.x_input = x_input
        self.y_input = y_input
        self.logits  = logits
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.clip_max = clip_max
        self.rand = random_start

        if loss_func == 'xent':
            loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                                    labels=self.y_input, logits=self.logits))
        elif loss_func == 'cw':
            label_mask = tf.one_hot(self.y_input,
                                    10,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * self.logits, axis=1)
            wrong_logit = tf.reduce_max((1-label_mask) * self.logits
                                            - 1e4*label_mask, axis=1)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')

        self.grad = tf.gradients(loss, self.x_input)[0]

    def perturb(self, x_nat, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.clip(x, 0, self.clip_max) # ensure valid pixel range
        else:
            x = np.copy(x_nat)

        for i in range(self.k):
            grad = sess.run(self.grad, feed_dict={self.x_input: x,
                                                  self.y_input: y})

            x += self.a * np.sign(grad)

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
            x = np.clip(x, 0, self.clip_max) # ensure valid pixel range

        return x
