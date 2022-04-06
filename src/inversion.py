# coding: utf-8

import keras
import numpy as np
import sys

from keras import backend as K
from keras import optimizers

class Trigger:
    def __init__(self,
                 model,             # subject model
                 img_rows=32,       # input height
                 img_cols=32,       # input width
                 img_channels=3,    # number of input channels
                 num_classes=10,    # number of classes of subject model
                 steps=1000,        # number of steps for trigger inversion
                 batch_size=32,     # batch size in trigger inversion
                 asr_bound=0.9,     # threshold for attack success rate
                 clip_max=255.0     # maximum pixel value
        ):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.steps = steps
        self.batch_size = batch_size
        self.asr_bound = asr_bound
        self.clip_max = clip_max

        self.shape = (self.img_rows, self.img_cols, self.img_channels)

        # initialize weight on trigger size loss
        self.cost_var = K.variable(0)

        # initialize mask variables
        self.mask_var = K.variable(np.zeros((self.img_rows, self.img_cols, 1)))
        self.mask = K.repeat_elements(
                        (K.tanh(self.mask_var) / (2 - K.epsilon()) + 0.5),
                        rep=self.img_channels,
                        axis=2
                    )

        # initialize pattern variables
        self.pattern_var = K.variable(np.zeros(self.shape))
        self.pattern = (K.tanh(self.pattern_var) / (2 - K.epsilon()) + 0.5)\
                            * self.clip_max

        # input and output placeholders
        input_ph  = K.placeholder(model.input_shape)
        output_ph = K.placeholder(model.output_shape)

        # stamp trigger pattern
        input_adv = (1 - self.mask) * input_ph + self.mask * self.pattern
        output_adv = model(input_adv)

        accuracy = keras.metrics.categorical_accuracy(output_adv, output_ph)

        loss_ce  = keras.losses.categorical_crossentropy(output_adv, output_ph)
        loss_reg = K.sum(K.abs(self.mask)) / self.img_channels

        # total loss
        loss = loss_ce + loss_reg * self.cost_var

        self.optimizer = optimizers.Adam(lr=0.1, beta_1=0.5, beta_2=0.9)
        # parameters to optimize
        updates = self.optimizer.get_updates(
                        params=[self.pattern_var, self.mask_var],
                        loss=loss
                  )
        self.train = K.function(
                        [input_ph, output_ph],
                        [loss_ce, loss_reg, loss, accuracy],
                        updates=updates
                     )

    def generate(self, pair, x_set, y_set, attack_size=100, steps=1000,
                 init_cost=1e-3, init_m=None, init_p=None):
        source, target = pair

        # update hyper-parameters
        self.steps = steps
        self.batch_size = np.minimum(self.batch_size, attack_size)

        # store best results
        mask_best    = np.zeros(self.shape)
        pattern_best = np.zeros(self.shape)
        reg_best     = float('inf')

        # hyper-parameters to dynamically adjust loss weight
        patience = 10
        cost_up_counter   = 0
        cost_down_counter = 0
        cost = init_cost
        K.set_value(self.cost_var, cost)

        # initialize mask and pattern
        if init_m is None:
            init_mask = np.random.random((self.img_rows, self.img_cols))
        else:
            init_mask = init_m

        if init_p is None:
            init_pattern = np.random.random(self.shape) * self.clip_max
        else:
            init_pattern = init_p

        init_mask    = np.clip(init_mask, 0.0, 1.0)
        init_mask    = np.expand_dims(init_mask, axis=2)
        init_mask    = np.arctanh((init_mask - 0.5) * (2 - K.epsilon()))
        init_pattern = np.clip(init_pattern, 0.0, self.clip_max)
        init_pattern = np.arctanh((init_pattern / self.clip_max - 0.5)\
                                        * (2 - K.epsilon()))

        # update mask and pattern variables with init values
        K.set_value(self.mask_var,    init_mask)
        K.set_value(self.pattern_var, init_pattern)

        # reset optimizer states
        K.set_value(self.optimizer.iterations, 0)
        for w in self.optimizer.weights:
            K.set_value(w, np.zeros(K.int_shape(w)))

        # select inputs for label-specific or universal attack
        if source < self.num_classes:
            indices = np.where(np.argmax(y_set, axis=1) == source)[0]
        else:
            indices = np.where(np.argmax(y_set, axis=1) == target)[0]
            if indices.shape[0] != y_set.shape[0]:
                indices = np.where(np.argmax(y_set, axis=1) != target)[0]

            # record loss change
            loss_start = np.zeros(x_set.shape[0])
            loss_end   = np.zeros(x_set.shape[0])

        # choose a subset of samples for trigger inversion
        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices]
        y_set = np.zeros_like(y_set[indices])
        y_set[:, target] = 1

        # avoid having the number of inputs smaller than batch size
        self.batch_size = np.minimum(self.batch_size, x_set.shape[0])

        # record samples' indices during suffling
        index_base = np.arange(x_set.shape[0])

        # start generation
        for step in range(self.steps):
            # shuffle training samples
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]
            index_base = index_base[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(int(np.ceil(x_set.shape[0] / self.batch_size))):
                # get a batch of data
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]

                (loss_ce_value, loss_reg_value, loss_value, acc_value)\
                    = self.train([x_batch, y_batch])

                # record loss and accuracy
                loss_ce_list.extend( list(loss_ce_value.flatten()))
                loss_reg_list.extend(list(loss_reg_value.flatten()))
                loss_list.extend(    list(loss_value.flatten()))
                acc_list.extend(     list(acc_value.flatten()))

            # record the initial loss value
            if source == self.num_classes\
                    and step == 0\
                    and len(loss_ce_list) == attack_size:
                loss_start[index_base] = loss_ce_list

            # calculate average loss and accuracy
            avg_loss_ce  = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss     = np.mean(loss_list)
            avg_acc      = np.mean(acc_list)

            # record the best mask and pattern
            if avg_acc >= self.asr_bound and avg_loss_reg < reg_best:
                mask_best    = K.eval(self.mask)
                pattern_best = K.eval(self.pattern)
                reg_best     = avg_loss_reg

                # add samll perturbations to mask and pattern
                # to avoid stucking in local minima
                epsilon = 0.01
                init_mask    = mask_best[..., 0]
                init_mask    = init_mask\
                                    + np.random.uniform(-epsilon,
                                                        epsilon,
                                                        init_mask.shape)
                init_mask    = np.clip(init_mask, 0.0, 1.0)
                init_mask    = np.expand_dims(init_mask, axis=2)
                init_mask    = np.arctanh((init_mask - 0.5) * (2 - K.epsilon()))
                init_pattern = pattern_best + self.clip_max\
                                    * np.random.uniform(-epsilon,
                                                        epsilon,
                                                        init_pattern.shape)
                init_pattern = np.clip(init_pattern, 0.0, self.clip_max)
                init_pattern = np.arctanh((init_pattern / self.clip_max - 0.5)\
                                                * (2 - K.epsilon()))

                K.set_value(self.mask_var,    init_mask)
                K.set_value(self.pattern_var, init_pattern)

                # record the final loss value when the best trigger is saved
                if source == self.num_classes\
                        and len(loss_ce_list) == attack_size:
                    loss_end[index_base] = loss_ce_list

            # helper variables for adjusting loss weight
            if avg_acc >= self.asr_bound:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            # adjust loss weight
            if cost_up_counter >= patience:
                cost_up_counter = 0
                if cost == 0:
                    cost = 1e-3
                else:
                    cost *= 1.5
                K.set_value(self.cost_var, cost)
            elif cost_down_counter >= patience:
                cost_down_counter = 0
                cost /= 1.5 ** 1.5
                K.set_value(self.cost_var, cost)

            # periodically print inversion results
            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                    .format(step, avg_acc, avg_loss)
                                 + 'ce: {:.2f}, reg: {:.2f}, reg_best: {:.2f}  '\
                                    .format(avg_loss_ce, avg_loss_reg, reg_best))
                sys.stdout.flush()

        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rmask norm of pair {:d}-{:d}: {:.2f}\n'\
                            .format(source, target, np.sum(np.abs(mask_best))))
        sys.stdout.flush()

        # compute loss difference
        if source == self.num_classes and len(loss_ce_list) == attack_size:
            indices = np.where(loss_start == 0)[0]
            loss_start[indices] = 1
            loss_monitor = (loss_start - loss_end) / loss_start
            loss_monitor[indices] = 0
        else:
            loss_monitor = np.zeros(x_set.shape[0])

        return mask_best, pattern_best, loss_monitor


class TriggerCombo:
    def __init__(self,
                 model,             # subject model
                 img_rows=32,       # input height
                 img_cols=32,       # input width
                 img_channels=3,    # number of input channels
                 steps=1000,        # number of steps for trigger inversion
                 batch_size=32,     # batch size in trigger inversion
                 asr_bound=0.9,     # threshold for attack success rate
                 clip_max=255.0     # maximum pixel value
        ):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.steps = steps
        self.batch_size = batch_size
        self.asr_bound = asr_bound
        self.clip_max = clip_max

        self.shape = (2, self.img_rows, self.img_cols, self.img_channels)

        # initialize weight on trigger size loss
        self.cost_var = K.variable([0] * 2)

        # initialize mask variables
        self.mask_var = K.variable(np.zeros((2, self.img_rows, self.img_cols, 1)))
        self.mask = K.repeat_elements(
                        (K.tanh(self.mask_var) / (2 - K.epsilon()) + 0.5),
                        rep=img_channels,
                        axis=3
                    )

        # initialize pattern variables
        self.pattern_var = K.variable(np.zeros(self.shape))
        self.pattern = (K.tanh(self.pattern_var) / (2 - K.epsilon()) + 0.5)\
                            * self.clip_max

        # input and output placeholders
        input_ph  = K.placeholder(model.input_shape)
        output_ph = K.placeholder(model.output_shape)
        combo_ph  = K.placeholder([None])

        # stamp trigger patterns for different pair directions
        input_adv = combo_ph[:, None, None, None]\
                        * ((1 - self.mask[0]) * input_ph\
                            + self.mask[0] * self.pattern[0])\
                    + (1 - combo_ph[:, None, None, None])\
                        * ((1 - self.mask[1]) * input_ph\
                            + self.mask[1] * self.pattern[1])
        output_adv = model(input_adv)
        output_tru = output_ph

        # attack accuracy
        accuracy = keras.metrics.categorical_accuracy(output_adv, output_tru)
        accuracy = [K.sum(combo_ph * accuracy) / K.sum(combo_ph),\
                    K.sum((1 - combo_ph) * accuracy) / K.sum(1 - combo_ph)]

        # cross entropy loss
        loss_ce = keras.losses.categorical_crossentropy(output_adv, output_tru)
        loss_ce = [K.sum(combo_ph * loss_ce), K.sum((1 - combo_ph) * loss_ce)]

        # trigger size loss
        loss_reg = K.sum(K.abs(self.mask), axis=(1, 2, 3)) / img_channels

        # total loss
        loss = loss_ce + loss_reg * self.cost_var

        self.optimizer = optimizers.Adam(lr=0.1, beta_1=0.5, beta_2=0.9)
        # parameters to optimize
        updates = self.optimizer.get_updates(
                        params=[self.pattern_var, self.mask_var],
                        loss=loss
                  )
        self.train = K.function(
                        [input_ph, output_ph, combo_ph],
                        [loss_ce, loss_reg, loss, accuracy],
                        updates=updates
                     )

    def generate(self, pair, x_set, y_set, m_set, attack_size=100, steps=1000,
                 init_cost=1e-3, init_m=None, init_p=None):
        source, target = pair

        # update hyper-parameters
        self.steps = steps
        self.batch_size = np.minimum(self.batch_size, attack_size)

        # store best results
        mask_best    = np.zeros(self.shape)
        pattern_best = np.zeros(self.shape)
        reg_best     = [float('inf')] * 2

        # hyper-parameters to dynamically adjust loss weight
        patience = 10
        cost_up_counter   = [0] * 2
        cost_down_counter = [0] * 2
        cost = [init_cost] * 2
        K.set_value(self.cost_var, cost)

        # initialize mask and pattern
        if init_m is None:
            init_mask = np.random.random((2, self.img_rows, self.img_cols))
        else:
            init_mask = init_m

        if init_p is None:
            init_pattern = np.random.random(self.shape) * self.clip_max
        else:
            init_pattern = init_p

        init_mask    = np.clip(init_mask, 0.0, 1.0)
        init_mask    = np.expand_dims(init_mask, axis=3)
        init_mask    = np.arctanh((init_mask - 0.5) * (2 - K.epsilon()))
        init_pattern = np.clip(init_pattern, 0.0, self.clip_max)
        init_pattern = np.arctanh((init_pattern / self.clip_max - 0.5)\
                                        * (2 - K.epsilon()))

        # update mask and pattern variables with init values
        K.set_value(self.mask_var,    init_mask)
        K.set_value(self.pattern_var, init_pattern)

        # reset optimizer states
        K.set_value(self.optimizer.iterations, 0)
        for w in self.optimizer.weights:
            K.set_value(w, np.zeros(K.int_shape(w)))

        # start inversion
        for step in range(self.steps):
            # shuffle training samples
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]
            m_set = m_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                # get a batch of data
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]
                m_batch = m_set[idx*self.batch_size : (idx+1)*self.batch_size]

                (loss_ce_value, loss_reg_value, loss_value, acc_value)\
                    = self.train([x_batch, y_batch, m_batch])

                # record loss and accuracy
                loss_ce_list.append( list(loss_ce_value))
                loss_reg_list.append(list(loss_reg_value))
                loss_list.append(    list(loss_value))
                acc_list.append(     list(acc_value))

            # calculate average loss and accuracy
            avg_loss_ce  = np.mean(loss_ce_list,  axis=0)
            avg_loss_reg = np.mean(loss_reg_list, axis=0)
            avg_loss     = np.mean(loss_list,     axis=0)
            avg_acc      = np.mean(acc_list,      axis=0)

            # update results for two directions of a pair
            for cb in range(2):
                # record the best mask and pattern
                if avg_acc[cb] >= self.asr_bound\
                        and avg_loss_reg[cb] < reg_best[cb]:
                    mask_best_local    = K.eval(self.mask)
                    pattern_best_local = K.eval(self.pattern)
                    mask_best[cb]      = mask_best_local[cb].copy()
                    pattern_best[cb]   = pattern_best_local[cb].copy()
                    reg_best[cb]       = avg_loss_reg[cb]

                    # add samll perturbations to mask and pattern
                    # to avoid stucking in local minima
                    epsilon = 0.01
                    init_mask    = mask_best_local[cb, ..., 0].copy()
                    init_mask    = init_mask\
                                        + np.random.uniform(-epsilon,
                                                            epsilon,
                                                            init_mask.shape)
                    init_mask    = np.expand_dims(init_mask, axis=2)
                    init_pattern = pattern_best_local[cb].copy()
                    init_pattern = init_pattern + self.clip_max\
                                        * np.random.uniform(-epsilon,
                                                            epsilon,
                                                            init_pattern.shape)

                    # stack mask and pattern in the corresponding direction
                    otr_idx = (cb + 1) % 2
                    if cb == 0:
                        init_mask    = np.stack(
                                        [init_mask,
                                            mask_best_local[otr_idx][..., :1]],
                                        axis=0
                                       )
                        init_pattern = np.stack(
                                        [init_pattern,
                                            pattern_best_local[otr_idx]],
                                        axis=0
                                       )
                    else:
                        init_mask    = np.stack(
                                        [mask_best_local[otr_idx][..., :1],
                                            init_mask],
                                        axis=0
                                       )
                        init_pattern = np.stack(
                                        [pattern_best_local[otr_idx],
                                            init_pattern],
                                        axis=0
                                       )
                    init_mask    = np.clip(init_mask, 0.0, 1.0)
                    init_pattern = np.clip(init_pattern, 0.0, self.clip_max)

                    init_mask    = np.arctanh(
                                        (init_mask - 0.5)\
                                                * (2 - K.epsilon())
                                   )
                    init_pattern = np.arctanh(
                                        (init_pattern / self.clip_max - 0.5)\
                                                * (2 - K.epsilon())
                                   )

                    K.set_value(self.mask_var,    init_mask)
                    K.set_value(self.pattern_var, init_pattern)

                # helper variables for adjusting loss weight
                if avg_acc[cb] >= self.asr_bound:
                    cost_up_counter[cb] += 1
                    cost_down_counter[cb] = 0
                else:
                    cost_up_counter[cb] = 0
                    cost_down_counter[cb] += 1

                # adjust loss weight
                if cost_up_counter[cb] >= patience:
                    cost_up_counter[cb] = 0
                    if cost[cb] == 0:
                        cost[cb] = 1e-3
                    else:
                        cost[cb] *= 1.5
                    K.set_value(self.cost_var, cost)
                elif cost_down_counter[cb] >= patience:
                    cost_down_counter[cb] = 0
                    cost[cb] /= 1.5 ** 1.5
                    K.set_value(self.cost_var, cost)

            # periodically print inversion results
            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: ({:.2f}, {:.2f}), '\
                                    .format(step, avg_acc[0], avg_acc[1])
                                 + 'loss: ({:.2f}, {:.2f}), '\
                                    .format(avg_loss[0], avg_loss[1])
                                 + 'ce: ({:.2f}, {:.2f}), '\
                                    .format(avg_loss_ce[0], avg_loss_ce[1])
                                 + 'reg: ({:.2f}, {:.2f}), '\
                                    .format(avg_loss_reg[0], avg_loss_reg[1])
                                 + 'reg_best: ({:.2f}, {:.2f})  '\
                                    .format(reg_best[0], reg_best[1]))
                sys.stdout.flush()

        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rmask norm of pair {:d}-{:d}: {:.2f}\n'\
                            .format(source, target, np.sum(np.abs(mask_best[0]))))
        sys.stdout.write('\rmask norm of pair {:d}-{:d}: {:.2f}\n'\
                            .format(target, source, np.sum(np.abs(mask_best[1]))))
        sys.stdout.flush()

        return mask_best, pattern_best
