# coding: utf-8

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import argparse
import keras
import numpy as np
import os
import sys
import time

from keras import backend as K
from keras import optimizers
from keras.layers import Activation
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from network import cnn, nin, resnet, vgg19
from dataset import cifar10_data, svhn_data, lisa_data, gtsrb_data
from pgd_attack import LinfPGDAttack
from inversion import Trigger, TriggerCombo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"

if('tensorflow' == K.backend()):
    import tensorflow as tf

    try:
        from tensorflow.python.util import module_wrapper as deprecation
    except ImportError:
        from tensorflow.python.util import deprecation_wrapper as deprecation
    deprecation._PER_MODULE_WARNING_LIMIT = 0
    tf.logging.set_verbosity(tf.logging.ERROR)


def cifar_norm(x):
    mean = [125.307, 122.950, 113.865]
    std  = [62.9932, 62.0887, 66.7048]

    r, g, b = tf.split(x, 3, 3)
    x_rgb = tf.concat([
                (r - mean[0]) / std[0],
                (g - mean[1]) / std[1],
                (b - mean[2]) / std[2],
            ], 3)
    return x_rgb


def get_dataset():
    if args.dataset == 'cifar10':
        x_train, y_train, x_val, y_val, x_test, y_test = cifar10_data()
    elif args.dataset == 'svhn':
        x_train, y_train, x_val, y_val, x_test, y_test = svhn_data()
    elif args.dataset == 'lisa':
        x_train, y_train, x_val, y_val, x_test, y_test = lisa_data()
    elif args.dataset == 'gtsrb':
        x_train, y_train, x_val, y_val, x_test, y_test = gtsrb_data()

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_model(use_logits=False):
    input_norm = cifar_norm if args.dataset == 'cifar10' else None

    if args.model == 'vgg19':
        model = vgg19(
                    num_classes=num_classes,
                    input_norm=input_norm,
                    use_logits=use_logits
                )
    elif args.model == 'nin':
        model = nin(
                    num_classes=num_classes,
                    input_norm=input_norm,
                    use_logits=use_logits
                )
    elif 'resnet' in args.model:
        num_layers = int(args.model[6:])
        model = resnet(
                    num_layers,
                    num_classes=num_classes,
                    input_norm=input_norm,
                    use_logits=use_logits
                )
    elif args.model == 'cnn':
        model = cnn(
                    num_classes=num_classes,
                    use_logits=use_logits
                )

    return model


def moth():
    # assisting variables/parameters
    trigger_steps = 500
    warmup_steps  = 1
    cost   = 1e-3
    count  = np.zeros(2)
    WARMUP = True

    # matrices for recording distance changes
    mat_univ  = np.zeros((num_classes, num_classes)) # warmup distance
    mat_size  = np.zeros((num_classes, num_classes)) # trigger size
    mat_diff  = np.zeros((num_classes, num_classes)) # distance improvement
    mat_count = np.zeros((num_classes, num_classes)) # number of selected pairs

    mask_dict    = {}
    pattern_dict = {}
    with tf.Session() as sess:
        sgd = optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=True)
        cce = keras.losses.CategoricalCrossentropy(from_logits=True)

        # load model with logits layer
        model = get_model(use_logits=True)
        model.load_weights(f'ckpt/{args.dataset}_{args.model}_{args.suffix}.h5')
        model.compile(loss=cce, optimizer=sgd, metrics=['accuracy'])

        # add softmax layer to the model
        soft_layer = Activation('softmax')(model.output)
        soft_model = Model(inputs=model.input, outputs=soft_layer)
        soft_model.compile(loss=cce, optimizer=sgd, metrics=['accuracy'])

        # load data
        x_train, y_train, x_val, y_val, x_test, y_test = get_dataset()

        # choose a subset of training data
        indices = np.random.choice(len(x_train),
                                   int(len(x_train) * args.data_ratio),
                                   replace=False)
        x_train = x_train[indices]
        y_train = y_train[indices]

        # data augmentation
        if args.dataset == 'cifar10':
            datagen = ImageDataGenerator(
                            horizontal_flip=True,
                            width_shift_range=0.125,
                            height_shift_range=0.125,
                            fill_mode='constant',
                            cval=0.0
                      )
        else:
            datagen = ImageDataGenerator()
        datagen.fit(x_train)
        dataflow = datagen.flow(x_train, y_train, batch_size=args.batch_size)

        # a subset for loss calculation during warmup
        l_train = np.argmax(y_train, axis=1)
        l_index = []
        for i in range(num_classes):
            indices = np.where(l_train == i)[0]
            l_index.append(indices)

        # set up trigger generation
        trigger       = Trigger(
                            soft_model,
                            steps=trigger_steps,
                            asr_bound=0.99,
                            img_rows=img_rows,
                            img_cols=img_cols,
                            clip_max=clip_max
                        )
        trigger_combo = TriggerCombo(
                            soft_model,
                            steps=trigger_steps,
                            img_rows=img_rows,
                            img_cols=img_cols,
                            clip_max=clip_max
                        )

        if args.type == 'adv':
            # attack parameters
            if args.dataset == 'cifar10':
                epsilon, k, a = 8, 7, 2
            elif args.dataset in ['svhn', 'gtsrb']:
                epsilon, k, a = 0.03, 8, 0.005
            elif args.dataset == 'lisa':
                epsilon, k, a = 0.1, 8, 0.02

            # set up variables for PGD attack
            x_input = tf.placeholder(tf.float32,
                                     [None, img_rows, img_cols, img_channels])
            y_input = tf.placeholder(tf.float32, [None, num_classes])

            # initialize pgd attack
            attack = LinfPGDAttack(
                        x_input,
                        y_input,
                        model(x_input),
                        epsilon,
                        k,
                        a,
                        clip_max,
                        True,
                        'xent'
                     )

        # hardening iterations
        max_warmup_steps = warmup_steps * num_classes
        steps_per_epoch = int(np.ceil(len(x_train) / args.batch_size))
        max_steps = max_warmup_steps + args.epochs * steps_per_epoch

        source, target = 0, -1

        # start hardening
        print('='*80)
        print('start hardening...')
        time_start = time.time()
        for step in range(max_steps):
            x_batch, y_batch = dataflow.next()

            if args.type == 'nat':
                x_adv = x_batch.copy()
            elif args.type == 'adv':
                x_adv = attack.perturb(x_batch, y_batch, sess)

            # update variables after warmup stage
            if step >= max_warmup_steps:
                if WARMUP:
                    mat_diff /= np.max(mat_diff)
                WARMUP = False
                warmup_steps = 3

            # periodically update corresponding variables in each stage
            if (WARMUP and step % warmup_steps == 0) or\
               (not WARMUP and (step - max_warmup_steps) % warmup_steps == 0):
                if WARMUP:
                    target += 1
                    trigger_steps = 500
                else:
                    if np.random.rand() < 0.3:
                        # randomly select a pair
                        source, target = np.random.choice(
                                            np.arange(num_classes),
                                            2,
                                            replace=False
                                         )
                    else:
                        # select a pair according to distance improvement
                        univ_sum = mat_univ + mat_univ.transpose()
                        diff_sum = mat_diff + mat_diff.transpose()
                        alpha = np.minimum(
                                    0.1 * ((step - max_warmup_steps) / 100),
                                    1
                                )
                        diff_sum = (1 - alpha) * univ_sum + alpha * diff_sum
                        source, target = np.unravel_index(np.argmax(diff_sum),
                                                          diff_sum.shape)
                        print('-'*50)
                        print('fastest pair: {:d}-{:d}, improve: {:.2f}'.format(\
                                source, target, diff_sum[source, target]))

                    trigger_steps = 200

                if source < target:
                    key = f'{source}-{target}'
                else:
                    key = f'{target}-{source}'

                print('-'*50)
                print('selected pair:', key)

                # count the selected pair
                if not WARMUP:
                    mat_count[source, target] += 1
                    mat_count[target, source] += 1

                # use existing previous mask and pattern
                if key in mask_dict:
                    init_mask    = mask_dict[key].copy()
                    init_pattern = pattern_dict[key].copy()
                else:
                    init_mask    = None
                    init_pattern = None

                # reset values
                cost = 1e-3
                count[...] = 0
                mask_size_list = []

            if WARMUP:
                # get a few samples from each label
                num_sample = 10
                indices = []
                for i in range(num_classes):
                    if i != target:
                        idx = np.random.choice(l_index[i], num_sample,
                                               replace=False)
                        indices.extend(list(idx))

                # trigger inversion set
                x_set = x_train[indices]
                l_set = l_train[indices]
                y_set = np.zeros((len(x_set), num_classes))
                y_set[:, target] = 1

                # generate universal trigger
                mask, pattern, speed\
                        = trigger.generate(
                                (num_classes, target),
                                x_set,
                                y_set,
                                attack_size=len(indices),
                                steps=trigger_steps,
                                init_cost=cost,
                                init_m=init_mask,
                                init_p=init_pattern
                          )

                trigger_size = [np.sum(np.abs(mask))] * 2

                # choose non-target samples to stamp the generated trigger
                y_batch_arg = np.argmax(y_batch, axis=1)
                indices = np.where(y_batch_arg != target)[0]
                length = int(len(indices) * args.warm_ratio)
                choice = np.random.choice(indices, length, replace=False)

                # stamp trigger
                x_batch_adv = (1 - mask) * x_batch[choice] + mask * pattern
                x_batch_adv = np.clip(x_batch_adv, 0.0, clip_max)

                x_adv[choice] = x_batch_adv.copy()

                # record approximated distance improvement during warmup
                for i in range(num_classes):
                    # mean loss change of samples of each source label
                    if i < target:
                        diff = np.mean(speed[i*num_sample : (i+1)*num_sample])
                    elif i > target:
                        diff = np.mean(speed[(i-1)*num_sample : i*num_sample])

                    if i != target:
                        mat_univ[i, target] = diff

                        # save generated triggers of a pair
                        src, tgt = i, target
                        key = f'{src}-{tgt}' if src < tgt else f'{tgt}-{src}'
                        if key not in mask_dict:
                            mask_dict[key]    = mask.copy()[..., 0]
                            pattern_dict[key] = pattern.copy()
                        else:
                            if src < tgt:
                                mask_dict[key]    = np.stack(
                                                        [mask[..., 0],
                                                            mask_dict[key]],
                                                        axis=0
                                                    )
                                pattern_dict[key] = np.stack(
                                                        [pattern,
                                                            pattern_dict[key]],
                                                        axis=0
                                                    )
                            else:
                                mask_dict[key]    = np.stack(
                                                        [mask_dict[key],
                                                            mask[..., 0]],
                                                        axis=0
                                                    )
                                pattern_dict[key] = np.stack(
                                                        [pattern_dict[key],
                                                            pattern],
                                                        axis=0
                                                    )

                        # initialize distance matrix entries
                        mat_size[i, target] = np.sum(np.abs(mask))
                        mat_diff[i, target] = mat_size[i, target]
            else:
                # get samples from source and target labels
                y_batch_arg = np.argmax(y_batch, axis=1)
                idx_source = np.where(y_batch_arg == source)[0]
                idx_target = np.where(y_batch_arg == target)[0]

                # use a portion of source/target samples
                length = int(min(len(idx_source), len(idx_target)) * args.portion)
                if length > 0:
                    # dynamically adjust parameters
                    if (step - max_warmup_steps) % warmup_steps > 0:
                        if count[0] > 0 or count[1] > 0:
                            trigger_steps = 200
                            cost = 1e-3
                            count[...] = 0
                        else:
                            trigger_steps = 40
                            cost = 2e-1

                    # construct generation set for both directions
                    # source samples with target labels
                    # target samples with source labels
                    x_set = np.concatenate((x_batch[idx_source],
                                            x_batch[idx_target]))
                    y_set = np.zeros((len(x_set), num_classes))
                    y_set[:len(idx_source), target] = 1
                    y_set[len(idx_source):, source] = 1

                    # indicator vector for source/target
                    m_set = np.zeros(len(x_set))
                    m_set[:len(idx_source)] = 1

                    # generate a pair of triggers
                    mask, pattern\
                            = trigger_combo.generate(
                                    (source, target),
                                    x_set,
                                    y_set,
                                    m_set,
                                    attack_size=len(x_set),
                                    steps=trigger_steps,
                                    init_cost=cost,
                                    init_m=init_mask,
                                    init_p=init_pattern
                              )

                    trigger_size = np.sum(np.abs(mask), axis=(1, 2, 3))

                    # operate on two directions
                    for cb in range(2):
                        # choose samples to stamp the generated trigger
                        indices = idx_source if cb == 0 else idx_target
                        choice = np.random.choice(indices, length, replace=False)

                        # stamp trigger
                        x_batch_adv = (1 - mask[cb]) * x_batch[choice]\
                                            + mask[cb] * pattern[cb]
                        x_batch_adv = np.clip(x_batch_adv, 0.0, clip_max)

                        x_adv[choice] = x_batch_adv.copy()

                        # save generated triggers of a pair
                        if init_mask is None:
                            init_mask    = mask.copy()[..., 0]
                            init_pattern = pattern.copy()

                            if key not in mask_dict:
                                mask_dict[key]    = init_mask.copy()
                                pattern_dict[key] = init_pattern.copy()
                        else:
                            if np.sum(mask[cb]) > 0:
                                init_mask[cb]    = mask.copy()[cb, ..., 0]
                                init_pattern[cb] = pattern.copy()[cb]
                                # save large trigger
                                if np.sum(init_mask[cb])\
                                        > np.sum(mask_dict[key][cb]):
                                    mask_dict[key][cb]    = init_mask[cb].copy()
                                    pattern_dict[key][cb] = init_pattern[cb].copy()
                            else:
                                # record failed generation
                                count[cb] += 1

                    mask_size_list.append(
                            list(np.sum(3 * np.abs(init_mask), axis=(1, 2)))
                    )

                # periodically update distance related matrices
                if (step - max_warmup_steps) % warmup_steps == warmup_steps - 1:
                    if len(mask_size_list) <= 0:
                        continue

                    # average trigger size of the current hardening period
                    mask_size_avg = np.mean(mask_size_list, axis=0)
                    if mat_size[source, target] == 0:
                        mat_size[source, target] = mask_size_avg[0]
                        mat_size[target, source] = mask_size_avg[1]
                        mat_diff = mat_size
                        mat_diff[mat_diff == -1] = 0
                    else:
                        # compute distance improvement
                        last_size = mat_size[source, target]
                        mat_diff[source, target]\
                                += (mask_size_avg[0] - last_size) / last_size
                        mat_diff[source, target] /= 2

                        last_size = mat_size[target, source]
                        mat_diff[target, source]\
                                += (mask_size_avg[1] - last_size) / last_size
                        mat_diff[target, source] /= 2

                        # update recorded trigger size
                        mat_size[source, target] = mask_size_avg[0]
                        mat_size[target, source] = mask_size_avg[1]

            x_batch = x_adv

            # train model
            loss, acc = model.train_on_batch(x_batch, y_batch)

            # evaluate and save model
            if (step+1) % 10 == 0:
                time_end = time.time()
                score = model.evaluate(x_val, y_val, verbose=0)

                time_cost = time_end - time_start
                print('*'*120)
                sys.stdout.write('step: {:4}/{:4} - {:.2f}s, '\
                                    .format(step+1, max_steps, time_cost)\
                                 + 'loss: {:.4f}, acc: {:.4f}\t'
                                    .format(loss, acc)\
                                 + 'val loss: {:.4f}, acc: {:.4f}\t'
                                    .format(score[0], score[1])
                                 + 'trigger size: ({:.2f}, {:.2f})\n'
                                    .format(trigger_size[0], trigger_size[1]))
                sys.stdout.flush()
                print('*'*120)

                save_name = f'{args.dataset}_{args.model}_{args.suffix}_moth'
                np.save(f'data/pair_count/{save_name}', mat_count)
                model.save(f'ckpt/{save_name}.h5')

                time_start = time.time()

        np.save(f'data/pair_count/{save_name}', mat_count)
        model.save(f'ckpt/{save_name}.h5')


def test():
    model = get_model()
    model.load_weights(f'ckpt/{args.dataset}_{args.model}_{args.suffix}.h5')
    model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizers.SGD(),
            metrics=['accuracy']
          )

    x_train, y_train, x_val, y_val, x_test, y_test = get_dataset()
    _, acc = model.evaluate(x_test, y_test, verbose=0)

    print('-'*50)
    print(f'acc: {acc:.4f}')


def measure():
    with tf.Session() as sess:
        # load model
        model = get_model()
        model.load_weights(f'ckpt/{args.dataset}_{args.model}_{args.suffix}.h5')

        # load data
        x_train, y_train, x_val, y_val, x_test, y_test = get_dataset()

        size = 100
        # input and output placeholders
        x_input = tf.placeholder(tf.float32,
                                 [size, img_rows, img_cols, img_channels])
        y_input = tf.placeholder(tf.float32, [size, num_classes])

        # attack parameters
        if args.dataset == 'cifar10':
            epsilon, k, a = 8, 20, 2
        elif args.dataset in ['svhn', 'gtsrb']:
            epsilon, k, a = 0.03, 100, 0.001
        elif args.dataset == 'lisa':
            epsilon, k, a = 0.1, 100, 0.01

        # initialize pgd attack
        method = LinfPGDAttack(
                    x_input,
                    y_input,
                    model(x_input),
                    epsilon,
                    k,
                    a,
                    clip_max,
                    True,
                    'xent'
                 )

        # generate adversarial examples
        results = np.zeros((num_classes, num_classes), dtype=int)
        print('-'*50)
        for start in range(0, 10):
            sys.stdout.write(f'\rattacking batch {start}')
            sys.stdout.flush()

            # get a batch
            inputs = x_val[start*size : (start+1)*size]
            labels = y_val[start*size : (start+1)*size]
            if len(inputs) != size:
                break

            # generate adv
            adv = method.perturb(inputs, labels, sess)
            pred_adv = model.predict(adv)

            # record prediction
            for i in range(size):
                id_ori = np.argmax(labels[i])
                id_adv = np.argmax(pred_adv[i])
                results[id_ori, id_adv] += 1
        print()

        # show robustness
        robust = []
        total = 0
        print('-'*80)
        for i in range(num_classes):
            for j in range(num_classes):
                print(results[i, j], end='\t')
                if i == j:
                    robust.append(results[i, j] / np.sum(results[i]))
                    total += results[i, j]
            print()
        print('-'*80)

        print('\n'.join(f'{i}: {robust[i]}' for i in range(len(robust))))
        print('-'*50)
        print(f'robustness: {total / np.sum(results)}')


def validate():
    prefix = f'{args.dataset}_{args.model}_{args.suffix}'

    # load model
    model = get_model()
    model.load_weights(f'ckpt/{prefix}.h5')
    model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizers.SGD(),
            metrics=['accuracy']
          )

    # load data
    x_train, y_train, x_val, y_val, x_test, y_test = get_dataset()

    # initialize trigger generation
    trigger = Trigger(
                  model,
                  img_rows=img_rows,
                  img_cols=img_cols,
                  num_classes=num_classes,
                  clip_max=clip_max
              )

    print('-'*80)
    print('validating pair distance...')
    print('-'*80)
    if args.pair != '0-0':
        # generate triggers for one class pair
        source, target = list(map(int, args.pair.split('-')))
        mask, pattern, _ = trigger.generate((source, target), x_val, y_val)
        size = np.sum(np.abs(mask))
        print(f'distance for {source}->{target}: {size:.2f}')
    else:
        fsave = open(f'data/distance/{prefix}_{args.seed}.txt', 'a')

        # generate triggers for all class pairs
        for source in range(num_classes):
            for target in range(num_classes):
                if source != target:
                    mask, pattern, _\
                            = trigger.generate((source, target), x_val, y_val)
                    size = np.sum(np.abs(mask))

                    fsave.write(str(size) + ',')
                    fsave.flush()
            fsave.write('\n')


def show():
    prefix = f'data/distance/{args.dataset}_{args.model}'

    data = []
    for i in range(3):
        data_seed = []
        for line in open(f'{prefix}_{args.suffix}_{i}.txt', 'r'):
            line = line.strip().split(',')[:-1]
            line = list(map(float, line))
            data_seed.append(line)
        data.append(data_seed)

    data = np.array(data)
    data[data == 0] = np.inf
    data = np.min(data, axis=0)
    data[data == np.inf] = 0
    np.save(f'{prefix}_{args.suffix}', data)

    print('-'*100)
    for i in range(num_classes):
        for j in range(num_classes):
            if j < i:
                print(str(data[i, j]), end='\t')
            elif j == i:
                print('-', end='\t')
            else:
                print(str(data[i, j-1]), end='\t')
        print()
    print('-'*100)
    print(f'average distance: {np.mean(data):.2f}')

    if 'nat' in args.suffix:
        base = np.load(f'{prefix}_nat.npy')
    elif 'adv' in args.suffix:
        base = np.load(f'{prefix}_adv.npy')
    diff = np.mean((data - base) / base)
    print(f'increase percent: {diff*100:.2f}%')



################################################################
############                  main                  ############
################################################################
def main():
    if args.phase == 'moth':
        moth()
    elif args.phase == 'test':
        test()
    elif args.phase == 'measure':
        measure()
    elif args.phase == 'validate':
        validate()
    elif args.phase == 'show':
        show()
    else:
        print('Option [{}] is not supported!'.format(args.phase))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')

    parser.add_argument('--gpu',     default='0',        help='gpu id')
    parser.add_argument('--seed',    default=0,          help='seed index', type=int)

    parser.add_argument('--phase',   default='test',     help='phase of framework')
    parser.add_argument('--dataset', default='cifar10',  help='dataset')
    parser.add_argument('--model',   default='resnet20', help='model')
    parser.add_argument('--type',    default='nat',      help='model type (natural or adversarial)')
    parser.add_argument('--suffix',  default='nat',      help='checkpoint path')
    parser.add_argument('--pair',    default='0-0',      help='label pair')

    parser.add_argument('--batch_size', default=128, type=int,   help='batch size')
    parser.add_argument('--epochs',     default=2,   type=int,   help='hardening epochs')
    parser.add_argument('--data_ratio', default=1.0, type=float, help='ratio of training samples for hardening')
    parser.add_argument('--warm_ratio', default=0.5, type=float, help='ratio of batch samples to stamp trigger during warmup')
    parser.add_argument('--portion',    default=0.1, type=float, help='ratio of batch samples to stamp trigger during orthogonalization')

    args = parser.parse_args()

    # set gpu usage
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.Session(config=config)

    # set random seed
    SEED = [1024, 557540351, 157301989]
    SEED = SEED[args.seed]
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    img_rows, img_cols, img_channels =  32, 32, 3
    clip_max = 255.0 if args.dataset == 'cifar10' else 1.0

    if args.dataset == 'lisa': 
        num_classes = 18
    elif args.dataset == 'gtsrb':
        num_classes = 43
    else:
        num_classes = 10

    # main function
    time_start = time.time()
    main()
    time_end = time.time()
    print('='*50)
    print('Running time:', (time_end - time_start) / 60, 'm')
    print('='*50)
