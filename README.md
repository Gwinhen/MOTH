# Model Orthogonalization: Class Distance Hardening in Neural Networks for Better Security

This is the implementation for IEEE S&P 2022 paper "Model Orthogonalization: Class Distance Hardening in Neural Networks for Better Security."

The PyTorch version is coming soon...

## Prerequisite

The code is implemented and tested on Keras with TensorFlow backend. It runs on Python 3.6.9.

### Keras Version

* Keras 2.3.0
* Tensorflow 1.14.0

## Usage

The main functions are located in `src/main.py` file.

### Model Orthogonalization

To harden a model using MOTH, please use the following command:

   ```bash
   python3 src/main.py --phase moth
   ```

The default dataset and model are CIFAR-10 and ResNet20. You can harden different model structures on other datasets by passing the arguments `--dataset [dataset]` and `--network [model structure]`. We have included four datasets (CIFAR-10, SVHN, LISA, and GTSRB) and four model structures (ResNet, VGG19, NiN, and CNN). (The datasets will be uploaded soon.)

To measure the pair-wise class distance, please run:

   ```bash
   python3 src/main.py --phase validate --suffix [suffix of checkpoint] --seed [seed id]
   ```

Models hardened by MOTH will have a suffix of `_moth` in addition to the original checkpoint path. Please provide the checkpoint extension using argument `--suffix`. The distance shall be measured using three different random seeds by passing seed ids `0`, `1`, and `2` to the argument `--seed` separately.

The final pair-wise class distance of the evalauted model can be obtained through the following command:

   ```bash
   python3 src/main.py --phsae show --suffix [suffix of checkpoint]
   ```

It prints out a matrix of class distances of all the pairs. Each row denotes the source label and each column the target label. The average distance and relative enlargement are also presented in the end.

### Model Functionality

To test the accuracy of a model, simply run:

   ```bash
   python3 src/main.py --phase test --suffix [suffix of checkpoint]
   ```

The robustness of a given model can be evaluated using PGD with the following command:

   ```bash
   python3 src/main.py --phase measure --suffix [suffix of checkpoint]
   ```

## Acknowledgement

The code of trigger inversion is inspired by [Neural Cleanse](https://github.com/bolunwang/backdoor).

The PGD code is adapted from [cifar10\_challenge](https://github.com/MadryLab/cifar10_challenge).

Thanks for their amazing implementations.

## Reference

Please cite for any purpose of usage.

```
@inproceedings{tao2022model,
  title={Model Orthogonalization: Class Distance Hardening in Neural Networks for Better Security},
  author={Tao, Guanhong and Liu, Yingqi and Shen, Guangyu and Xu, Qiuling and An, Shengwei and Zhang, Zhuo and Zhang, Xiangyu},
  booktitle={2022 IEEE Symposium on Security and Privacy (SP)},
  year={2022},
  organization={IEEE}
}
```
