# Weakly supervised convolutional dictionary learning (WSCDL)

This project is the code for the paper "Weakly supervised common and distinctive convolutional dictionary learning". Demo data will be generated in the code.
## Requirements
```
Python 3.6+
PyTorch 1.1.0+
Numpy
Matplotlib
```
## Details
The default data type is `torch.tensor` with precision `float32`, the code is written for single
GPU or CPU. 
### Input
Input data `X` should be matrix-like with the labels `Y`. The label for each sample should be marked a vector of 0 or 1. 0 means there is no such feature, while 1 means such feature exits.

### Output
There are two type of outputs: 
1. Graphic outputs
2. the console outputs. 

Graphic output will show the input data, labels, reconstructed data and reconstructed labels. 
And the console output will show the training details with ecliped time per epoch, sparsity level(optional), and classification accuracy.

### Config
The configuration is controled by a class `OPT` in the `util.py` file. All the third-party functions are all imported and definded in the `utils.py`.
Here lists some parameters critical to training results. For more information, please refer to the paper.

| Hyper parameter | Description |
| --------------- | ----------- |
| \lambda | the sparsity panelty |
| \eta | the label panelty |
| \mu | the low-rank panelty |

## FAQ
1.Why sometimes cpu is used a lot?

Because `PyTorch` `v1.2.0` gpu version of the function `svd` is slower than cpu version, in some steps, cpu version of `svd` is used for speed issue.

2. Does it use tensor calculation?

Yes. Although the input data format is matrix, it actually will be reshaped into tensor for calculation efficiency.

