# Weakly supervised convolutional dictionary learning (WSCDL)

This project is the code for the paper "Weakly supervised common and distinctive convolutional dictionary learning". 
## Requirements
```
Python 3.13+
PyTorch 2.7.0+
Numpy
Matplotlib
```
Suggest using uv to preproduce the depandency
`uv sync`
## Details
The default data type is `torch.tensor` with precision `float32`, the code is written for single GPU or CPU.
User can start with `main_toy.py` to get the general idea.

### Input
Input data `X` with the labels `Y`. The label for each sample should be marked a vector of 0 or 1. 0 means there is no such feature, while 1 means such feature exits.
The data `X` could be N samples of 1-d vector or 2-d matrix, which means `X` could be a matrix or 3d tensor.

### Output
There are two type of outputs: 
1. Graphic outputs
2. the console outputs. 

Graphic output will show the input data, labels, reconstructed data and reconstructed labels. 
And the console output will show the training details with ecliped time per epoch, sparsity level(optional), and classification accuracy.

### Config
There 3 `util` files, namely `util.py`, `util2.py` and `util3.py`, containing the functions for 1d-data, 2d-data and 2d-data using auto-encoder structure, respectly.

The configuration is controled by a class `OPT` in the `util` file. All the third-party functions are all imported and definded in the `utils.py`.
Here lists some parameters critical to training results. For more information, please refer to the paper.

| Hyper parameter | Description |
| --------------- | ----------- |
| \lambda | the sparsity panelty |
| \eta | the label panelty |
| \mu | the low-rank panelty |

## FAQ
1.Why sometimes cpu is used a lot?

Because `PyTorch` `v1.2.0` gpu version of the function `svd` is slower than cpu version, in some steps, cpu version of `svd` is used for speed issue.

2.Does it use tensor calculation?

Yes. Although the input data format is matrix, it actually will be reshaped into tensor for calculation efficiency.

