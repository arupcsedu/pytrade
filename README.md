# Libcudf Examples
### Author: Arup Sarker (arupcsedu@gmail.com; djy8hg@virginia.edu)

pytrade is a AI-based trading system to predict stock price

## Install Rapids
`conda create -n rapids-24.06 -c rapidsai -c conda-forge -c nvidia  rapids=24.06 python=3.11 cuda cuda-version=12.0`

## Activate Conda environment
`conda activate rapids-24.06`

## Install pytorch
`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

## Export LD_LIBRARY_PATH
`export LD_LIBRARY_PATH=$CONDA_PREFIX/lib`

## If you see nvtx error, get NVTX using Conda
 `conda install -c conda-forge nvtx`
 

 Current examples:

- Basic: demonstrates a basic use case with libcudf and building a custom application with libcudf
- Strings: demonstrates using libcudf for accessing and creating strings columns and for building custom kernels for strings
- Nested Types: demonstrates using libcudf for some operations on nested types

