## Installation

1\) Environment requirements

* Python 3.x
* Pytorch 1.11
* CUDA 9.2 or higher

The following installation guide assumes ``python=3.7`` ``pytorch=1.11`` and ``cuda=11.3``. You may change them depending on your system.

Create a conda virtual environment and activate it.
```
conda create -n softgroup python=3.7
conda activate softgroup
```


2\) Clone the repository.
```
git clone https://github.com/thangvubk/SoftGroup.git
```


3\) Install the dependencies.
```
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install spconv-cu113
pip install -r requirements.txt
```

4\) Install build requirement.

```
sudo apt-get install libsparsehash-dev
```

4b\) An alternative to Step 4\) that doesn't require superuser access is to install the conda package:

```
conda install -c bioconda google-sparsehash
```

You will need to modify the `include_dirs` parameter in `setup.py`. Its value should be the `include` folder within your `softgroup` conda environment.

5\) Setup
```
python setup.py build_ext develop
```
