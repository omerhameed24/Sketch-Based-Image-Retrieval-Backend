## Sketch-Based-Image-Retrieval-Backend

## Installation of Caffe on ubuntu 18.04 in cpu mode conda environment.

1. conda update conda
1. conda create -n testcaffe python=3.5
1. source activate testcaffe
1. conda install -c menpo opencv3
1. sudo apt-get update
1. sudo apt-get upgrade
1. sudo apt-get install -y build-essential cmake git pkg-config
1. sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev protobuf-compiler
1. sudo apt-get install -y libatlas-base-dev
1. sudo apt-get install -y --no-install-recommends libboost-all-dev
1. sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev

1. mkdir build
1. cd build

1. make -j8 l8
1. make all
1. make install

1. conda install cython scikit-image ipython h5py nose pandas protobuf pyyaml jupyter
1. sed -i -e 's/python-dateutil>=1.4,<2/python-dateutil>=2.0/g' requirements.txt

1. for req in $(cat requirements.txt); do pip install $req; done

1. cd ../build
1. cd ../python

1. export PYTHONPATH=pwd${PYTHONPATH:+:${PYTHONPATH}}

1. python -c "import caffe;print(caffe.version)"


After installing caffe, all the packages that are required are needed to be installed from requirements.txt.

