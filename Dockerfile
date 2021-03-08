FROM intel/oneapi-basekit

RUN apt-get update && apt-get install -y \
  python3 \
  tree \
  tar \
  libssl-dev \
  wget \
  gfortran \
  unzip \
  libblas-dev \
  liblapack-dev \
  libsundials-dev \
  libgfortran3 \
  nano \
  python-opengl

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
  yes "yes" | bash Miniconda3-latest-Linux-x86_64.sh &&\
  rm Miniconda3-latest-Linux-x86_64.sh

# only works with python 3.5
RUN conda create -n py35 python=3.5 &&\
  wget https://github.com/Kitware/CMake/releases/download/v3.19.3/cmake-3.19.3.tar.gz &&\
  tar xfz cmake-3.19.3.tar.gz &&\
  rm cmake-3.19.3.tar.gz &&\
  cd cmake-3.19.3 &&\
  ./configure &&\
  make &&\
  make install &&\
  wget https://github.com/xianyi/OpenBLAS/releases/download/v0.3.13/OpenBLAS-0.3.13.tar.gz &&\
  tar xfz OpenBLAS-0.3.13.tar.gz &&\
  rm OpenBLAS-0.3.13.tar.gz &&\
  cd OpenBLAS-0.3.13 &&\
  make && make install &&\
  conda install -n py35 -c conda-forge sundials &&\
  wget https://github.com/modelon-community/Assimulo/archive/Assimulo-3.2.4.tar.gz &&\
  tar xfz Assimulo-3.2.4.tar.gz &&\
  cd Assimulo-Assimulo-3.2.4 &&\
  python setup.py install --sundials-home=/opt/intel/oneapi/intelpython/latest/envs/py35 --lapack-home=/usr/lib/x86_64-linux-gnu/ --blas-home=/usr/lib/x86_64-linux-gnu/ &&\
  wget https://jmodelica.org/FMILibrary/FMILibrary-2.0.1-src.zip &&\
  unzip FMILibrary-2.0.1-src.zip &&\
  cd FMILibrary-2.0.1 &&\
  mkdir build-fmil &&\
  cd build-fmil &&\
  cmake .. && make install test

# CMD conda install -c conda-forge -c hcc -c pytorch pyfmi gym scikit-learn pandas pytorch

COPY resources/ModelicaGym /usr/local/JModelica/ThirdParty/MSL/ModelicaGym
