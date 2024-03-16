wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.2.1+cu121.zip
rm libtorch-cxx11-abi-shared-with-deps-2.2.1+cu121.zip
mv libtorch torch
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Release

#MPI
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.2.tar.gz