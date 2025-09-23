#!/bin/bash
conda env create -f sugar_environment.yml
conda activate r2r2r_sugar
cd dependencies/SuGaR/gaussian_splatting/submodules/simple_knn
pip install -e .
# NVCC_FLAGS="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=sm_80" pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0
pip install gsplat==1.4.0
cd ../../../
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
cd ../
