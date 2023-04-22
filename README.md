# HiGGs: Hierarchical Generation of Graphs
### Github repository for HiGGs, submitted to NeurIPS 23, Anonymous Authors

HiGGs is a framework to generate large (attributed) graphs, making use of conditional sampling to produce graphs a quadratic order larger than those possible with its component models.

The graphs we produce are (to our knowledge) the largest produced using deep-learning methods.

Real Facebook Page-Page            |  HiGGs Facebook Page-Page
:-------------------------:|:-------------------------:
![](https://github.com/higgs-neurips-23/HiGGs/blob/main/figures/Real_fb.png)  |  
![](https://github.com/higgs-neurips-23/HiGGs/blob/main/figures/HiGGs_fb.png)
 22470 Nodes, 171002 Edges  |  21643 Nodes, 283552 Edges  


This implementation makes use of DiGress (https://github.com/cvignac/DiGress) as well as our variant implementation for edge prediction edge-DiGress.

## Environment installation
 - Create a conda env while specifying a python version`env create -n digress python=3.8`
 - Install basic libraries `conda install numpy pandas matplotlib scikit-learn`
 - Check CUDA version (`nvidia-smi` displays this)
 - Install PyTorch + others (`conda install pytorch torchvision torchaudio cudatoolkit=CUDA_VERSION -c pytorch -c conda-forge`)
 - Install PyTorch-Geometric (`conda install pyg -c pyg`)
 - Install hydra/graph-tool (`conda install hydra-core graph-tool -c conda-forge`) (I think this is where issues start?)
 - Run `python dgd/hydra_main_test.py` to check hydra with non-digress packages
 - Run `python dgd/hydra_main_test_with_digress.py` to check hydra with digress imports
 
 ### Alternative:
  - Run `conda env create -f higgs.yml` (this assumes you're on CUDA 11.8)

## Download the data

  - SBM data available at https://github.com/KarolisMart/SPECTRE/tree/main/data
  - Cora and Facebook graphs are downloaded in-code during training or sampling

## Running Code

 - Pre-trained models for each dataset can be downloaded using `download_models.sh`
 - This will create a directory `/saved_models/` which contains the pre-trained models for each hierarchy

### Model Training
  - Model code is under `dgd/` and is based on DiGress from https://github.com/cvignac/DiGress
  - Config files are available for the SBM, Cora and Facebook graphs
  - Training for an individual hierarchy can be run as `python dgd/main.py dataset=dataset h=hierarchy`
  - Scripts for training all models are under scripts/train_X.sh

### HiGGs Sampling
   - Sampling code is under `higgs/`
   - Run similarly to DiGress, ie
   - `python higgs/sample_main.py general=sampling_config`
   - Sampling config is under `configs/general/sample_X.yml`
   - **IMPORTANT**: Before running please change the config file to contain the global paths to the sampling models, i.e.
   - `h1_model: /path/to/repository/saved_models/cora/h1_cora.ckpt`
