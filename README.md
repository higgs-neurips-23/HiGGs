# HiGGs: Hierarchical Generation of Graphs
### Github repository for HiGGs, submitted to NeurIPS 23, Anonymous Authors

HiGGs is a framework to generate large (attributed) graphs, making use of conditional sampling to produce graphs a quadratic order larger than those possible with its component models.

The graphs we produce are (to our knowledge) the largest produced using deep-learning methods.

Real Facebook Page-Page            |  HiGGs Facebook Page-Page
:-------------------------:|:-------------------------:
![](https://github.com/higgs-neurips-23/HiGGs/blob/main/figures/Real_fb.png)  |  ![](https://github.com/higgs-neurips-23/HiGGs/blob/main/figures/HiGGs_fb.png)
 22470 Nodes, 171002 Edges  |  21643 Nodes, 283552 Edges  

This implementation makes use of DiGress (https://github.com/cvignac/DiGress) as well as our variant implementation for edge prediction edge-DiGress.

## Download the data
  - We use the SBM graph data from Martinkus et al., available at https://github.com/KarolisMart/SPECTRE/tree/main/data
  - The Cora and Facebook graphs we use are downloaded in-code during training or sampling, and are available from SNAP at https://snap.stanford.edu/data/

## Running Code
 - Pre-trained models for each dataset can be downloaded using `download_models.sh`
 - This will create a directory `/saved_models/` which contains the pre-trained models for each hierarchy

### Environment installation
 - For conda simply run `conda env create -f higgs.yml`

### Model Training
  - Model code is under `dgd/` and is based on DiGress from Vignac et al., available at https://github.com/cvignac/DiGress
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
