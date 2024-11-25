# DeepLearningProject



## Setup

### Pace Setup

First, login into PACE-ICE via `ssh <GT_USERNAME>@login-ice.pace.gatech.edu`. It will ask you for a password, which will just be your gt password. For example, for me it would be `ssh athalanki3@login-ice.pace.gatech.edu`. Then you want to allocate youself an interactive compute node with a gpu via `salloc -N1 -t6:00:00 --gres=gpu:1 --cpus-per-task=15 --mem=200G`. The `-t6:00:00` specifies you want 6 hours of time. The excessive number of cpus cores really helps when loading data.

### Repo Setup

You will first want to go into the scratch directory, via `cd scratch`. From there you can clone the repo: `git clone https://github.com/NikhilVyas7/DeepLearningProject.git`, then go into it: `cd DeepLearningProject`. Once you've cloned the repository you can set up the conda
environment by running `./scripts/setup_env.sh`.

### Download Dataset

The dataset is located at this [dropbox](https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&e=2&dl=0). You can simply download the dataset, scp it to bring it to pace-ice, then unzip it so that the
dataset is located in the repo with the name `FloodNet`. To unzip, run 
`UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip FloodNet.zip -d FloodNet`. The unzip disable zipbomb is required because the unzip function falsely
thinks the dataset is a zipbomb.

### Shrink Dataset

Once in a compute node with a lot of cores and memory,and in the conda environment, simply run `python data/shrink_dataset.py FloodNet ShrunkenFloodNet`. This will very quickly create the shrunken dataset.


## Task List


- Create script to convert label image containign (1s,2s.) etc to something that can be visualized with colors ,according to `FloodNet/ColorMasks-FloodNetv1.0/ColorPalette-Values.xlsx`.
- Explore DataDistributedParallel so we can use more than 2 gpus.

- Explore using Diffusion or GAN models to create the images from the labels

- Explore HPO (Hyper-parameter Optimization) on the UNet to improve baseline performance

- Explore methods of optimizing the model inference time in general





