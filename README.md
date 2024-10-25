# DeepLearningProject



## Setup

### Pace Setup

First, login into PACE-ICE via `ssh <GT_USERNAME>@login-ice.pace.gatech.edu`. It will ask you for a password, which will just be your gt password. For example, for me it would be `ssh athalanki3@login-ice.pace.gatech.edu`

### Repo Setup

You will first want to go into the scratch directory, via `cd scratch`. From there you can clone the repo: `git clone https://github.com/NikhilVyas7/DeepLearningProject.git`, then go into it: `cd DeepLearningProject`. Once you've cloned the repository you can set up the conda
environment by running `./scripts/setup_env.sh`.

### Download Dataset

The dataset is located at this [dropbox](https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&e=2&dl=0). You can simply download the dataset, scp it to bring it to pace-ice, then unzip it so that the
dataset is located in the repo with the name `FloodNet`. To unzip, run 
`UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip FloodNet.zip -d FloodNet`. The unzip disable zipbomb is required because the unzip function falsely
thinks the dataset is a zipbomb.




