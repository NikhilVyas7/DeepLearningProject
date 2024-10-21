# DeepLearningProject



## Setup

### Pace Setup

First, login into PACE-ICE via `ssh <GT_USERNAME>@login-ice.pace.gatech.edu`. It will ask you for a password, which will just be your gt password. For example, for me it would be `ssh athalanki3@login-ice.pace.gatech.edu`

### Repo Setup

You will first want to go into the scratch directory, via `cd scratch`. From there you can clone the repo: `git clone https://github.com/NikhilVyas7/DeepLearningProject.git`, then go into it: `cd DeepLearningProject`. Once you've cloned the repository you can create a conda environnment via `conda create -n dl_project`.  When the environment is fully created, you can go into it via `conda activate dl_project` then install other dependencies: `pip install -r requirements.txt`.


 Then, run `sbatch download_data.sh` to download the dataset in a seperate process. 



### Download

First, run `conda install -c conda-forge awscli` inside the conda environment to download the awscli. Once aws has been setup, run `aws s3 sync s3://radiantearth/landcovernet/ ./data --endpoint-url=https://data.source.coop` to verify
that you have proper access to AWS. If you can, skip right away to downloading the dataset via `sbatch download_data.sh`. If you have access issues, follow the AWS setup steps.


### AWS Setup
Next, you will have to sign up for AWS to get access to the data
Go to https://aws.amazon.com/console/ and create an account (pick the free option).
Once created, navigate to the IAM Console through the search bar
You will have to create a new user in this section. Click users, then click create user and follow the steps.
Once created, on the IAM menu, you will be able to create an access key. Make sure to save the access key id and the secret access key.
In your terminal, try `aws configure` again to see if it has been installed properly.
This time it should prompt you for an access key id, a secret access key, a region name, and an output format.
Access key id: paste the access key id from the IAM 

Secret access key: paste the secret access key from the IAM

Region name: us-east-1

Output format: json



