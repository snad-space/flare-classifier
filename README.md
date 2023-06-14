# flare-classifier

## Project description

This repository is dedicated to the work related to the search for red dwarf flares using machine learning techniques.
The task was reformulated as a binary classification problem, where label 1 indicates flare event, 0 – other events. 
Three models of classificators based on features extracted from light curve were trained: Random forest, CatBoost and MLP.

### Data Version Control

[DVC](https://dvc.org), Data Version Control, is a useful tool for data versioning with git. We use it with our S3-compatible storage available at s3.lpc.snad.space

#### Credentials

You need a login at https://minio.lpc.snad.space, contact hombit@gmail.com if you believe you could have one.

Steps to setup `aws-cli` before using `dvc` with our remote.
1. Install `aws-cli` from yout package manager, like `brew install awscli` or `python3 -mpip install awscli`
2. Run `aws configure` and keep your terminal open
3. Go to https://minio.lpc.snad.space/access-keys and create an access key, copy-paste public and private keys into the terminal, set region to `us-east-1`, nothing for output format (see details [here](https://min.io/docs/minio/linux/integrations/aws-cli-with-minio.html))
5. Run `aws configure set default.s3.signature_version s3v4`
6. Save the access key in the browser, keep the policy empty

#### Start with DVC

1. Install DVC with s3 extra, for example via `python3 -mpip install 'dvc[s3]'` (note quotes) or via your package manager
2. `dvc pull` gets the data from the remote and put it where it belongs
3. `dvc add path` adds new / updated datafile to a local dvc repository
4. `dvc push` pushes data to the remote
5. `git commit` commits dvc hashes (not data files themselves) into the local git repo
6. `git push`

Do not forget to `dvc push` when `git push` (maybe we need git post-commit hook for this?).

### How to run demo DVC pipeline

To run a pipeline use `dvc repro` command in main repo. 
The pipeline consists of several stages:

- prepare_datasets: feature extraction process, raw data combines to train/val/test samples.
- train_rf: training process of Random Forest model.
- train_catboost: training process of CatBoost model.
- train_mlp: training process of MLP model.
- evaluate_rf: Random Forest evaluation
- evaluate_catboost: CatBoost evaluation
- evaluate_mlp: MLP evaluation

All metrics and metadata placed in `metrics` folder and respective subfolder after pipeline execution.

### LINCC-Frameworks Python Project Template

This project was automatically generated using the LINCC-Frameworks [python-project-template](https://github.com/lincc-frameworks/python-project-template).

For more information about the project template see the [readme](https://github.com/lincc-frameworks/python-project-template#readme) documentation.
