# Optimization over Trained (and Sparse) Neural Networks: A Surrogate within a Surrogate

Corresponding Author: Dr. Thiago Serra

Co-Authors: Hung Pham; Aiden Ren; Ibrahim Tahir; Jiatai Tong

## Setting Up the Environment

To set up the environment for this project, follow these steps:

### 1. Install Conda
If you don't already have Conda installed, download and install it from the [official Conda website](https://docs.conda.io/en/latest/miniconda.html). This is a program to manage dependencies of the project to make sure everyone is running the same version. Most linux computer already have it, just type `conda --version`.

### 2. Clone the Repository
Clone this repository to your local machine, I use [Github cli](https://cli.github.com/), which is much easier to pull and push code:
```bash
gh auth login
gh repo clone https://github.com/hqp001/MLResearch2024_2ndPhase.git
cd MLResearch2024_2ndPhase
```

### 3. Set up Conda environment
```bash
conda env create -n venv -f environment.yml
conda activate venv
conda deactivate # Do this after you don't write code for this project
```

### 4. To make changes and push to this repo
There might be another ways to do this. For me, I use git command for everything local and github cli for everything with push and pull.
To make changes:
```bash
git checkout -b "Yourname"
# Make some changes
git add .
git commit -m "Create new file"
git add .
git commit -m "Delete this file" # You can git add or commit anytimes you want
# After making sure everything is good, create a pull request to this repo using
gh pr create --title "Abc" --body "DEF"
```

## How to Run the Scripts

### Running the Trainer

You have to run the training first to train the sparse model

```bash
# Using Python script
python adversarial_example/RunTraining.py

# or

sbatch bisonnet-template_gpu.sh # on Bisonnet
```


To run the solver, you can either execute the `adversarial_example/RunSolver.py` script or use the `bisonnet-template_cpu.sh` script in the main directory. 

```bash
# Using Python script
python adversarial_example/RunSolver.py

# Using shell script
sh ./bisonnet-template_cpu.sh

# or

sbatch bisonnet-template_cpu.sh # on Bisonnet
```
