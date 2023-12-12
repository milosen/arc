# ARC
ARC: A tool for creating Artificial languages with Rhythmicity Control

## Setup
Welcome to ARC!

### Optional: create a new virtual environment
Create a new virtual environment, e.g. using anaconda
```shell
conda create -n test_arc python=3.9
```
... and activate it 
```shell
conda activate test_arc
```

### Install Package
You can either install ARC directly from git
```shell
pip install git+https://github.com/milosen/arc.git
```

Or clone this repository and install in editable mode:
```shell
pip install -e .
```

## Tutorials
It is recommended to run the tutorial notebooks in the `notebooks/` directory:
Install jupyter
```shell
pip install jupyter
```
... and set up the virtualenv based ipython-kernel (if you haven't already):
```shell
python -m ipykernel install --user --name=arc
```

Run
```shell
jupyter notebook
```

If you did not clone the repository, download the contents of the `notebooks/` directory and select one of the tutorial notbooks from inside the jupyter session.
Don't forget to select the `arc` kernel in the jupyter session.
