# ARC
ARC: A tool for creating Artificial languages with Rhythmicity Control

![ARC Flowchart](ARC_Flowchart.pdf)

## Setup
Welcome to ARC! 

This is code for the paper (insert paper).

The following describes how you can set up the software and run the experiments from the paper.

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
The simplest is to clone this repository and install ARC in editable mode:
```shell
pip install -e .
```

You can can also install ARC directly from git as a python package
```shell
pip install git+https://github.com/milosen/arc.git
```

## Run the code from the paper

Install jupyter
```shell
pip install jupyter
```
... and set up the virtualenv based ipython-kernel (if you haven't already):
```shell
python -m ipykernel install --user --name=arc
```

Start jupyter
```shell
jupyter notebook
```
and select the notebook `controlled_stream_generation.ipynb`.
Don't forget to select the `arc` kernel in the jupyter session.

## Optional: ARC's type system
If you want to adapt ARC to your own research, you'll probably want to take a look at `arc_types.ipynb`.
