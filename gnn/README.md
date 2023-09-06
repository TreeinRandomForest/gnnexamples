## Project Description

This project is a framework for training self-supervised graph neural network models on code. More precisely, the inputs are graphs derived from structures like abstract syntax trees, control flow graphs, dataflow graphs and call flow graphs.

This is work in progress and will change substantially.

### Code outline

data.py: functions to read source files (GIMPLE representation) and derive basic blocks and control flow graphs. This files also includes a preliminary parser that extracts instruction sequences from basic blocks.

model.py: all model definitions should go here. It currently has a convolutional GNN (acting on the control flow graph) and an RNN that creates a basic block embedding from the sequence of instructions. This basic block embedding can then be used for each node by the GNN.

train.py: defines the optimizer, loss function ("criterion") and training loop

validate.py: defines the validation loop

config.py: eventually all choices (hyperparameters, task definition etc.) should go here so it's easy to run multiple experiment by switching the config file

### Work in progress:

- Finish the RNN implementation
- Test preprocessing of GIMPLE files
- Add self-supervised tasks e.g. GNN VAE, masked node prediction
- Scaling tests including neighborhood sampling
-- Want test loss, FLOPS, and # model parameters for various experiments