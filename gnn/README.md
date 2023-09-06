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

NOTES:

Various possibilities for node features:
- Pre-computed features: these can be both structural (e.g. degrees) or semantic (e.g. number of live variables)
- Embeddings for nodes: These require node ids that are generalizable across examples. Assigning a unique id to each node is not useful if we want to generalize to unseen source code.
-- The mapping from nodes (basic blocks) to a unique id should therefore, depend on contents of the blocks. These can then be used compute embeddings for unseen blocks.
-- There are various possibilities including using an RNN (LSTM or GRU in practice) to create an embedding from instructions in the block. Since the set of instructions is small, one can use the RNN to generate an embedding for any block. This is the strategy being implemented.

Mechanics:
- Data is read as a tuple (node features, edge list)
- Node features are expected to be packed in a tensor if using torch_geometric.data.Data
- Instruction sequences from blocks are of varying lengths
- We will:
-- Read block sequences and pad them with a special token for the no-op instruction: shape = (#nodes, max seq length)
-- Pack the sequences and pass to the RNN to create embeddings: shape = (#nodes, hidden_dim)
-- Create a new torch_geometric.data.Data object with embeddings as features and edge_list

- Note that when a dataloader is used, torch geometric follows the strategy of creating a union of graphs (block diagonal adjacency matrix). The padding/packing strategy still works.

- Have to test the strategy when node sampling is used.