import json

config_data = {
    "bidir": True,
    "batch_size": 64,
    "num_layers_enc": 2,
    "hidden_dim_enc": 64,
    "num_layers_dec": 2,
    "hidden_dim_dec": 32,
    "emb_dim": 64,
    "layer_dims_gnn": [32]
}

# Dump configuration to JSON file
with open('config.json', 'w') as f:
    json.dump(config_data, f, indent=4)

print("Configuration dumped to config.json successfully.")
