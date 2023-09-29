# Decoding Topological Codes with Graph Neural Networks
**Description**

## Project setup

1. Install Python 3.9 (and cuda11.7 for GPU support)
2. Install poetry: `pip install poetry`
3. Go to project directoryL `cd ../project/path/DTC-GNN`
4. Install poetry environment: `install poetry`
5. If cuda11.7 is installed run:
    - `poetry run poe force-pt-cuda11`
    - `poetry run poe force-dgl-cuda11`

## Project navigation

* `dtc_gnn` - main package
  * data_management - all data manipulation modules
    * data_transforms - data transforms modules
  * utils - utility functions
* static_files - configurations, hydra run scripts etc.

## Project execution

### Data generation
Data generation is one-shot necessity - project pipeline is not adjusted to running multiple experiments with different 
datasets generation parameters, as the training data in the project is pre-defined and is not subject to research. This 
is also the reason why the data generation process (especially the generation of graphs for decoding by GNN) was not 
written in a maximally optimal way. I cared more about the readability of the operations than optimizing them, which 
could be done, for example, by computing only at the level of indexes and node positions, rather than abstract auxiliary
classes.

Data generation run: 
- default parameters: `python static_files/hydra_runs/data_generation.py`
- override example: `python static_files/hydra_runs/data_generation.py ++n_samples=100000 ++split_ratio=0.7`

## Author
+ [Mateusz Papierz](m2papierz@gmail.com)