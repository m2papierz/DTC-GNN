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

## Author
+ [Mateusz Papierz](m2papierz@gmail.com)