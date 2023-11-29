# Decoding Topological Codes with Graph Neural Networks

The project aims to investigate the potential utility of employing Graph Neural Networks (GNNs) to decode the syndrome 
of topological quantum codes. The two most important aspects of QEC Codes decoding algorithms are their efficiency and 
speed, between which there is an inevitable trade-off. In this project, the speed aspect is not considered - the 
algorithm implementations are only prototypical in order to generally illustrate the effectiveness of using GNNs.



Topological codes and operations on them were implemented using the qecsim library, see:  
`D. K. Tuckett, Tailoring surface codes: "Improvements in quantum error correction with biased noise",`  
`Ph.D. thesis, University of Sydney (2020), (qecsim: https://github.com/qecsim/qecsim)`

## Environment setup

1. Install Python 3.9 (and cuda11.7 for GPU support)
2. Install poetry: `pip install poetry`
3. Go to project directoryL `cd ../project/path/DTC-GNN`
4. Install poetry environment: `install poetry`
5. If cuda11.7 is installed run:
    - `poetry run poe force-pt-cuda11`
    - `poetry run poe force-dgl-cuda11`
