SubOptGraph is a tool for accurately predicting optical properties of organic emitters constructed by the Pytorch 1.10 and trained on the Nvidia 4090Ti 

Your can use the model parameters in the checkpoint file based on the Deep4Chem dataset or train the model on your own.

All the related data are listed in the 'SubOptGraph/data/' file

To construct the molecular graph, torch_geometric package is needed and some changes of the source code for the torch_geomeric is included in the collate.py and data.py in order to conduct the edge-centered message passing phase
