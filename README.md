SubOptGraph is a DL tool for accurately predicting optical properties of organic emitters constructed by the Pytorch 1.10 and trained on the Nvidia 4090 Ti.

Your can use the model parameters in the folders based on the Deep4Chem dataset or train the model on your own.

All the related data are listed in the 'SubOptGraph/data/'.

To construct the molecular graph, torch_geometric package is needed and some changes for the source code of the torch_geomeric is included in the collate.py and data.py files in order to conduct the edge-centered message passing phase, so your can replace them after installing relevant packages.
