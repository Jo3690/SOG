SubOptGraph is a deep learning framework for accurately predicting optical properties of organic emitters constructed by the Pytorch 1.10.

__1. Requirements:__

Python    3.7.3

Pytorch    1.10.0

RDKit    2020.03.3

torch_geometric    2.2.0

Scikit-learn


__2. Usage:__

Some changes for the source code of the torch_geomeric package are included in the SubOptGraph/collate.py and SubOptGraph/data.py. In order to conduct the edge-centered message passing fluently, you need to replace these two files after installing torch_geometric package. And you need to change all the directory mentioned in the scripts (i.e., "SubOptGraph/*.py") to your own path.

In the "SubOptGraph/" folder, all the scripts with the format of ".py" are used to train the model or make predictions.

You can run the script in your terminal using the following command as an example:

python -m absorption.py model.num_layers 5 model.mini_layers 3 train.epochs 100

You can also change the hyperparameters by yourself according to the instructions in "core/config.py"


__3.  Files__

absorption.py; emission.py; fwhm.py; plqy.py. These four are used for absorption, emission, fwhm, and plqy training.

10-foldchemabs.py; 10-foldchememi.py; 10-foldchemplqy.py. These three are used for ChemFluor dataset.

10fold-abs.py is used for the comparison with ChemMF deep learning model.

smfluoabs.py is used for SMFluo1 dataset.

5-foldbodipy.py is used for BODIPYs dataset.

GCN.ipynb is used for GCN model.

transfer.py is used for transfer learning. 

test.py is used for making predictions.


__4.  Data generation__

All the related data are listed in the 'SubOptGraph/data/' with the ".txt" and "xlsx" formats. To construct the molecular graph data, you need to make two folders like:

"SubOptGraph/data/Absorption/raw/" "SubOptGraph/data/Absorption/full/processed/". 

After putting the data file such as 'deep4chem_absoption.txt' into "SubOptGraph/data/Absorption/raw/", then, by running the script 'absorption.py', the model can construct and save molecular graph features in the "SubOptGraph/data/Absorption/full/processed/" automatically.(You need to change the file path in the 'absorption.py' to your own path)


__5. Code and hyperparameters__

The codes for node-centered mpnn, edge-centered mpnn, subgraph mpnn, and data training are placed in "core/" folder.

The codes for model training and testing are placed in "SubOptGraph" folder.

"SubOptGraph/configs/*.yaml" are used for model hyperparameters. 

The model parameters are listed in "SubOptGraph/Model_paras/" and "SubOptGraph/blue_OLED_paras/" folders. You can load them to make predictions with "test.py"

