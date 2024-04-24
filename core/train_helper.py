
import torch
import time
import numpy as  np

from core.log import config_logger
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import r2_score,mean_squared_error
import os
from scipy.stats import spearmanr, kendalltau



from collections import defaultdict
import logging
from random import Random
from typing import Dict, List, Set, Tuple, Union
import warnings

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import numpy as np

def generate_scaffold(mol: Union[str, Chem.Mol], include_chirality: bool = False) -> str:
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.
    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).
    :param mols: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def scaffold_split(smi:list,
                   data,
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   balanced: bool = False,
                   seed: int = 0):
    r"""
    Splits a :class:`~chemprop.data.MoleculeDataset` by scaffold so that no molecules sharing a scaffold are in different splits.
    :param data: A :class:`MoleculeDataset`.
    :param sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    :param balanced: Whether to balance the sizes of scaffolds in each set rather than putting the smallest in test set.
    :param seed: Random seed for shuffling when doing balanced splitting.
    :return: A tuple of :class:`~chemprop.data.MoleculeDataset`\ s containing the train,
             validation, and test splits of the data.
    """
    assert sum(sizes) == 1

    
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    
    scaffold_to_indices = scaffold_to_smiles(smi, use_indices=True)

    
    random = Random(seed)

    if balanced:  
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)
    
    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    
    train = data[train]
    val = data[val]
    test = data[test]

    return train, val, test




def verify_dir_exists(dirname):  
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))

def mean_relative_error(y_list, pre_list):
    lis = []
    for i in range(len(y_list)):
        lis.append(abs(y_list[i] - pre_list[i])/y_list[i])
    return np.mean(lis)


def run(cfg, create_dataset, create_model, train, test, snapshot_path='path/train/ModelParas',evaluator=None,mean=None,std=None):
    if cfg.seed is not None:
        set_random_seed(cfg.seed)
        cfg.train.runs = 1 

    
    

    
    writer, logger, config_string = config_logger(cfg)

    
    train_dataset, val_dataset, test_dataset = create_dataset(cfg)
   
    
    
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(42)
    train_loader = DataLoader(train_dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers,follow_batch=['edge_attr'])
    val_loader = DataLoader(val_dataset,  cfg.train.batch_size//cfg.sampling.batch_factor, shuffle=False, num_workers=cfg.num_workers,follow_batch=['edge_attr'])
    test_loader = DataLoader(test_dataset, cfg.train.batch_size//cfg.sampling.batch_factor, shuffle=False, num_workers=cfg.num_workers,follow_batch=['edge_attr'])
    test_perfs = []
    vali_perfs = []
    
    
    

    for run in range(1, cfg.train.runs+1):
        
        model = create_model(cfg).to(cfg.device)
        
        
        model.reset_parameters()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
        scheduler = StepLR(optimizer, step_size=cfg.train.lr_patience, gamma=cfg.train.lr_decay)
        
        
        reports = {}
        reports['valid R2'] = 0.0
        Pearson = 0.0
        start_outer = time.time()
        best_val_perf = test_perf = 100000
        for epoch in range(1, cfg.train.epochs+1):
            start = time.time()    
            
            train_loss = train(train_loader, model, optimizer, device=cfg.device)
            train_loss, train_mae, train_output, y_train = test(train_loader, model, evaluator=evaluator, device=cfg.device,std=std)
            val_perf, valid_mae, valid_output, y_valid = test(val_loader, model, evaluator=evaluator, device=cfg.device,std=std)
            valid_loss = val_perf
            valid_r2 = r2_score(y_valid, valid_output) 
            train_r2 = r2_score(y_train, train_output)
            y_valid = np.asarray(y_valid)  
            y_train = np.asarray(y_train)  
            valid_output = np.asarray(valid_output) 
            train_output = np.asarray(train_output)

                
            pearson_coef_valid = np.corrcoef(valid_output.ravel(),y_valid.ravel())[0][1]
            pearson = pearson_coef_valid**2

            train_rmse = np.sqrt(mean_squared_error(y_train*std+mean,train_output*std+mean))
            valid_rmse = np.sqrt(mean_squared_error(y_valid*std+mean,valid_output*std+mean))

            scheduler.step()
            memory_allocated = torch.cuda.max_memory_allocated(cfg.device) // (1024 ** 2)
            memory_reserved = torch.cuda.max_memory_reserved(cfg.device) // (1024 ** 2)
            

           
           
           
            if valid_r2 > reports['valid R2']:
                if os.path.isdir(os.path.dirname(snapshot_path)) == False:
                    os.makedirs(os.path.dirname(snapshot_path))
                
                torch.save(model.state_dict(), snapshot_path+'/{}ModelParams.pkl'.format(epoch))  
                open(snapshot_path+'/model-{}_info.txt'.format(epoch), 'w').write('\n'.join(['Step:{}'.format(epoch),str(train_r2),str(valid_r2), str(train_mae), str(train_loss), str(valid_mae), 
                                                                str(valid_loss), str(valid_rmse),str(y_valid), str(valid_output),str(pearson),]))
                reports['valid R2'] = valid_r2
            if pearson > Pearson:
                if os.path.isdir(os.path.dirname(snapshot_path)) == False:
                    os.makedirs(os.path.dirname(snapshot_path))
                
                torch.save(model.state_dict(), snapshot_path+'/{}PearsonTransferModelParams.pkl'.format(epoch))  
                open(snapshot_path+'/model-{}_info.txt'.format(epoch), 'w').write('\n'.join(['Step:{}'.format(epoch),str(train_r2),str(valid_r2), str(train_mae), str(train_loss), str(valid_mae), 
                                                                str(valid_loss), str(valid_rmse),str(y_valid), str(valid_output),str(pearson),]))
                Pearson = pearson

            if val_perf < best_val_perf:
                best_val_perf = val_perf
                
            time_per_epoch = time.time() - start 

            
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                  f'train_R2: {train_r2:.4f},valid_r2:{valid_r2:.4f},valid_pearson:{pearson:.4f} ,train_rmse: {train_rmse},valid_rmse:{valid_rmse},train_mae:{train_mae},valid_mae:{valid_mae},' 
                  f'Val: {val_perf:.4f},  Seconds: {time_per_epoch:.4f}, '
                  f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.',)

            
            writer.add_scalar(f'Run{run}/train-loss', train_loss, epoch)
            writer.add_scalar(f'Run{run}/val-perf', val_perf, epoch)
            writer.add_scalar(f'Run{run}/train_mae', train_mae, epoch)
            writer.add_scalar(f'Run{run}/valid_mae', valid_mae, epoch)
            writer.add_scalar(f'Run{run}/train_rmse', train_rmse, epoch)
            writer.add_scalar(f'Run{run}/valid_rmse', valid_rmse, epoch)
           
            writer.add_scalar(f'Fold{run}/train-r2', train_r2, epoch)
            writer.add_scalar(f'Fold{run}/valid-r2', valid_r2, epoch)
            writer.add_scalar(f'Fold{run}/valid_pearson', pearson, epoch)
            
            writer.add_scalar(f'Run{run}/seconds', time_per_epoch, epoch)   
            writer.add_scalar(f'Run{run}/memory', memory_allocated, epoch)   

            torch.cuda.empty_cache() 

        time_average_epoch = time.time() - start_outer
        print(f'Run {run}, Vali: {best_val_perf}, Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')  
        test_perfs.append(test_perf)
        vali_perfs.append(best_val_perf)

    
    vali_perf = torch.tensor(vali_perfs)
    logger.info("-"*50)
    logger.info(config_string)
    
    logger.info(f'Final Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}, ' 
                f'Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
    print(f'Final Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}, ' 
                f'Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')

def run_k_fold(cfg, create_dataset, create_model, train, test,snapshot_path='path/train/ModelParas',evaluator=None,mean=None,std=None, k=10): 
    

    writer, logger, config_string = config_logger(cfg)
    dataset, transform, transform_eval = create_dataset(cfg)
    
   
    
    
    


    if hasattr(dataset, 'train_indices'):
        k_fold_indices = dataset.train_indices, dataset.test_indices
    else:
        k_fold_indices = k_fold(dataset, k)

    test_perfs = []
    test_curves = []
    test_r2_list = []
    test_mae_list = []
    test_rmse_list = []
    for fold, (train_idx, test_idx) in enumerate(zip(*k_fold_indices)):
        
        
        
        

        
        train_list = [str(i)+'\n' for i in train_idx.tolist()]
        test_list = [str(i)+'\n' for i in test_idx.tolist()]
        
        if os.path.isdir(os.path.dirname(snapshot_path)) == False:
            os.makedirs(os.path.dirname(snapshot_path))
            
        with open(snapshot_path+'/train_idx-{}_info.txt'.format(fold), 'w') as txt:
            txt.writelines(train_list)
            
        with open(snapshot_path+'/test_idx-{}_info.txt'.format(fold), 'w') as txt_1:
            txt_1.writelines(test_list)
        
        train_dataset = [dataset[i] for i in train_idx.tolist()]
        test_dataset = [dataset[i] for i in test_idx.tolist()]
        for i in test_dataset:
            i.transform = transform_eval
        for j in train_dataset:
            j.transform = transform
        
        
        test_dataset = [x for x in test_dataset]
        if (cfg.sampling.mode is None and cfg.subgraph.walk_length == 0) or (cfg.subgraph.online is False):
                train_dataset = [x for x in train_dataset]
       
        train_loader = DataLoader(train_dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers,follow_batch=['edge_attr'])
        test_loader = DataLoader(test_dataset, cfg.train.batch_size//cfg.sampling.batch_factor, shuffle=False, num_workers=cfg.num_workers,follow_batch=['edge_attr'])
       

        model = create_model(cfg).to(cfg.device)
        model.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
        scheduler = StepLR(optimizer, step_size=cfg.train.lr_patience, gamma=cfg.train.lr_decay)

        start_outer = time.time()
        best_test_perf = test_perf = 100000
        test_curve = []
        reports = {}
        reports['test R2'] = 0.0
        reports['test MAE'] = 100.0
        reports['test RMSE'] = 100.0
        for epoch in range(1, cfg.train.epochs+1):
            start = time.time()
            
            train_loss = train(train_loader, model, optimizer, device=cfg.device)
            train_loss, train_mae, train_output, y_train = test(train_loader, model, evaluator=evaluator, device=cfg.device,std=std)
            test_perf, test_mae, test_output, y_test = test(test_loader, model, evaluator=evaluator, device=cfg.device,std=std)
            test_loss = test_perf
            test_r2 = r2_score(y_test, test_output) 
            train_r2 = r2_score(y_train, train_output)
            y_test = np.asarray(y_test)  
            y_train = np.asarray(y_train)  
            test_output = np.asarray(test_output) 
            train_output = np.asarray(train_output)
            pearson_coef_test = np.corrcoef(test_output.ravel(),y_test.ravel())[0][1]
            
            train_rmse = np.sqrt(mean_squared_error(y_train*std+mean,train_output*std+mean))
            test_rmse = np.sqrt(mean_squared_error(y_test*std+mean,test_output*std+mean))
            scheduler.step()
            memory_allocated = torch.cuda.max_memory_allocated(cfg.device) // (1024 ** 2)
            memory_reserved = torch.cuda.max_memory_reserved(cfg.device) // (1024 ** 2)

            if test_r2 > reports['test R2']:
                if os.path.isdir(os.path.dirname(snapshot_path)) == False:
                    os.makedirs(os.path.dirname(snapshot_path))
                
                torch.save(model.state_dict(), snapshot_path+'/{}_{}ModelParams.pkl'.format(fold,epoch))  
                open(snapshot_path+'/model-{}_{}info.txt'.format(fold,epoch), 'w').write('\n'.join(['Step:{}'.format(epoch),str(train_r2),str(test_r2), str(train_mae), str(train_loss), str(test_mae), 
                                                                str(test_loss), str(test_rmse),str(y_test), str(test_output),str(pearson_coef_test)]))
                reports['test R2'] = test_r2
                reports['test MAE'] = test_mae
                reports['test RMSE'] = test_rmse
            
            
            test_curve.append(test_loss)
            best_test_perf = test_loss if test_perf < best_test_perf else best_test_perf
  
            time_per_epoch = time.time() - start 

            
            print(f'Epoch/Fold: {epoch:03d}/{fold}, Train Loss: {train_loss:.4f},Train R2: {train_r2:.4f}, Train MAE: {train_mae:.4f},Train RMSE: {train_rmse:.4f},'
                  f'Test:{test_perf:.4f}, Best-Test: {best_test_perf:.4f}, Seconds: {time_per_epoch:.4f},Test-R2{test_r2:.4f},Test-MAE{test_mae:.4f}, Test-RMSE{test_rmse:.4f},Test-Pearson{pearson_coef_test:.4f},'
                  f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')

            
            writer.add_scalar(f'Fold{fold}/train-loss', train_loss, epoch)
            writer.add_scalar(f'Fold{fold}/train-r2', train_r2, epoch)
            writer.add_scalar(f'Fold{fold}/train-mae', train_mae, epoch)
            writer.add_scalar(f'Fold{fold}/train-rmse', train_rmse, epoch)
            writer.add_scalar(f'Fold{fold}/test-perf', test_loss, epoch)
            writer.add_scalar(f'Fold{fold}/test-r2', test_r2, epoch)
            writer.add_scalar(f'Fold{fold}/test-mae', test_mae, epoch)
            writer.add_scalar(f'Fold{fold}/test-rmse', test_rmse, epoch)
            writer.add_scalar(f'Fold{fold}/test-pearson', pearson_coef_test, epoch)
            writer.add_scalar(f'Fold{fold}/test-best-perf', best_test_perf, epoch)
            writer.add_scalar(f'Fold{fold}/seconds', time_per_epoch, epoch)   
            writer.add_scalar(f'Fold{fold}/memory', memory_allocated, epoch)   

            torch.cuda.empty_cache() 

        time_average_epoch = time.time() - start_outer
        print(f'Fold {fold}, Test: {best_test_perf}, Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
        test_perfs.append(best_test_perf)
        test_r2_list.append(reports['test R2'])
        test_mae_list.append(reports['test MAE'])
        test_rmse_list.append(reports['test RMSE'])
        test_curves.append(test_curve)

    logger.info("-"*50)
    logger.info(config_string)
    test_perf = torch.tensor(test_perfs)
    test_r2_tensor = torch.tensor(test_r2_list)
    test_mae_tensor = torch.tensor(test_mae_list)
    test_rmse_tensor = torch.tensor(test_rmse_list)
    logger.info(" ===== Final result 1, based on average of max validation  ========")
    print(" ===== Final result 1, based on average of max validation  ========")
    msg = (
        f'Dataset:        {cfg.dataset}\n'
        f'Accuracy:       {test_perf.mean():.4f} ± {test_perf.std():.4f}\n'
        f'Test_R2:{test_r2_tensor.mean():.4f} ± {test_r2_tensor.std():.4f}\n'
        f'Test_MAE:{test_mae_tensor.mean():.4f} ± {test_mae_tensor.std():.4f}\n'
        f'Test_RMSE:{test_rmse_tensor.mean():.4f} ± {test_rmse_tensor.std():.4f}\n'
        f'Test_r2_list:{test_r2_tensor}\n',
        f'Test_mae_list:{test_mae_tensor}\n',
        f'Test_rmse_list:{test_rmse_tensor}\n',
        f'Seconds/epoch:  {time_average_epoch/cfg.train.epochs}\n'
        f'Memory Peak:    {memory_allocated} MB allocated, {memory_reserved} MB reserved.\n'
        '-------------------------------\n')
    logger.info(msg)
    print(msg)  

    logger.info("-"*50)
    test_curves = torch.tensor(test_curves)
    avg_test_curve = test_curves.mean(axis=0)
    best_index = np.argmax(avg_test_curve)
    mean_perf = avg_test_curve[best_index]
    std_perf = test_curves.std(axis=0)[best_index]

    logger.info(" ===== Final result 2, based on average of validation curve ========")
    print(" ===== Final result 2, based on average of validation curve ========")
    msg = (
        f'Dataset:        {cfg.dataset}\n'
        f'Accuracy:       {mean_perf:.4f} ± {std_perf:.4f}\n'
        f'Best epoch:     {best_index}\n'
        '-------------------------------\n')
    logger.info(msg)
    print(msg)   




import random, numpy as np
import warnings
def set_random_seed(seed=0, cuda_deterministic=True):
    """
    This function is only used for reproducbility, 
    DDP model doesn't need to use same seed for model initialization, 
    as it will automatically send the initialized model from master node to other nodes. 
    Notice this requires no change of model after call DDP(model)
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('You have chosen to seed training with CUDNN deterministic setting,'
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        warnings.warn('You have chosen to seed training WITHOUT CUDNN deterministic. '
                       'This is much faster but less reproducible')

from sklearn.model_selection import KFold
def k_fold(dataset, folds=10):
    kf = KFold(folds, shuffle=True, random_state=6)  

    train_indices, test_indices = [], []
    
    ys = [graph.y.item() for graph in dataset]
    for train, test in kf.split(torch.zeros(len(dataset)), ys):
        train_indices.append(torch.from_numpy(train).to(torch.long))
        test_indices.append(torch.from_numpy(test).to(torch.long))
        
        
    return train_indices, test_indices



def count_parameters(model):
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_transfer(cfg, create_dataset, model, train, test, snapshot_path='path/train/ModelParas',evaluator=None,mean=None,std=None):
    if cfg.seed is not None:
        set_random_seed(cfg.seed)
        cfg.train.runs = 1 

    
    

    
    writer, logger, config_string = config_logger(cfg)

    
    train_loader, val_loader = create_dataset(cfg)


    test_perfs = []
    vali_perfs = []


    for run in range(1, cfg.train.runs+1):
        
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
        scheduler = StepLR(optimizer, step_size=cfg.train.lr_patience, gamma=cfg.train.lr_decay)
        
        
        reports = {}
        reports['valid R2'] = 0.0
        reports['train R2'] = 0.0
        start_outer = time.time()
        best_val_perf = test_perf = 100000
        for epoch in range(1, cfg.train.epochs+1):
            start = time.time()    
            
            train_loss = train(train_loader, model, optimizer, device=cfg.device)
            train_loss, train_mae, train_output, y_train = test(train_loader, model, evaluator=evaluator, device=cfg.device,std=std,)
            val_perf, valid_mae, valid_output, y_valid = test(val_loader, model, evaluator=evaluator, device=cfg.device,std=std,)
            valid_loss = val_perf
            y_valid = np.asarray(y_valid)  
            y_train = np.asarray(y_train)  
            valid_output = np.asarray(valid_output) 
            train_output = np.asarray(train_output)
            valid_r2 = r2_score(y_valid, valid_output) 
            train_r2 = r2_score(y_train, train_output)

            train_rmse = np.sqrt(mean_squared_error(y_train*std+mean,train_output*std+mean))
            valid_rmse = np.sqrt(mean_squared_error(y_valid*std+mean,valid_output*std+mean))

            scheduler.step()
            memory_allocated = torch.cuda.max_memory_allocated(cfg.device) // (1024 ** 2)
            memory_reserved = torch.cuda.max_memory_reserved(cfg.device) // (1024 ** 2)

            if valid_r2 > reports['valid R2']:
                if os.path.isdir(os.path.dirname(snapshot_path)) == False:
                    os.makedirs(os.path.dirname(snapshot_path))
                torch.save(model.state_dict(), snapshot_path+'/{}ModelParams.pkl'.format(epoch))  
                open(snapshot_path+'/model-{}_info.txt'.format(epoch), 'w').write('\n'.join(['Step:{}'.format(epoch),str(train_r2),str(valid_r2), str(train_mae), str(train_loss), str(valid_mae), 
                                                                str(valid_loss), str(valid_rmse),str(y_valid), str(valid_output),]))
                reports['valid R2'] = valid_r2
                
            if val_perf < best_val_perf:
                best_val_perf = val_perf

            time_per_epoch = time.time() - start 

            
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                  f'train_R2: {train_r2:.4f},valid_r2:{valid_r2:.4f},train_rmse: {train_rmse},valid_rmse:{valid_rmse},train_mae:{train_mae},valid_mae:{valid_mae},' 
                  f'Val: {val_perf:.4f},  Seconds: {time_per_epoch:.4f}, '
                  f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')

            
            writer.add_scalar(f'Run{run}/train-loss', train_loss, epoch)
            writer.add_scalar(f'Run{run}/val-perf', val_perf, epoch)
            writer.add_scalar(f'Run{run}/train_mae', train_mae, epoch)
            writer.add_scalar(f'Run{run}/valid_mae', valid_mae, epoch)
            writer.add_scalar(f'Run{run}/train_rmse', train_rmse, epoch)
            writer.add_scalar(f'Run{run}/valid_rmse', valid_rmse, epoch)
            writer.add_scalar(f'Fold{run}/train-r2', train_r2, epoch)
            writer.add_scalar(f'Fold{run}/valid-r2', valid_r2, epoch)
            writer.add_scalar(f'Run{run}/seconds', time_per_epoch, epoch)   
            writer.add_scalar(f'Run{run}/memory', memory_allocated, epoch)   

            torch.cuda.empty_cache() 

        time_average_epoch = time.time() - start_outer
        print(f'Run {run}, Vali: {best_val_perf}, Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')  
        test_perfs.append(test_perf)
        vali_perfs.append(best_val_perf)

    
    vali_perf = torch.tensor(vali_perfs)
    logger.info("-"*50)
    logger.info(config_string)
    
    logger.info(f'Final Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}, ' 
                f'Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
    print(f'Final Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}, '
                f'Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')