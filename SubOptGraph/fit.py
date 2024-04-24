import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_error
import random
import numpy as np
import os
import time
from scipy.sparse import coo_matrix



def verify_dir_exists(dirname):  #
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))

def coo_format(A):
    coo_A = np.zeros([A.shape[0],A.shape[2]])
    for i in range(A.shape[1]):
        coo_A = coo_A + A[:,i,:]
    coo_A = coo_matrix(coo_A)
    edge_index = [coo_A.row, coo_A.col]
    edge_attr = []
    for j in range(len(edge_index[0])):
        edge_attr.append(A[edge_index[0][j],:,edge_index[1][j]])
    edge_attr = np.array(edge_attr)
    return edge_index, edge_attr

def func(pred):
    return pred.index(max(pred))

def train(model, train_loader, device, optimizer):
    model.train()  #train()的作用是启用Batch Normalization和 Dropout，并非函数的嵌套
    loss_all = 0
    for d in train_loader:
        data = d[0]
        data_sol = d[1]
        data = data.to(device)   #配置cuda，将数据配置到GPU上，加速运行
        data_sol = data_sol.to(device)
        
        #optimizer.zero_grad()  #使用优化器，梯度归零，属于中阶api，低阶api为手动对梯度更新后的parameter进行改变，其中@是pytorch中的矩阵乘法符号
        output = model(data,data_sol)
        #print(data.y.shape)
        loss = F.mse_loss(output, data.y)
        loss_all += loss.item() * data.num_graphs #loss.item()获取loss张量中的值
        optimizer.zero_grad() 
        loss.backward()      #求梯度
        #loss_all += loss.item() * data.num_graphs
        
        optimizer.step()     #更新所有参数
    return loss_all/len(train_loader.dataset)

def test(model, loader, device, mean, std):
    model.eval()   #不启用Batch Normalization和 Dropout，不改变权值
    error = 0
    loss_all = 0
    model_output = []
    y = []
    att_list = []
   # print(data.e_index)
    for d in loader:
        data = d[0]
        data_sol = d[1]
        data = data.to(device)
        data_sol = data_sol.to(device)
        output = model(data,data_sol)
        #pred = output.max(dim=1)[1]
        #loss = F.mse_loss(output, data.y)
        #loss = Loss(output, data.y)
       # rmse = np.sqrt(mean_squared_error(T_train,output))
        error += (output * std - data.y * std).abs().sum().item()              # tensor.item() 只用在只有一个元素的tensor
                                                                                   # tensor.detach() 从原来的计算图的张量中返回一个新的张量，不需要梯度
        loss = F.mse_loss(output, data.y)
        #print('loss: ',loss.item())
        #print(output)
        loss_all += loss.item() * data.num_graphs                           #   Returns a new Tensor, detached from the current graph. The result will never require gradient.
        model_output.extend(output.tolist())                                       # tensor.tolist() 从张量中返回一个python列表，对应的值与它相同
        y.extend(data.y.tolist())
        #tags.extend(data.tag)
        #att_list.extend(att.tolist())
    return loss_all, error/len(loader.dataset), model_output, y


def training(Model, data_loaders, patience,n_epoch=100, snapshot_path='./snapshot/',  optimizer=None, std=None, mean=None):
    '''
    data_loaders is a dict that contains Data_Loaders for training, validation and testing, respectively
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    #data = dataset[0].to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    if optimizer == None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=10, min_lr=0.000001)
    
    if len(data_loaders)>2:
        with_test = True
    else:
        with_test = False
        
    history = {}
    history['Train Loss'] = []
    history['Train Mae'] = []
    history['Train_rmse'] = []
    history['Train_r2'] = []
    history['Valid Loss'] = []
    history['Valid Mae'] = []
    history['Valid_rmse'] = []
    history['Valid_r2'] = []
    
    reports = {}
    reports['valid mae'] = 0.0
    reports['valid loss'] = float('inf')
    reports['valid R2'] = 0.0

    patience_count = 0
    valid_loss_pre = 1

    for epoch in range(1, n_epoch+1):
        start_time_1 = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']
        
        train_loss = train(model, data_loaders['train'], device, optimizer)  #系数更新
        train_loss, train_mae, train_output, y_train = test(model, data_loaders['train'], device, mean, std)
        valid_loss, valid_mae, valid_output, y_valid = test(model, data_loaders['valid'], device, mean, std)


        valid_r2 = r2_score(y_valid, valid_output)
        train_r2 = r2_score(y_train, train_output)
        y_valid = np.asarray(y_valid)
        y_train = np.asarray(y_train)
        valid_output = np.asarray(valid_output)
        train_output = np.asarray(train_output)
        train_rmse = np.sqrt(mean_squared_error(y_train*std+mean,train_output*std+mean))
        valid_rmse = np.sqrt(mean_squared_error(y_valid*std+mean,valid_output*std+mean))
        pearson_coef_valid = np.corrcoef(valid_output.ravel(),y_valid.ravel())[0][1]
        pearson = float(pearson_coef_valid**2)
        #print(pearson)
    #   train_rmse = np.sqrt((y_train-train_output)**2*(std**2)/len(train_output))
       # valid_rmse = np.sqrt((y_valid-valid_output)**2*(std**2/len(valid_output))
        scheduler.step(epoch)
        
        history['Train Loss'].append(train_loss)
        history['Train Mae'].append(train_mae)
        history['Valid Mae'].append(valid_mae)
        history['Valid Loss'].append(valid_loss)
        history['Train_rmse'].append(train_rmse)
        history['Valid_rmse'].append(valid_rmse)
        history['Train_r2'].append(train_r2)
        history['Valid_r2'].append(valid_r2)
       # history['Valid_pearson'].append(pearson)

        #if valid_mae < reports['valid loss']:
        if valid_r2 > reports['valid R2']:
            verify_dir_exists(snapshot_path)
            torch.save(model.state_dict(), snapshot_path+'/ModelParams.pkl')  #储存r2最好的参数
            open(snapshot_path+'/model-{}_info.txt'.format(epoch), 'w').write('\n'.join(['Step:{}'.format(epoch), str(train_mae), str(train_loss), str(valid_mae), 
                                                               str(valid_loss), str(y_valid), str(valid_output),str(pearson)]))
            reports['valid mae'] = valid_mae
            reports['valid loss'] = valid_loss
            reports['valid R2'] = valid_r2
            if with_test:
                test_mae, test_loss, test_output, y_test = test(model, data_loaders['test'], device, mean, std)
                open(snapshot_path+'/model-test-info.txt'.format(epoch), 'w').write('\n'.join([str(test_mae), str(y_test), str(test_output)]))
        
        #定义早停
        if epoch < 2:
            valid_loss_pre = valid_loss
        else:
            if (valid_loss >= valid_loss_pre) and (valid_loss < 145):
                patience_count += 1
            else:
                patience_count = 0
            valid_loss_pre = valid_loss
        if patience_count >= patience:
            print(f"Early Stop in epoch {epoch} with patience {patience}")
            break
                

        end_time_1 = time.time()
        elapsed_time = end_time_1 - start_time_1
        print('Epoch:{:03d} ==> LR: {:7f}, Train Mae: {:.4f}, Train RMSE:{:.5F}, Train Loss: {:.5f}, Valid Mae: {:.4f}, Valid RMSE:{:.5F}, Valid loss: {:.5f}, Valid R2: {:.5f}, Train R2: {:.5f}, Elapsed Time:{:.2f} s'.format(epoch, 
                                                                                                                                                     lr, 
                                                                                                                                                     train_mae, 
                                                                                                                                                     train_rmse,
                                                                                                                                                     train_loss, 
                                                                                                                                                     valid_mae, 
                                                                                                                                                     valid_rmse,
                                                                                                                                                     valid_loss,
                                                                                                                                                     valid_r2,
                                                                                                                                                     train_r2,
                                                                                                                                                     elapsed_time))
    open(snapshot_path+'history','w').write(str(history))
    print('\nLoss: {}'.format(reports['valid loss'] ))
