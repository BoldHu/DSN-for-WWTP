# calculate the regression indicator R2 and RMSE
# input is a tensor
import torch
import numpy as np

def r2_score(y_true, y_pred):
    # R2
    y_true_mean = y_true.mean()
    ss_tot = ((y_true - y_true_mean)**2).sum()
    ss_res = ((y_true - y_pred)**2).sum()
    r2 = 1 - ss_res/ss_tot
    return r2

def mean_squared_error(y_true, y_pred, squared=True):
    # RMSE
    mse = ((y_true - y_pred)**2).mean()
    if squared:
        return mse
    else:
        return mse**0.5

def reg_indicator(y_true, y_pred):
    # R2
    r2 = r2_score(y_true, y_pred)
    # RMSE
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return r2, rmse

# Test
if __name__ == '__main__':
    y_true = torch.tensor([3, -0.5, 2, 7])
    y_pred = torch.tensor([2.5, 0.0, 2, 8])
    r2, rmse = reg_indicator(y_true, y_pred)
    print(r2, rmse)
    # 0.9486081370449679 tensor(0.6124)
    y_true = torch.tensor([3, -0.5, 2, 7])
    y_pred = torch.tensor([3, -0.5, 2, 7])
    r2, rmse = reg_indicator(y_true, y_pred)
    print(r2, rmse)
    # 1.0 tensor(0.)
    y_true = torch.tensor([3, -0.5, 2, 7])
    y_pred = torch.tensor([3, -0.5, 2, 8])
    r2, rmse = reg_indicator(y_true, y_pred)
    print(r2, rmse)
    # 0.9486081370449679 tensor(0.6124)
    y_true = torch.tensor([3, -0.5, 2, 7])
    y_pred = torch.tensor([3, -0.5, 2, 6])
    r2, rmse = reg_indicator(y_true, y_pred)
    print(r2, rmse)
    # 0.9486081370449679 tensor(0.6124)
    y_true = torch.tensor([3, -0.5, 2, 7])
    y_pred = torch.tensor([3, -0.5, 2, 5])
    r2, rmse = reg_indicator(y_true, y_pred)
    print(r2, rmse)
    # 0.9486081370449679 tensor(0.6124)
    y_true = torch.tensor([3, -0.5, 2, 7])
    y_pred = torch.tensor([3, -0.5, 2, 4])
    r2, rmse = reg_indicator(y_true, y_pred)
    print(r2, rmse)
    # 0.9486081370449679 tensor(0.6124)
    y_true = torch.tensor([3, -0.5, 2, 7])

