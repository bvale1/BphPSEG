import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def confusion(pred, gt):
    # evaluates a binary classification confusion matrix
    TP = torch.sum(
        pred * gt
    ).item()
    TN = torch.sum(
        torch.logical_not(pred) * torch.logical_not(gt)
    ).item()
    FP = torch.sum(
        pred * torch.logical_not(gt)
    ).item()
    FN = torch.sum(
        torch.logical_not(pred) * gt
    )
    
    return [[TP, FN], [FP, TN]]


def perfomance_metrics(pred, gt, output=True):
    # evaluates a bunch of binary classification performance metrics
    
    # get binary confusion matrix
    [[TP, FN], [FP, TN]] = confusion(
        pred,
        gt
    )
    
    # sensitivity, also known as recall and true positive rate
    sens = TP / (TP + FN)
    # specificity, alse known as selectivity and true negative rate
    sele = TN / (FP + TN)
    
    # F1 score
    F1 = 2*TP / (2*TP + FP + FN)
    # Intersection Over Union, also referred to as Jaccard index
    IOU = TP / (TP + FN + FP)
    # Overall Accuracy, in clustering also referred to as Rand index
    OA = (TP + TN) / (TP + TN + FP + FN)
    # Matthews Correlation Coefficient or Phi Coefficient
    # equal to Pearson Correlation Coefficient for binary classification
    MCC = (TP*TN - FP+FN) / (((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**(0.5))
    
    if output == True:
        print('binary classification confusion matrix:')
        print(f'[[TP={TP}, FN={FN}],\n [FP={FP}, TN={TN}]]')
        print('performance metrics:')
        print(f'sensitivity={sens}\nselectivity={sele}')
        print(f'F1={F1}\nIOU={IOU}\nOA={OA}\nMCC={MCC}')
    
    return [F1, IOU, OA, MCC]


def confusion_image(pred, gt, fig=None, ax=None, extent=None):
    if type(pred) == torch.Tensor:
        pred = pred.detach().numpy()
    if type(gt) == torch.Tensor:
        gt = gt.detach().numpy()
    shape = np.shape(pred)
    if len(shape) == 1:
        pred = np.reshape(pred, (int(shape[0]**(0.5)), int(shape[0]**(0.5))))
    shape = np.shape(gt)
    if len(shape) == 1:
        gt = np.reshape(gt, (int(shape[0]**(0.5)), int(shape[0]**(0.5))))
    confusion_array = np.stack([
        pred * gt, # TP
        np.logical_not(pred) * np.logical_not(gt), # TN
        pred * np.logical_not(gt), # FP
        np.logical_not(pred) * gt], # FN
        axis=0
    )
    confusion_array = np.sum(confusion_array * np.arange(1,5).reshape(4, 1, 1), 
                             axis=0)
    labels = [   'TP'    ,  'TN'  ,  'FP'  ,   'FN'  ]
    colors = ['limegreen', 'white', 'black', 'salmon']
    #colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    if ax == None and fig == None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    
    img = ax.imshow(
        confusion_array,
        cmap=plt.cm.colors.ListedColormap(colors),
        vmin=1, 
        vmax=4,
        extent=extent
    )
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[i],
                                    label=labels[i]) for i in range(4)])
        
    return (fig, ax, img)

class RegressionTestMetricCalculator():
    # class to evaluate test metrics over the entire test set, which is passed
    # through in batches
    def __init__(self) -> None:
        self.metrics = {
            'sample_names' : [],
            'RMSE' : [],
            'MAE' : [],
            'R2' : []
        }        
    
    def __call__(self, Y : torch.Tensor | np.ndarray, Y_pred : torch.Tensor | np.ndarray,
                 sample_names: list[str], Y_transform=None, Y_mask=None) -> None:
        
        if type(Y) == torch.Tensor: # case pytorch model used for preditions
            Y = Y.detach().cpu()
            Y_pred = Y_pred.detach().cpu()
        else: # case xgboost model used for predictions
            Y = torch.from_numpy(Y).reshape(-1, 256, 256)
            Y_pred = torch.from_numpy(Y_pred).reshape(-1, 256, 256)
        
        b = Y.shape[0]
        Y = Y.view(b, -1) # [b, c*h*w]
        Y_pred = Y_pred.view(b, -1) # [b, c*h*w]
        if Y_transform:
                Y = Y_transform.inverse(Y)
                Y_pred = Y_transform.inverse(Y_pred)
                
        if type(Y_mask) == torch.Tensor:
            Y_mask = Y_mask.detach().cpu().view(b, -1) # [b, c*h*w]
        elif type(Y_mask) == np.ndarray:
            Y_mask = torch.from_numpy(Y_mask).view(b, -1) # [b, c*h*w]
        
        if type(Y_mask) == torch.Tensor:
            Y_mask_sum = Y_mask.sum(axis=1, keepdims=True) # [b, 1]
            # [b, c*h*w] * [b, c*h*w] = [b, c*h*w] -> [b, 1]
            RMSE = torch.sqrt((((Y - Y_pred)*Y_mask)**2).sum(dim=1, keepdim=True) / Y_mask_sum)
            MAE = torch.abs((Y - Y_pred)*Y_mask).sum(dim=1, keepdim=True) / Y_mask_sum
            mean_Y = (Y*Y_mask).sum(dim=1, keepdim=True) / Y_mask_sum
            SSr = (((Y - Y_pred)**2)*Y_mask).sum(dim=1, keepdim=True) # sum of squares of residuals
            SSt = (((Y - mean_Y)**2)*Y_mask).sum(dim=1, keepdim=True) # total sum of squares
        else:
            # [b, c*h*w] * [b, c*h*w] = [b, c*h*w] -> [b, 1]
            RMSE = torch.sqrt(torch.mean((Y - Y_pred)**2, dim=1, keepdim=True))
            MAE = torch.mean(torch.abs(Y - Y_pred), dim=1, keepdim=True)
            mean_Y = torch.mean(Y, dim=1, keepdim=True)
            SSr = torch.sum((Y - Y_pred)**2, dim=1, keepdim=True) # sum of squares of residuals
            SSt = torch.sum((Y - mean_Y)**2, dim=1, keepdim=True) # total sum of squares
        R2 = 1 - (SSr / SSt)
        
        self.metrics['sample_names'] += sample_names
        self.metrics['RMSE'] += RMSE.reshape(-1).tolist()
        self.metrics['MAE'] += MAE.reshape(-1).tolist()
        self.metrics['R2'] += R2.reshape(-1).tolist()
        
        
    def get_median_metrics(self) -> dict:
        return {
            'median_RMSE' : np.nanmedian(np.asarray(self.metrics['RMSE'])),
            'median_MAE' : np.nanmedian(np.asarray(self.metrics['MAE'])),
            'median_R2' : np.nanmedian(np.asarray(self.metrics['R2'])),
        }
    
    def get_all_metrics(self):
        return self.metrics


class BinaryTestMetricCalculator():
    # class to evaluate binary classification test metrics over the entire test
    # set, which is passed through in batches
    def __init__(self) -> None:
        self.metrics = {
            'sample_names' : [],
            'Dice'        : [],
            'IOU'         : [],
            'MCC'         : [],
            'Sensitivity' : [],
            'Specificity' : [],
            'Accuracy'    : []
        }

    def __call__(self, Y : torch.Tensor | np.ndarray, Y_pred : torch.Tensor | np.ndarray,
                 sample_names: list[str], Y_mask=None) -> None:

        if type(Y) == torch.Tensor:
            Y = Y.detach().cpu()
            Y_pred = Y_pred.detach().cpu()
        else: # xgboost predictions are flat numpy arrays
            Y = torch.from_numpy(Y).reshape(-1, 256, 256)
            Y_pred = torch.from_numpy(Y_pred).reshape(-1, 256, 256)

        # ensure boolean tensors
        Y = Y.bool()
        Y_pred = Y_pred.bool()

        b = Y.shape[0]
        Y = Y.view(b, -1)         # [b, h*w]
        Y_pred = Y_pred.view(b, -1) # [b, h*w]

        if type(Y_mask) == torch.Tensor:
            Y_mask = Y_mask.detach().cpu().bool().view(b, -1)  # [b, h*w]
        elif type(Y_mask) == np.ndarray:
            Y_mask = torch.from_numpy(Y_mask).bool().view(b, -1)  # [b, h*w]

        if Y_mask is not None:
            TP = ( Y_pred &  Y & Y_mask).sum(dim=1).float()  # [b]
            TN = (~Y_pred & ~Y & Y_mask).sum(dim=1).float()
            FP = ( Y_pred & ~Y & Y_mask).sum(dim=1).float()
            FN = (~Y_pred &  Y & Y_mask).sum(dim=1).float()
        else:
            TP = ( Y_pred &  Y).sum(dim=1).float()
            TN = (~Y_pred & ~Y).sum(dim=1).float()
            FP = ( Y_pred & ~Y).sum(dim=1).float()
            FN = (~Y_pred &  Y).sum(dim=1).float()

        eps = 1e-8
        Dice        = (2 * TP) / (2 * TP + FP + FN + eps)
        IOU         = TP / (TP + FP + FN + eps)
        Sensitivity = TP / (TP + FN + eps)
        Specificity = TN / (TN + FP + eps)
        Accuracy    = (TP + TN) / (TP + TN + FP + FN + eps)
        MCC_num     = TP * TN - FP * FN
        MCC_den     = torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps)
        MCC         = MCC_num / MCC_den

        self.metrics['sample_names'] += sample_names
        self.metrics['Dice']        += Dice.reshape(-1).tolist()
        self.metrics['IOU']         += IOU.reshape(-1).tolist()
        self.metrics['MCC']         += MCC.reshape(-1).tolist()
        self.metrics['Sensitivity'] += Sensitivity.reshape(-1).tolist()
        self.metrics['Specificity'] += Specificity.reshape(-1).tolist()
        self.metrics['Accuracy']    += Accuracy.reshape(-1).tolist()

    def get_median_metrics(self) -> dict:
        return {
            'median_Dice'        : np.nanmedian(np.asarray(self.metrics['Dice'])),
            'median_IOU'         : np.nanmedian(np.asarray(self.metrics['IOU'])),
            'median_MCC'         : np.nanmedian(np.asarray(self.metrics['MCC'])),
            'median_Sensitivity' : np.nanmedian(np.asarray(self.metrics['Sensitivity'])),
            'median_Specificity' : np.nanmedian(np.asarray(self.metrics['Specificity'])),
            'median_Accuracy'    : np.nanmedian(np.asarray(self.metrics['Accuracy']))
        }

    def get_all_metrics(self):
        return self.metrics