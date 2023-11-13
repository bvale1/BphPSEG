import torch
import numpy as np
import matplotlib.pyplot as plt


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


def confusion_image(pred, gt):
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
     
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    img = ax.imshow(
        confusion_array,
        cmap=plt.cm.colors.ListedColormap(colors),
        vmin=1, 
        vmax=4
    )
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[i],
                                      label=labels[i]) for i in range(4)])
    return (fig, ax, img)