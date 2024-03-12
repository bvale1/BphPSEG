import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def metrics_overleaf_table(
        results : dict, # dictionary to index as [model][metric]
        title : Optional[str] = 'Metrics',
        caption : Optional[str] = 'insert caption here', 
        label : Optional[str] = 'insert label here',
    ):
    # prints and returns a string of a formatted table for overleaf
    
    keys = results.keys()
    # these are not performance metrics but often included in results the dictionary
    for key in ['cfg', 'config', 'hyperparameters', 'args', 'arguments']:
        if key in keys:
            del results[key]
    
    # the same metrics must be evaluated for each model
    metrics = results[results.keys()[0]].keys()
    for metric in metrics:
        if metric.split('_')[0] != 'test':
            
    
    
    # use %{str} or {{str}} for .format() to ignore the curly braces
    tabel = '''
    \begin%{table}[H]
    \centering
    \begin%{tabular}%{|l|l|l|l|l|l|l|l|}
    \hline \hline 
    \multicolumn%{8}%{|l|}{title} \\
    \hline
    Model & F1($\uparrow$) & Accuracy($\uparrow$) & Precision($\uparrow$) & Recall($\uparrow$) & Specificity($\uparrow$) & MCC($\uparrow$) & IoU($\uparrow$) \\
    \hline
    KNN & 0.5270 & 0.8687 & 0.3807 & 0.8558 & 0.8699 & 0.5150 & 0.3578 \\
    SVM & 0.4645 & 0.8406 & 0.3257 & 0.8093 & 0.8435 & 0.4462 & 0.3025 \\
    RF & 0.6561 & 0.9204 & 0.5200 & 0.8887 & 0.9234 & 0.6429 & 0.4882 \\
    XGB & 0.6223 & 0.9061 & 0.4742 & 0.9049 & 0.9063 & 0.6138 & 0.4517 \\
    MLP & 0.6476 & 0.9185 & 0.5136 & 0.8760 & 0.9225 & 0.6326 & 0.4788 \\
    U-Net (features) & 0.9799 & 0.9965 & 0.9801 & \textbf{0.9797} & 0.9981 & 0.9780 & 0.9609 \\
    \textbf{U-Net (images)} & \textbf{0.9845} & \textbf{0.9977} & \textbf{0.9895} & 0.9795 & \textbf{0.9991} & \textbf{0.9832} & \textbf{0.9694} \\
    \hline \hline
    \end{tabular}
    \caption{The performance metrics of the models applied in binary classification. Where all models are evaluated on the same test dataset made up of 40 samples not included in training.}
    \label{tab:binaryPerformance}
    \end{table}
    '''.format(title)
    
    print()

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