import numpy as np
import pandas as pd
import scipy.stats as st

file = 'wandb_export_2025-03-18T12_28_37.136+00_00.csv'

models = ['RF_features_binary', 'RF_features_regression', 
          'XGB_features_binary', 'XGB_features_regression',
          'mlp_features_binary', 'mlp_features_regression',
          'Unet_features_binary', 'Unet_features_regression',
          'Unet_images_binary', 'Unet_images_regression',
          'deeplabv3_resnet101_features_binary', 'deeplabv3_resnet101_features_regression',
          'deeplabv3_resnet101_images_binary', 'deeplabv3_resnet101_images_regression',
          'segformerb5_features_binary', 'segformerb5_features_regression',
          'segformerb5_images_binary', 'segformerb5_images_regression']

models_binary = [model for model in models if 'binary' in model]
models_regression = [model for model in models if 'regression' in model]

models_dict = {model : 0.0 for model in models_binary}
IoUs_dict = {'noise_std0': models_dict.copy(), 'noise_std2': models_dict.copy(), 'noise_std6': models_dict.copy()}


# load dataframe and convert to numpy array
df = pd.read_csv(file).to_numpy()

for model in models_binary:
    IoU_means = []; IoU_95CIs = []; Acc_means = []; Acc_95CIs = []
    for noise_lvl in ['noise_std0', 'noise_std2', 'noise_std6']:
        print(f'========== Noise level: {noise_lvl}, model: {model} ==========')
        data = df[df[:, 1] == noise_lvl]
        data = data[data[:, 0] == model]
        IoUs = data[:5, 6]
        print(f'IoUs: {IoUs}, len: {len(IoUs)}')
        IoU_mean = np.mean(IoUs)
        # interval for 95% confidence
        IoU_95CI = st.t.interval(0.95, len(IoUs)-1, loc=IoU_mean, scale=st.sem(IoUs))
        IoU_95CI = abs(IoU_95CI[0] - IoU_95CI[1])/2
        print(f'IoU mean: {IoU_mean}, 95% CI: {IoU_95CI}')
        if np.isnan(IoU_95CI):
            IoU_95CI = 0.0
        IoU_means.append(IoU_mean)
        IoU_95CIs.append(IoU_95CI)
        IoUs_dict[noise_lvl][model] = IoU_mean
        
        Accuracies = data[:5, 5]
        print(f'Accuracies: {Accuracies}, len: {len(Accuracies)}')
        Acc_mean = np.mean(Accuracies)
        # interval for 95% confidence
        Acc_95CI = st.t.interval(0.95, len(Accuracies)-1, loc=Acc_mean, scale=st.sem(Accuracies))
        Acc_95CI = abs(Acc_95CI[0] - Acc_95CI[1])/2
        print(f'Acc mean: {Acc_mean}, 95% CI: {Acc_95CI}')
        if np.isnan(Acc_95CI):
            Acc_95CI = 0.0
        Acc_means.append(Acc_mean)
        Acc_95CIs.append(Acc_95CI)
    #print(f'{Acc_means[0]:.3f} \\pm {Acc_95CIs[0]:.3f} & {IoU_means[0]:.3f} \\pm {IoU_95CIs[0]:.3f} & {Acc_means[1]:.3f} \\pm {Acc_95CIs[1]:.3f} & {IoU_means[1]:.3f} \\pm {IoU_95CIs[1]:.3f} & {Acc_means[2]:.3f} \\pm {Acc_95CIs[2]:.3f} & {IoU_means[2]:.3f} \\pm {IoU_95CIs[2]:.3f}')
    print(f'{Acc_means[0]:.3f} & {IoU_means[0]:.3f} & {Acc_means[1]:.3f} & {IoU_means[1]:.3f} & {Acc_means[2]:.3f} & {IoU_means[2]:.3f}')
    print(f'$\\pm$ {Acc_95CIs[0]:.3f} & $\\pm$ {IoU_95CIs[0]:.3f} & $\\pm$ {Acc_95CIs[1]:.3f} & $\\pm$ {IoU_95CIs[1]:.3f} & $\\pm$ {Acc_95CIs[2]:.3f} & $\\pm$ {IoU_95CIs[2]:.3f}')
    print('\n \n')

models_dict = {model : 0.0 for model in models_regression}
RMSEs_dict = {'noise_std0': models_dict.copy(), 'noise_std2': models_dict.copy(), 'noise_std6': models_dict.copy()}
R2s_dict = {'noise_std0': models_dict.copy(), 'noise_std2': models_dict.copy(), 'noise_std6': models_dict.copy()}
    
for model in models_regression:
    RMSE_means = []; RMSE_95CIs = []; R2_means = []; R2_95CIs = []
    for noise_lvl in ['noise_std0', 'noise_std2', 'noise_std6']:
        print(f'========== Noise level: {noise_lvl}, model: {model} ==========')
        data = df[df[:, 1] == noise_lvl]
        data = data[data[:, 0] == model]
        RMSEs = np.sqrt(data[:5, 7].astype(np.float32)) * 1e-3 * 1e9  # [mol m^-3] -> [1e-9 M]
        print(f'RMSEs: {RMSEs}, len: {len(RMSEs)}')
        RMSE_mean = np.mean(RMSEs)
        # interval for 95% confidence
        RMSE_95CI = st.t.interval(0.95, len(RMSEs)-1, loc=RMSE_mean, scale=st.sem(RMSEs))
        RMSE_95CI = abs(RMSE_95CI[0] - RMSE_95CI[1])/2
        print(f'RMSE mean: {RMSE_mean}, 95% CI: {RMSE_95CI}')
        if np.isnan(RMSE_95CI):
            RMSE_95CI = 0.0
        RMSE_means.append(RMSE_mean)
        RMSE_95CIs.append(RMSE_95CI)
        RMSEs_dict[noise_lvl][model] = RMSE_mean
        
        R2s = data[:5, 8]
        print(f'R2s: {R2s}, len: {len(R2s)}')
        R2_mean = np.mean(R2s)
        # interval for 95% confidence
        R2_95CI = st.t.interval(0.95, len(R2s)-1, loc=R2_mean, scale=st.sem(R2s))
        R2_95CI = abs(R2_95CI[0] - R2_95CI[1])/2
        print(f'R2 mean: {R2_mean}, 95% CI: {R2_95CI}')
        if np.isnan(R2_95CI):
            R2_95CI = 0.0
        R2_means.append(R2_mean)
        R2_95CIs.append(R2_95CI)
        R2s_dict[noise_lvl][model] = R2_mean
    #print(f'{RMSE_means[0]:.3g} \\pm {RMSE_95CIs[0]:.3f} & {R2_means[0]:.3g} \\pm {R2_95CIs[0]:.3f} & {RMSE_means[1]:.3g} \\pm {RMSE_95CIs[1]:.3f} & {R2_means[1]:.3g} \\pm {R2_95CIs[1]:.3f} & {RMSE_means[2]:.3g} \\pm {RMSE_95CIs[2]:.3f} & {R2_means[2]:.3g} \\pm {R2_95CIs[2]:.3f}')
    print(f'{RMSE_means[0]:.3f} & {R2_means[0]:.3f} & {RMSE_means[1]:.3f} & {R2_means[1]:.3f} & {RMSE_means[2]:.3f} & {R2_means[2]:.3f}')
    print(f'$\\pm$ {RMSE_95CIs[0]:.3f} & $\\pm$ {R2_95CIs[0]:.3f} & $\\pm$ {RMSE_95CIs[1]:.3f} & $\\pm$ {R2_95CIs[1]:.3f} & $\\pm$ {RMSE_95CIs[2]:.3f} & $\\pm$ {R2_95CIs[2]:.3f}')
    print('\n \n')
    

print('\n RMSE drops as result of training with images instead of features across datasets 1, 2 and 3 respectively:')
for noise_lvl in ['noise_std0', 'noise_std2', 'noise_std6']:
    print(f'========== Noise level: {noise_lvl} ==========')
    mean_features_RMSE = 0.0
    for model in ['Unet_features_regression','deeplabv3_resnet101_features_regression','segformerb5_features_regression']:
        mean_features_RMSE += RMSEs_dict[noise_lvl][model]
    mean_features_RMSE /= 3
        
    mean_images_RMSE = 0.0
    for model in ['Unet_images_regression','deeplabv3_resnet101_images_regression','segformerb5_images_regression']:
        mean_images_RMSE += RMSEs_dict[noise_lvl][model]
    mean_images_RMSE /= 3        
    
    try:
        print(f'percent RMSE drop: {((mean_features_RMSE - mean_images_RMSE)/mean_images_RMSE):.3f}')
    except Exception as e:
        breakpoint()

print('\n mean IoU and R2 for CNNs and transformers respectively')
CNN_IoUs = []; CNN_R2s = []; transformer_IoUs = []; transformer_R2s = []
for noise_lvl in ['noise_std0', 'noise_std2', 'noise_std6']:
    for model in []:
        if 'Unet' in model or 'deeplabv3_resnet101' in model:
            if 'binary' in model:
                CNN_IoUs.append(IoUs_dict[noise_lvl][model])
            elif 'regression' in model:
                CNN_R2s.append(R2s_dict[noise_lvl][model])            
        elif 'segformerb5' in model:
            if 'binary' in model:
                transformer_IoUs.append(IoUs_dict[noise_lvl][model])
            elif 'regression' in model:
                transformer_R2s.append(R2s_dict[noise_lvl][model])
    
CNN_IoUs = np.asarray(CNN_IoUs); CNN_R2s = np.asarray(CNN_R2s)
transformer_IoUs = np.asarray(transformer_IoUs); transformer_R2s = np.asarray(transformer_R2s)
print(f'mean CNN IoU: {CNN_IoUs.mean():.3f}, mean CNN R2: {CNN_R2s.mean():.3f}')
print(f' mean transformer IoU: {transformer_IoUs.mean():.3f}, mean transformer R2: {transformer_R2s.mean():.3f}')