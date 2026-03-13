import numpy as np
import matplotlib.pyplot as plt
import optuna
from preprocessing.dataloader import heatmap
import argparse, wandb
from preprocessing.sample_train_val_test_sets import *
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, log_loss
from custom_pytorch_utils.peformance_metrics import (
    BinaryTestMetricCalculator, RegressionTestMetricCalculator
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
#from sklearn.svm import NuSVC, SVR
#from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from xgboost import XGBRFClassifier, XGBRFRegressor
#from sklearn.neural_network import MLPClassifier, MLPRegressor
import h5py, logging, json, os, timeit
import torch
from custom_pytorch_utils.custom_transforms import create_dataloaders


logging.basicConfig(level=logging.INFO)


def plot_features(path):
    with h5py.File(os.path.join(path, 'dataset.h5'), 'r') as f:
        samples = list(f.keys())
        X = []
        Y = []
        bg_masks = []
        for sample in samples:
            bg_masks.append(f[sample]['bg_mask'][()])
            import matplotlib.pyplot as plt
            X.append(f[sample]['features'][()])
            Y.append(f[sample]['c_mask'][()])

    with open(os.path.join(path, 'config.json'), 'r') as f:
        data_cfg = json.load(f)

    heatmap(
        X[-1], 
        labels=data_cfg['feature_names'],
        title=sample,
        sharescale=False,
        cmap='cool'
    )
    plt.savefig(os.path.join(path, 'features.png'))
    plt.close()
    plt.imshow(Y[-1])
    plt.savefig(os.path.join(path, 'c_mask.png'))
    plt.close()
    plt.imshow(bg_masks[-1])
    plt.savefig(os.path.join(path, 'bg_mask.png'))
    plt.close()


def plot_PCA(X, Y):
    pca = Pipeline([('scaler', StandardScaler()), ('PCA', PCA())])
    X_pca = pca.fit_transform(X)
    print(f'explained variance ratio: {pca.named_steps["PCA"].explained_variance_ratio_}')
    print(f'explained variance: {pca.named_steps["PCA"].explained_variance_}')
    print(f'principal components: {pca.named_steps["PCA"].components_}')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), subplot_kw={"projection": "3d"})
    ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=Y, s=1, alpha=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.savefig('PCA.png')
    
    return pca, fig, ax


# sanity check for NaNs and Infs
def percent_of_array_is_finite(arr):
    if type(arr) == torch.Tensor:
        return 100 * torch.sum(torch.isfinite(arr)) / torch.prod(torch.tensor(arr.shape))
    elif type(arr) == np.ndarray:
        return 100 * np.sum(np.isfinite(arr)) / np.prod(arr.shape)


def suggest_xgb_params(trial, model_type):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
    }
    if model_type == 'RF':
        params['colsample_bynode'] = trial.suggest_float('colsample_bynode', 0.6, 1.0)
    return params


def build_xgb_model(model_type, task, seed, params):
    if task == 'classification':
        model_class = XGBRFClassifier if model_type == 'RF' else XGBClassifier
        return model_class(
            seed=seed,
            objective='binary:logistic',
            eval_metric='logloss',
            **params
        )
    model_class = XGBRFRegressor if model_type == 'RF' else XGBRegressor
    return model_class(
        seed=seed,
        objective='reg:squarederror',
        eval_metric='rmse',
        **params
    )


def tune_xgb_model(model_type, task, X_train, Y_train, X_val, Y_val, seed, n_trials):
    def objective(trial):
        params = suggest_xgb_params(trial, model_type)
        model = build_xgb_model(model_type, task, seed, params)
        model.fit(X_train, Y_train)

        if task == 'classification':
            Y_pred_proba = model.predict_proba(X_val)
            return log_loss(Y_val, Y_pred_proba)
        Y_pred = model.predict(X_val)
        return mean_squared_error(Y_val, Y_pred)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root_dir', type=str, default='preprocessing/20240517_BphP_cylinders_no_noise/')
    #['20240517_BphP_cylinders_noise_std6','20240502_BphP_cylinders_noise_std2','20240517_BphP_cylinders_no_noise']
    argparser.add_argument('--git_hash', type=str, default='None')
    argparser.add_argument('--input_normalisation', choices=['MinMax', 'MeanStd'], default='MinMax', help='normalisation method for input data')
    argparser.add_argument('--wandb_log', help='disable log to wandb', action='store_false')
    argparser.add_argument('--save_test_example', help='disable save test examples to wandb', action='store_false')
    argparser.add_argument('--optuna_trials', type=int, default=100, help='number of optuna trials per model')
    argparser.add_argument('--wandb_notes', type=str, default='noise_std0')
    
    args = argparser.parse_args()
    path = args.root_dir
    cfg = {}
    cfg['root_dir'] = path
    cfg['git_hash'] = args.git_hash
    

    with open(os.path.join(os.path.dirname(args.root_dir), 'config.json'), 'r') as f:
        config = json.load(f) # <- dataset config contains normalisation parameters

    wandb.login()

    # ===========================BINARY CLASSIFICATION==========================
    
    
    with open(os.path.join(os.path.dirname(args.root_dir), 'config.json'), 'r') as f:
        config = json.load(f) # <- dataset config contains normalisation parameters
    (_, _, _, dataset, train_dataset, val_dataset, test_dataset, Y_mean, normalise_y, normalise_x) = create_dataloaders(
        root_dir=args.root_dir, 
        input_type='features',
        gt_type='binary',
        normalisation_type=args.input_normalisation,
        batch_size=16,
        config=config,
    )
    
    logging.info(f'train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}')
    
    (X_train, Y_train, bg_mask_train, inclusion_mask_train, sample_names_train,
     X_val, Y_val, bg_mask_val, inclusion_mask_val, sample_names_val,
     X_test, Y_test, bg_mask_test, inclusion_mask_test, sample_names_test) = get_sklearn_train_test_sets(
        train_dataset,
        val_dataset,
        test_dataset,
    )
    
    logging.info(f'binary classification dataset loaded \n \
        X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape} \n \
        X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape} \n \
        X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}')
    
    # sanity check features
    #for i in range(X_train.shape[1]):
    #    logging.info(f'percent of X_train[:,{i}] is finite: {percent_of_array_is_finite(X_train[:,i])}')
    
    #plot_PCA(X_test, Y_test)
    
    # classifiers
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    Y_train_full = np.concatenate([Y_train, Y_val], axis=0)

    # for i, model_name in enumerate(['RF', 'XGB']):
    #     start = timeit.default_timer()
    #     best_params, best_val_loss = tune_xgb_model(
    #         model_name, 'classification', X_train, Y_train, X_val, Y_val, 1, args.optuna_trials
    #     )
    #     for seed in [1, 2, 3, 4, 5]:
    #         pipeline = Pipeline([
    #             ('clf', build_xgb_model(model_name, 'classification', seed, best_params))
    #         ])

    #         wandb.init(
    #             project='BphPSEG2', name=model_name+'_features_binary', save_code=True, reinit=True
    #         )
    #         if args.wandb_log:
    #             wandb.config.update({
    #                 'optuna_best_params': best_params,
    #                 'optuna_best_val_loss': best_val_loss
    #             }, allow_val_change=True)
    #             if args.wandb_notes:
    #                 wandb.run.notes = args.wandb_notes

    #         logging.info(f'Training {model_name} with best params: {best_params}')
    #         pipeline.fit(X_train_full, Y_train_full)
    #         Y_pred = pipeline.predict(X_test)

    #         sample_names = sample_names_test
    #         bg_metric_calc = BinaryTestMetricCalculator()
    #         bg_metric_calc(
    #             Y_test.astype(bool), Y_pred.astype(bool),
    #             sample_names, Y_mask=bg_mask_test
    #         )
    #         inclusion_metric_calc = BinaryTestMetricCalculator()
    #         inclusion_metric_calc(
    #             Y_test.astype(bool), Y_pred.astype(bool),
    #             sample_names, Y_mask=inclusion_mask_test
    #         )
    #         bg_median_metrics = bg_metric_calc.get_median_metrics()
    #         bg_all_metrics = bg_metric_calc.get_all_metrics()
    #         inclusion_median_metrics = inclusion_metric_calc.get_median_metrics()
    #         inclusion_all_metrics = inclusion_metric_calc.get_all_metrics()
    #         logging.info(f'{model_name} binary bg test metrics: {bg_median_metrics}')
    #         logging.info(f'{model_name} binary inclusion test metrics: {inclusion_median_metrics}')
    #         logging.info(f'best validation logloss: {best_val_loss}')
    #         logging.info(f'time taken: {timeit.default_timer() - start}')

    #         if args.wandb_log:
    #             bg_log = {f'bg_{k}': v for k, v in bg_median_metrics.items()}
    #             inclusion_log = {f'inclusion_{k}': v for k, v in inclusion_median_metrics.items()}
    #             wandb.log({**bg_log, **inclusion_log, 'optuna_best_val_loss': best_val_loss,
    #                     'git_hash': args.git_hash, 'seed': seed})
    #             artifact = wandb.Artifact('test_per_sample_metrics', type='dataset')
    #             with artifact.new_file('bg.json', mode='w') as f:
    #                 json.dump(bg_all_metrics, f)
    #             with artifact.new_file('inclusion.json', mode='w') as f:
    #                 json.dump(inclusion_all_metrics, f)
    #             wandb.log_artifact(artifact)

    #         if args.save_test_example:
    #             # test example idx. 6 is 'c143423.p31' when 42 is sampler seed
    #             (X, Y, _, _, _) = test_dataset[6] # ([in_channels, x, z], [out_channels, x, z])
    #             shape = Y.shape
    #             Y_pred = pipeline.predict(
    #                 X.reshape(X.shape[0], -1).numpy().T
    #             )# [x*z, in_channels]
    #             Y_pred = torch.from_numpy(Y_pred).to(dtype=torch.bool)
    #             Y_pred = torch.stack([~Y_pred, Y_pred], dim=0).reshape(shape).transpose(1, 2)
    #             (fig, ax) = dataset.plot_sample(X, Y, Y_pred, y_transform=normalise_y, x_transform=normalise_x)
    #             if args.wandb_log:
    #                 wandb.log({'test_example': wandb.Image(fig)})

    #         wandb.finish()
        
    # ================================REGRESSION================================
    
    with open(os.path.join(os.path.dirname(args.root_dir), 'config.json'), 'r') as f:
        config = json.load(f) # <- dataset config contains normalisation parameters
    
    (_, _, _, dataset, train_dataset, val_dataset, test_dataset, Y_mean, normalise_y, normalise_x) = create_dataloaders(
        args.root_dir, 'features', 'regression', args.input_normalisation, 
        16, config
    )
    logging.info(f'train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}')
    
    (X_train, Y_train, bg_mask_train, inclusion_mask_train, sample_names_train,
     X_val, Y_val, bg_mask_val, inclusion_mask_val, sample_names_val,
     X_test, Y_test, bg_mask_test, inclusion_mask_test, sample_names_test) = get_sklearn_train_test_sets(
        train_dataset,
        val_dataset,
        test_dataset,
    )
    
    logging.info(f'regression dataset loaded \n \
        X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape} \n \
        X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape} \n \
        X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}')
    
    Y_test_inv = normalise_y.inverse_numpy_flat(Y_test)

    X_train_full = np.concatenate([X_train, X_val], axis=0)
    Y_train_full = np.concatenate([Y_train, Y_val], axis=0)

    for i, model_name in enumerate(['RF', 'XGB']):
        start = timeit.default_timer()
        best_params, best_val_loss = tune_xgb_model(
            model_name, 'regression', X_train, Y_train, X_val, Y_val, 1, args.optuna_trials
        )
        for seed in [1, 2, 3, 4, 5]:
            pipeline = Pipeline([
                ('reg', build_xgb_model(model_name, 'regression', seed, best_params))
            ])
            wandb.init(
                project='BphPSEG2', name=model_name+'_features_regression', save_code=True, reinit=True
            )
            if args.wandb_log:
                wandb.config.update({
                    'optuna_best_params': best_params,
                    'optuna_best_val_loss': best_val_loss
                }, allow_val_change=True)
                if args.wandb_notes:
                    wandb.run.notes = args.wandb_notes
                    
            logging.info(f'Training {model_name} with best params: {best_params}')
            pipeline.fit(X_train_full, Y_train_full)
            Y_pred = pipeline.predict(X_test)
            Y_pred_inv = normalise_y.inverse_numpy_flat(Y_pred)

            sample_names = sample_names_test
            bg_metric_calc = RegressionTestMetricCalculator()
            bg_metric_calc(
                Y_test_inv, Y_pred_inv,
                sample_names, Y_mask=bg_mask_test
            )
            inclusion_metric_calc = RegressionTestMetricCalculator()
            inclusion_metric_calc(
                Y_test_inv, Y_pred_inv,
                sample_names, Y_mask=inclusion_mask_test
            )
            bg_median_metrics = bg_metric_calc.get_median_metrics()
            bg_all_metrics = bg_metric_calc.get_all_metrics()
            inclusion_median_metrics = inclusion_metric_calc.get_median_metrics()
            inclusion_all_metrics = inclusion_metric_calc.get_all_metrics()
            logging.info(f'{model_name} regression bg test metrics: {bg_median_metrics}')
            logging.info(f'{model_name} regression inclusion test metrics: {inclusion_median_metrics}')
            logging.info(f'best validation MSE: {best_val_loss}')
            logging.info(f'time taken: {timeit.default_timer() - start}')

            if args.wandb_log:
                bg_log = {f'bg_{k}': v for k, v in bg_median_metrics.items()}
                inclusion_log = {f'inclusion_{k}': v for k, v in inclusion_median_metrics.items()}
                wandb.log({**bg_log, **inclusion_log, 'optuna_best_val_loss': best_val_loss,
                        'git_hash': args.git_hash, 'seed': seed})
                artifact = wandb.Artifact('test_per_sample_metrics', type='dataset')
                with artifact.new_file('bg.json', mode='w') as f:
                    json.dump(bg_all_metrics, f)
                with artifact.new_file('inclusion.json', mode='w') as f:
                    json.dump(inclusion_all_metrics, f)
                wandb.log_artifact(artifact)

            if args.save_test_example:
                # test example idx. 6 is 'c143423.p31' when 42 is sampler seed
                (X, Y, _, _, _) = test_dataset[6] # ([in_channels, x, z], [out_channels, x, z])
                shape = Y.shape
                Y_pred = pipeline.predict(
                    X.reshape(X.shape[0], -1).numpy().T
                )# [x*z, in_channels]
                Y_pred = torch.from_numpy(Y_pred).reshape(shape).T
                (fig, ax) = dataset.plot_sample(X, Y, Y_pred, y_transform=normalise_y, x_transform=normalise_x)
                if args.wandb_log:
                    wandb.log({'test_example': wandb.Image(fig)})

            wandb.finish()
