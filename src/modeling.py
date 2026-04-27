import numpy as np
from scipy.stats import sem
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

def random_search_with_kfoldcv(model,
                               df_X_train, df_y_train,
                               n_comb, k, hyperparam_distributions,
                               transformer_X, transformer_y,
                               global_seed=7, verbose_tqdm=True):
    """
    Performa otimização de hiperparâmetros por maximização da AUROC utilizando busca aleatória com validação cruzada k-fold estratificada.

    Args:
        model: Modelo;
        df_X_train (pd.DataFrame): Dataset de treinamento com atributos de entrada;
        df_y_train (pd.DataFrame or pd.Series): Dataset de treinamento com atributo de saída;
        n_comb (int): Número de combinações de hiperparâmetros a serem avaliadas;
        k (int): Número de partições da validação cruzada;
        hyperparam_distributions (dict[str, scipy.stats or list]): Coleção de distribuições dos hiperparâmetros;
        transformer_X: Transformação ou conjunto de transformações nas variáveis de entrada;
        transformer_y: Transformação ou conjunto de transformações na variável de saída;
        global_seed (int, optional): Semente aleatória a ser aplicada em todos os algoritmos estocásticos (reprodutibilidade);
        verbose_tqdm (bool, optional): Indica se barras de progresso do tqdm devem ser mostradas;

    Returns:
        pd.DataFrame: Tabela com as combinações testadas juntamente com suas métricas médias nos conjuntos de validação.

    Notes:
        Em "hyperparam_distributions", é possível passar uma lista com os elementos a serem escolhidos aleatoriamente.
        A escolha será de modo uniforme.
    """
    masks = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=global_seed).split(df_X_train, df_y_train))
    hyperparam_combinations = {}
    for key in hyperparam_distributions.keys():
        if type(hyperparam_distributions[key]) == list:
            rng = np.random.default_rng(seed=global_seed)
            hyperparam_combinations[key] = rng.choice(hyperparam_distributions[key], size=n_comb)
        else:
            hyperparam_combinations[key] = hyperparam_distributions[key].rvs(size=n_comb, random_state=global_seed)

    metrics_hyperparams = np.empty((k, n_comb))
    for i in range(k):
        est_mask, val_mask = masks[i]
        df_X_est, df_y_est = df_X_train.iloc[est_mask], df_y_train.iloc[est_mask]
        df_X_val, df_y_val = df_X_train.iloc[val_mask], df_y_train.iloc[val_mask]

        X_est_transformed = transformer_X.fit_transform(df_X_est)
        y_est = transformer_y.fit_transform(df_y_est)
        X_val_transformed = transformer_X.transform(df_X_val)
        y_val = transformer_y.transform(df_y_val)

        if verbose_tqdm:
            range_combinations = tqdm(range(n_comb), desc=f'Validação Cruzada K-Fold - {i+1}/{k}')
        else:
            range_combinations = range(n_comb)
            
        for comb in range_combinations:
            for key in hyperparam_combinations.keys():
                model.__dict__[key] = hyperparam_combinations[key][comb]

            model.fit(X=X_est_transformed,
                      y=y_est.reshape(-1))
            y_val_prob_pred = model.predict_proba(X=X_val_transformed)[:, 1]
        
            metrics_hyperparams[i][comb] = roc_auc_score(y_score=y_val_prob_pred, y_true=y_val)
            
    hyperparam_combinations['auc (mean)'] = np.mean(metrics_hyperparams, axis=0)
    hyperparam_combinations['auc (std)'] = np.std(metrics_hyperparams, axis=0, ddof=1)
    hyperparam_combinations['auc (se)'] = sem(metrics_hyperparams, axis=0, ddof=1)
    
    return pd.DataFrame(hyperparam_combinations)

def parallelized_random_search_with_kfoldcv(model,
                                            df_X_train, df_y_train,
                                            n_comb, k, hyperparam_distributions,
                                            transformer_X, transformer_y,
                                            n_jobs, global_seed=7):
    """
    Performa otimização de hiperparâmetros por maximização da AUROC utilizando busca aleatória com validação cruzada k-fold estratificada
    com a possibilidade de paralelização nos folds.

    Args:
        model: Modelo;
        df_X_train (pd.DataFrame): Dataset de treinamento com atributos de entrada;
        df_y_train (pd.DataFrame or pd.Series): Dataset de treinamento com atributo de saída;
        n_comb (int): Número de combinações de hiperparâmetros a serem avaliadas;
        k (int): Número de partições da validação cruzada;
        hyperparam_distributions (dict[str, scipy.stats or list]): Coleção de distribuições dos hiperparâmetros;
        transformer_X: Transformação ou conjunto de transformações nas variáveis de entrada;
        transformer_y: Transformação ou conjunto de transformações na variável de saída;
        n_jobs: Número de núcleos;
        global_seed (int, optional): Semente aleatória a ser aplicada em todos os algoritmos estocásticos (reprodutibilidade);

    Returns:
        pd.DataFrame: Tabela com as combinações testadas juntamente com suas métricas médias nos conjuntos de validação.

    Notes:
        Em "hyperparam_distributions", é possível passar uma lista com os elementos a serem escolhidos aleatoriamente.
        A escolha será de modo uniforme.
    """
    masks = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=global_seed).split(df_X_train, df_y_train))
    hyperparam_combinations = {}
    for key in hyperparam_distributions.keys():
        if type(hyperparam_distributions[key]) == list:
            rng = np.random.default_rng(seed=global_seed)
            hyperparam_combinations[key] = rng.choice(hyperparam_distributions[key], size=n_comb)
        else:
            hyperparam_combinations[key] = hyperparam_distributions[key].rvs(size=n_comb, random_state=global_seed)

    def parallel_kfoldcv_fn(i):
        metrics_hyperparams = np.empty(n_comb)
        
        est_mask, val_mask = masks[i]
        df_X_est, df_y_est = df_X_train.iloc[est_mask], df_y_train.iloc[est_mask]
        df_X_val, df_y_val = df_X_train.iloc[val_mask], df_y_train.iloc[val_mask]

        X_est_transformed = transformer_X.fit_transform(df_X_est)
        y_est = transformer_y.fit_transform(df_y_est)
        X_val_transformed = transformer_X.transform(df_X_val)
        y_val = transformer_y.transform(df_y_val)

        for comb in range(n_comb):
            for key in hyperparam_combinations.keys():
                model.__dict__[key] = hyperparam_combinations[key][comb]

            model.fit(X=X_est_transformed,
                      y=y_est.reshape(-1))
            y_val_prob_pred = model.predict_proba(X=X_val_transformed)[:, 1]
        
            metrics_hyperparams[comb] = roc_auc_score(y_score=y_val_prob_pred, y_true=y_val)
            
        return metrics_hyperparams
    metrics_hyperparams = np.array( Parallel(n_jobs=n_jobs)(delayed(parallel_kfoldcv_fn)(i) for i in range(k)) )
            
    hyperparam_combinations['auc (mean)'] = np.mean(metrics_hyperparams, axis=0)
    hyperparam_combinations['auc (std)'] = np.std(metrics_hyperparams, axis=0, ddof=1)
    hyperparam_combinations['auc (se)'] = sem(metrics_hyperparams, axis=0, ddof=1)
    
    return pd.DataFrame(hyperparam_combinations)
