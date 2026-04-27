import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class ZScoreNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalização Z-Score que utiliza desvio padrão amostral.

    Attributes:
        ddof (int): Graus de liberdade no cálculo do desvio padrão;
        mean_vec (np.array): Vetor de médias dos atributos;
        std_vec (np.array): Vetor de desvios padrões dos atributos.

    Notes:
        A classe pai BaseEstimator proporciona compatabilidade com outros métodos em sklearn;
        A classe pai TransformerMixin adiciona o método ".fit_transform" automaticamente.
    """
    def __init__(self, ddof=1):
        """
        Inicializa o Normalizador.

        Args:
            ddof (int, optional): Graus de liberdade no cálculo do desvio padrão.
        """
        self.mean_vec = None
        self.std_vec = None
        self.ddof = ddof

    def fit(self, X, y=None):
        """
        Aprende os parâmetros da normalização.

        Args:
            X (np.array): Matriz de padrões e atributos, no formato "n x d";
            y (None): Apenas para compatibilização com outros métodos em sklearn.
        """
        self.mean_vec = np.mean(X, axis=0)
        self.std_vec = np.std(X, axis=0, ddof=self.ddof)
        return self

    def transform(self, X):
        """
        Aplica a normalização.

        Args:
            X (np.array): Matriz de padrões e atributos, no formato "n x d".

        Returns:
            np.array: Matriz "X" normalizada.
        """
        return (X - self.mean_vec) / self.std_vec

    def inverse_transform(self, X):
        """
        Aplica o inverso da normalização.

        Args:
            X (np.array): Matriz de padrões e atributos normalizada, no formato "n x d".

        Returns:
            np.array: Matrix "X" com médias e desvios padrões originais.
        """
        return (X * self.std_vec) + self.mean_vec

def clean_data(df):
    """
    Aplica todas as transformações necessárias ao dataset original para torná-lo pronto para modelagem.

    Args:
        df (pd.DataFrame): Dataset original (raw).

    Returns:
        pd.DataFrame: Dataset tratado (processed).

    Notes:
        As transformações para o dataset deste problema incluem:
            1. Retirada da coluna de ID "customerID";
            2. Preenchimento de valores faltantes em "TotalCharges";
            3. Conversão do tipo de "SeniorCitizen", "tenure" e "TotalCharges".
    """
    df = df.drop(columns='customerID')

    col_missing_values = np.where(df['TotalCharges'] == ' ')[0]
    df.loc[col_missing_values, 'TotalCharges'] = df.loc[col_missing_values, 'MonthlyCharges']

    df = df.astype({'SeniorCitizen':'object', 'tenure':'float64', 'TotalCharges':'float64'})
    return df

def get_transformers(df_X, df_y):
    """
    Obtém as transformações específicas necessárias aos dados, para cada modelo.

    Args:
        df_X (pd.DataFrame): Dataset completo dos atributos de entrada;
        df_y (pd.DataFrame or pd.Series): Dataset  completo do atributo de saída.

    Returns:
        dict: Coleção das transformações nos atributos de saída, para cada modelo;
        OneHotEncoder: Transformação one-hot para o atributo de saída.
    """
    
    transformers_X = {
    'LR':ColumnTransformer(
        [('OneHot', OneHotEncoder(sparse_output=False, drop='first'), df_X.dtypes[df_X.dtypes == 'object'].index),
         ('ZScore', ZScoreNormalizer(), df_X.dtypes[df_X.dtypes == 'float64'].index)], remainder='passthrough'),
    'KNN':ColumnTransformer(
        [('OneHot', OneHotEncoder(sparse_output=False, drop='first'), df_X.dtypes[df_X.dtypes == 'object'].index),
         ('ZScore', ZScoreNormalizer(), df_X.dtypes[df_X.dtypes == 'float64'].index)], remainder='passthrough'),
    'SVM':ColumnTransformer(
        [('OneHot', OneHotEncoder(sparse_output=False, drop='first'), df_X.dtypes[df_X.dtypes == 'object'].index),
         ('ZScore', ZScoreNormalizer(), df_X.dtypes[df_X.dtypes == 'float64'].index)], remainder='passthrough'),
    'DT':ColumnTransformer(
        [('OneHot', OneHotEncoder(sparse_output=False), df_X.dtypes[df_X.dtypes == 'object'].index)], remainder='passthrough'),
    'RF':ColumnTransformer(
        [('OneHot', OneHotEncoder(sparse_output=False), df_X.dtypes[df_X.dtypes == 'object'].index)], remainder='passthrough')
    }

    transformer_y = OneHotEncoder(sparse_output=False, drop='first', categories=[['No', 'Yes']])

    return transformers_X, transformer_y
