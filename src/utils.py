import numpy as np
import pandas as pd
import joblib

def highlight_nthmax(nth_max=1):
    """
    Retorna uma função para realce do n-ésimo máximo.
    
    Args:
        nth_max (int, optional): N-ésimo valor máximo requerido para realce.

    Returns:
        function: Função que aplicará o realce.

    Note:
        A função de saída deve ser utilizada nos objetos Styler do pandas, como argumento do método ".apply".
    """
    def highlight_max(s, props=""):
        a = np.array([''] * np.size(s), dtype='object')
        a[np.argsort(s.values)[-nth_max]] = props
        return a
    return highlight_max

def save_models(models_dict, path='../models/', format='.pkl'):
    """
    Salva modelos utilizando a chave do dicionário como nome do arquivo.

    Args:
        models_dict (dict): Dicionário que armazena os modelos;
        path (str, optional): Caminho do diretório, SEM CONTER NOME E FORMATO do arquivo;
        format (str, optional): Formato de armazenamento;
    """
    for key in models_dict.keys():
        joblib.dump(models_dict[key], path + key + format)

def load_models(models_names, path='../models/', format='.pkl'):
    """
    Carrega modelos utilizando seus nomes como chaves do dicionário.
    
    Args:
        models_names (list[str]): Nomes dos modelos/ arquivos;
        path (str, optional): Caminho do diretório, SEM CONTER NOME E FORMATO dos arquivos;
        format (str, optional): Formato de armazenamento;
        
    Returns:
        dict: Coleção de modelos.

    Note:
        As chaves do dicionário de saída serão os nomes dos arquivos sem formato.
    """
    models = {}
    for name in models_names:
        models[name] = joblib.load(path + name + format)
    return models
