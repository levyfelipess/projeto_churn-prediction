import numpy as np
from numpy import format_float_positional as ffp
import pandas as pd
import joblib
from datetime import timedelta

def highlight_nthmax(nth_max=1):
    """
    Retorna uma função para realce do n-ésimo máximo.
    
    Args:
        nth_max (int, optional): N-ésimo valor máximo requerido para realce.

    Returns:
        function: Função que aplicará o realce.

    Notes:
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

    Notes:
        As chaves do dicionário de saída serão os nomes dos arquivos sem formato.
    """
    models = {}
    for name in models_names:
        models[name] = joblib.load(path + name + format)
    return models

def display_elapsed_time(final_time, initial_time, elapsed_time=None):
    """
    Exibe o tempo decorrido durante um processo.

    Args:
        final_time (float): Marcação de tempo ao final do processo, em segundos;
        initial_time (float): Marcação de tempo ao início do processo, em segundos;
        elapsed_time (float or None, optional): Tempo total transcorrido, em segundos.

    Notes:
        Apenas um, entre 'elapsed_time' e '(final_time, initial_time)' necessita ser passado;
        Sendo ambos fornecidos, a preferência será para 'elapsed_time'.
    """
    if elapsed_time != None:
        dt_s = elapsed_time
    else:
        dt_s = final_time - initial_time
    h, m, s = str(timedelta(seconds=dt_s)).split(':')
    dt_min = dt_s / 60
    dt_h = dt_min / 60
    print(f"Tempo Total: {dt_s:.0f} s | {dt_min:.1f} min | {dt_h:.1f} h")
    print(f"             "+h+"h "+m+"min "+s[:2]+"s")
