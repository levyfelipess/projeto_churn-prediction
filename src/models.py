import numpy as np
from scipy.stats import uniform

class BinaryRandomGuess:
    """
    Modelo do tipo "chute aleatório".

    Attributes:
        threshold (float): Limiar de decisão de classe positiva;
        is_informative (bool): Indica se o modelo deve ser informado pela proporção de amostras positivas;
        seed (int or None): Semente aleatória para reprodutibilidade.
        _uniform (scipy.stats): distribuição uniform entre 0 e 1.
    """
    def __init__(self, is_informative=False, threshold=0.5, seed=None):
        """
        Inicializa o modelo.

        Args:
            threshold (float, optional): Limiar de decisão de classe positiva;
            is_informative (bool, optional): Indica se o modelo deve ser informado pela proporção de amostras positivas;
            seed (int or None, optional): Semente aleatória para reprodutibilidade.
        """
        self.threshold = threshold
        self.is_informative = is_informative
        self.seed = seed
        self._uniform = uniform(0, 1)

    def fit(self, y=None):
        """
        Aprende a proporção de exemplos positivos, caso "is_informative = True"; caso contrário, não tem efeito.

        Args:
            y (np.array or None): Vetor do atributo de saída.
        """
        if self.is_informative:
            self.threshold = 1 - y.mean()

    def predict(self, X):
        """
        Realiza a predição:
            - Classe positiva, caso a amostra aleatória seja maior ou igual ao limiar;
            - Classe negativa, caso contrário.

        Args:
            X (np.array): Matriz com atributos de entrada.

        Returns:
            np.array: Vetor com predições no formato One-Hot.
        """
        n = X.shape[0]
        y_pred = self._uniform.rvs(size=n, random_state=self.seed)
        y_pred = np.int32(y_pred >= self.threshold)
        return y_pred

    def predict_proba(self, X):
        """
        Realiza a predição no formato do valor aleatório como probabilidade da classe positiva.

        Args:
            X (np.array): Matriz com atributos de entrada.

        Returns:
            np.array: Matriz com predições em 2 colunas:
                      - A primeira com probabilidade da classe negativa;
                      - A segunda com probabilidade da classe positiva.
        """
        n = X.shape[0]
        y_prob_pred = self._uniform.rvs(size=n, random_state=self._seed).reshape(-1, 1)
        y_prob_pred = np.concatenate((1. - y_prob_pred, y_prob_pred), axis=1)
        return y_prob_pred
