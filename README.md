# Projeto de Previsão de *Churn* de Clientes

Projeto de Ciência de Dados focado na previsão de _churn_ em clientes de telecom, com aplicação de modelos de aprendizagem de máquina e otimização orientada a decisões de negócio.

## Problema

A evasão de clientes (_churn_) impacta diretamente a receita recorrente de empresas de telecom. Antecipar quais clientes têm essa maior propensão permite ações preventivas de retenção.

Este projeto tem como objetivo prever o _churn_ e apoiar decisões de retenção baseadas em custo-benefício.

## Metodologia

### _Dataset_
#### Links de acesso
- [Link 1: Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn);
- [Link 2: IBM](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).

#### Sobre o _dataset_
- Os dados contêm informações sobre uma empresa fictícia que forneceu serviços de telefonia fixa e internet para 7043 clientes na Califórnia (fonte: [Link 2](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113));
- Cada linha representa um cliente, e cada coluna contém os atributos deste (fonte: [Link 1](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)):
  - Clientes que cancelaram o serviço no último mês: coluna `Churn`;
  - Serviços contratados por cada cliente: telefone (coluna `PhoneService`), múltiplas linhas (coluna `MultipleLines`), internet (coluna `InternetService`), segurança online (coluna `OnlineSecurity`), backup online (coluna `OnlineBackup`), proteção de dispositivos (coluna `DeviceProtection`), suporte técnico (coluna `TechSupport`), streaming de TV (coluna `StreamingTV`) e de filmes (coluna `StreamingMovies`);
  - Informações da conta do cliente: tempo de relacionamento (coluna `tenure`), tipo de contrato (coluna `Contract`), forma de pagamento (coluna `PaymentMethod`), faturamento eletrônico (coluna `PaperlessBilling`), valores mensais (coluna `MonthlyCharges`) e valor total a pagar (coluna `TotalCharges`);
  - Informações demográficas sobre os clientes: sexo (coluna `gender`), faixa etária (coluna `SeniorCitizen`) e se possuem cônjuge (coluna `Partner`) ou dependentes (coluna `Dependents`).

### Abordagem
- Análise exploratória de dados (EDA);
- Pré-processamento de dados (tratamento, normalização e codificação);
- Comparação entre múltiplas abordagens:
  - Regressão Logística (**LR**);
  - K-Vizinhos mais Próximos (**KNN**);
  - Árvores de Decisão (**DT**);
  - Florestas Aleatórias (**RF**);
  - Máquinas de Vetores de Suporte (**SVM**);
- Otimização de hiperparâmetros com busca aleatória e validação cruzada;
- Análise de _threshold_ baseado nos custos desiguais dos erros.

## Principais Diferenciais do Projeto

### Organização e Engenharia de Projetos
- Estrutura modular organizada (`src/`, `notebooks/`, `data/`, `models/`, `reports/`);
- Separação entre código reutilizável e notebooks exploratórios;
- Persistência de modelos treinados;
- Funções e classes documentadas;
- Reprodutibilidade e rastreamento dos experimentos (Git);

### EDA
- Teste estatístico $\chi^2$ de associação entre variáveis categóricas de entrada e a saída;
  - Avaliação da intensidade de associação com coeficiente V de Cramér;
- _Insights_ de negócio;

### Pipeline de Aprendizagem de Máquina
- _Data splitting_ preservando a estratificação da classe de saída;
- Comparação entre abordagens lineares, não lineares, paramétricas, não paramétricas;
- Práticas de prevenção total a _data leakage_;
- Otimização de hiperparâmetros com busca aleatória, validação cruzada estratificada, maximização da AUROC e regra "1SE";
- Estratégias de monitoramento e proteção a _overfitting_;
- Avaliação com métricas diversas e foco nas mais adequadas levando em consideração: **1)** problema, e **2)** desbalanceamento do _dataset_;

### Análise de _Threshold_
- Ajuste de _threshold_ baseado na consideração de custos desiguais de negócio (não apenas métricas padrão);
- Comparação entre múltiplos cenários de custo (R = 2, 5, 10);

## Estrutura do Projeto
```
projeto_churn-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling_evaluation.ipynb
│   └── 03_model-comparison_threshold-tuning.ipynb
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   ├── eda.py
│   ├── evaluation.py
│   ├── modeling.py
│   ├── models.py
│   ├── preprocessing.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Como Executar o Projeto
1. Clonar o repositório:
```
git clone https://github.com/levyfelipess/projeto_churn-prediction
```
2. Entrar no diretório:
```
cd projeto_churn-prediction
```
3. Instalar dependências específicas:
```
pip install -r requirements.txt
```
E então, rodar os notebooks.

### (OPCIONAL, mas recomendado) Utilizar ambiente virtual
1. Clonar o repositório:
```
git clone https://github.com/levyfelipess/projeto_churn-prediction
```
2. Entrar no diretório:
```
cd projeto_churn-prediction
```
3. Criar ambiente virtual:
```
python -m venv .venv
```
4. Ativar ambiente virtual (Git bash):
```
source .venv/Scripts/activate
```
5. Instalar dependências específicas:
```
pip install -r requirements.txt
```
6. Criar kernel específico para o ambiente:
```
python -m ipykernel install --user --name=nome-do-kernel
```
Após, selecionar o kernel criado quando abrir os notebooks.

## Visualização dos Notebooks
Para uma experiência de visualização completa, se possível, acessar pelo NBViewer (principalmente o Notebook 3):

> [Notebook 1: EDA](https://nbviewer.org/github/levyfelipess/projeto_churn-prediction/blob/main/notebooks/01_eda.ipynb) \
> [Notebook 2: Modelagem e Avaliação](https://nbviewer.org/github/levyfelipess/projeto_churn-prediction/blob/main/notebooks/02_modeling_evaluation.ipynb) \
> [Notebook 3: Comparação entre modelos e Análise de *Threshold*](https://nbviewer.org/github/levyfelipess/projeto_churn-prediction/blob/main/notebooks/03_model-comparison_threshold-tuning.ipynb)

## Resultados

O modelo **LR** apresentou o melhor equilíbrio entre precisão e _recall_ nos cenários de diferentes _thresholds_, além de ser computacionalmente um dos mais leves e também mais interpretáveis. (O modelo **KNN** apresentou melhor equilíbrio na situação de _threshold_ padrão.)

Exemplo de impacto (cenário com R=5):

- aproximadamente 93% dos _churns_ identificados;
- **_trade-off_**: aproximadamente 54% de clientes abordados desnecessariamente.

## Conclusão
O projeto demonstra como aprendizagem de máquina pode ser utilizada para decisões reais de negócio, considerando diferença nos custos de erro e impacto na operacionalização.
