# Explicação do Código
Basicamente tem 2 funções que geram os dados dos modelos, conforme foi solicitado pelo Fábio. A partir disso é possivel calcular as métricas e comparar os modelos.
Existe também o código `monte_carlo.py` que é responsável por rodar o método de Monte Carlo para ver a persistência das métricas obtidas e assim observar se o modelo é estável.
## Estrutura do Código
### Classe `Avaliador`

A classe `Avaliador` gerencia a avaliação dos modelos.

#### Construtor (`__init__`)
```python
def __init__(self):
    self.anotacao_path = "test/anotacao/"
    self.old_model_path = "test/output_longaxis_old/"
    self.new_model_path = "test/output_longaxis_standard/"
    self.acertos = 0
    self.falsos_positivos = 0
    self.verdadeiros_positivos = 0
    self.falsos_negativos = 0
    self.total_dentes = 0
    self.dentes = gera_dentes()
```
Define os caminhos para os arquivos de anotação e saídas dos modelos, inicializa contadores de métricas e gera uma lista de dentes.

#### Obtenção de Arquivos
```python
def obtem_dados(self, model_path):
```
Essa função retorna os arquivos das anotações e as saídas dos modelos.

#### Cálculo de Métricas
```python
def calcula_metricas(self):
```
Essa função calcula métricas de desempenho do modelo, incluindo erro, acurácia, precisão, recall e F1-score.

#### Avaliação dos Modelos
##### Modelo Antigo
```python
def old_model_score(self):
```
Essa função avalia o modelo antigo comparando sua saída com as anotações.

##### Modelo Novo
```python
def new_model_score(self):
```
A avaliação do modelo novo segue a mesma lógica, mas sem um limite de score.

### Visualização Gráfica
```python
def visualiza_modelos(old, new):
```
Gera um gráfico comparando as métricas dos modelos.

