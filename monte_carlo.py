import json
import random
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join


class Avaliador:
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

    def obtem_dados(self, model_path):
        files_folder_output = sorted(
            [f for f in listdir(model_path) if isfile(join(model_path, f))]
        )  # Arquivos da pasta output
        files_folder_anotacao = sorted(
            [
                f
                for f in listdir(self.anotacao_path)
                if isfile(join(self.anotacao_path, f))
            ]
        )  # Arquivos da pasta anotacao
        return files_folder_anotacao, files_folder_output

    def calcula_metricas(self):
        # Cálculo do erro: Proporção de exemplos classificados incorretamente
        erro = (self.falsos_positivos + self.falsos_negativos) / self.total_dentes

        # Cálculo da acurácia: Proporção de exemplos classificados corretamente
        acuracia = self.acertos / self.total_dentes

        # Cálculo da precisão: Proporção de exemplos positivos classificados corretamente (É útil quando o custo de falsos positivos é alto)
        precisao = self.verdadeiros_positivos / (
            self.verdadeiros_positivos + self.falsos_positivos
        )

        # Cálculo do recall: Proporção de exemplos positivos corretamente identificados (É útil quando o custo de falsos negativos é alto)
        recall = self.verdadeiros_positivos / (
            self.verdadeiros_positivos + self.falsos_negativos
        )

        # Cálculo do F1-score: Média harmônica de precisão e recall
        f1_score = (2 * precisao * recall) / (precisao + recall)

        return erro, acuracia, precisao, recall, f1_score



class MonteCarloAvaliador(Avaliador):
    def __init__(self, num_iteracoes=1000, amostra_tamanho=0.7):
        super().__init__()
        self.num_iteracoes = num_iteracoes
        self.amostra_tamanho = amostra_tamanho  # Proporção de dados usados por iteração

    def sincroniza_listas(self, files_anotacao, files_output):
        # Remover os últimos 7 arquivos da lista maior
        if len(files_anotacao) > len(files_output):
            files_anotacao = files_anotacao[:len(files_output)]
        elif len(files_output) > len(files_anotacao):
            files_output = files_output[:len(files_anotacao)]

        return files_anotacao, files_output


    def monte_carlo(self, modelo="old"):
        metricas = {
            "erro": [],
            "acuracia": [],
            "precisao": [],
            "recall": [],
            "f1_score": [],
        }

        # Obter dados apropriados para o modelo
        model_path = self.old_model_path if modelo == "old" else self.new_model_path
        files_folder_anotacao, files_folder_output = self.obtem_dados(model_path)

        # Sincronizar listas
        files_folder_anotacao, files_folder_output = self.sincroniza_listas(
            files_folder_anotacao, files_folder_output
        )

        for _ in range(self.num_iteracoes):
            # Resetar contadores
            self.acertos = 0
            self.falsos_positivos = 0
            self.verdadeiros_positivos = 0
            self.falsos_negativos = 0
            self.total_dentes = 0

            # Amostragem aleatória
            sample_size = int(len(files_folder_anotacao) * self.amostra_tamanho)
            sample_indices = random.sample(
                range(len(files_folder_anotacao)), sample_size
            )
            sampled_files_anotacao = [files_folder_anotacao[i] for i in sample_indices]
            sampled_files_output = [files_folder_output[i] for i in sample_indices]

            for file_ot, file_an in zip(sampled_files_output, sampled_files_anotacao):
                ot = read_file(model_path + file_ot)
                an = read_file(self.anotacao_path + file_an)

                self.total_dentes += 32

                for dente in self.dentes:
                    dente_ot = [
                        entity
                        for entity in ot["entities"]
                        if entity["class_name"] == dente
                    ]
                    dente_an = [entity for entity in an if entity["label"] == dente]

                    if modelo=="old":
                        if dente_ot:
                            if dente_an:
                                if dente_ot[0]["score"] > 0.1:
                                    self.acertos += 1
                                    self.verdadeiros_positivos += 1
                                elif dente_ot[0]["score"] < 0.1:
                                    self.falsos_negativos += 1
                            else:
                                if dente_ot[0]["score"] > 0.1:
                                    self.falsos_positivos += 1
                                elif dente_ot[0]["score"] < 0.1:
                                    self.acertos += 1
                    else:
                        if dente_ot:
                            if dente_an:
                                self.acertos += 1
                                self.verdadeiros_positivos += 1
                            else:
                                self.falsos_positivos += 1
                        else:
                            if dente_an:
                                self.falsos_negativos += 1
                            else:
                                self.acertos += 1

            # Calcular métricas para a amostra
            erro, acuracia, precisao, recall, f1_score = self.calcula_metricas()
            metricas["erro"].append(erro)
            metricas["acuracia"].append(acuracia)
            metricas["precisao"].append(precisao)
            metricas["recall"].append(recall)
            metricas["f1_score"].append(f1_score)

        return metricas


def read_file(file_path):
    with open(file_path, "r") as file:
        data = json.loads(file.read())
    return data


def gera_dentes():
    dentes = []
    for quadrante in range(1, 5):  # Quadrantes 1 a 4
        for posicao in range(1, 9):  # Posições 1 a 8
            dentes.append(f"{quadrante}{posicao}")
    return dentes


def visualiza_modelos(old, new):
    # Criando uma visualização gráfica
    metricas = [
        "Erro (%)",
        "Acurácia (%)",
        "Precisão (%)",
        "Recall (%)",
        "F1-Score",
    ]
    erro_old, acuracia_old, precisao_old, recall_old, f1_score_old = (
        old.calcula_metricas()
    )
    erro_new, acuracia_new, precisao_new, recall_new, f1_score_new = (
        new.calcula_metricas()
    )

    valores = [
        [erro_old, acuracia_old, precisao_old, recall_old, f1_score_old],
        [erro_new, acuracia_new, precisao_new, recall_new, f1_score_new],
    ]

    # Criar o gráfico de barras
    x = np.arange(len(metricas))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, valores[0], width, label="Modelo Antigo")
    ax.bar(x + width / 2, valores[1], width, label="Modelo Novo")
    # Adicionar rótulos e título
    ax.set_ylabel("Valor")
    ax.set_title("Comparação das Métricas dos Modelos")
    ax.set_xticks(x)
    ax.set_xticklabels(metricas)
    ax.legend()

    # Mostrar o gráfico
    plt.show()


def visualiza_distribuicoes(metricas):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()
    metricas_nomes = list(metricas.keys())

    for i, metrica in enumerate(metricas_nomes):
        axs[i].hist(
            metricas[metrica], bins=20, alpha=0.7, color="blue", edgecolor="black"
        )
        axs[i].set_title(f"Distribuição de {metrica.capitalize()}")
        axs[i].set_xlabel(metrica.capitalize())
        axs[i].set_ylabel("Frequência")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("------------- Monte Carlo: OLD Model ---------------")
    old_mc = MonteCarloAvaliador()
    metricas_old = old_mc.monte_carlo(modelo="old")
    visualiza_distribuicoes(metricas_old)

    print("\n------------- Monte Carlo: NEW Model ---------------")
    new_mc = MonteCarloAvaliador()
    metricas_new = new_mc.monte_carlo(modelo="new")
    visualiza_distribuicoes(metricas_new)
