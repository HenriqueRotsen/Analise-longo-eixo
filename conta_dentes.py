import json
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join, splitext


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
        # Listar e ordenar arquivos das duas pastas
        files_folder_output = sorted(
            [f for f in listdir(model_path) if isfile(join(model_path, f))]
        )
        files_folder_anotacao = sorted(
            [
                f
                for f in listdir(self.anotacao_path)
                if isfile(join(self.anotacao_path, f))
            ]
        )

        # Obter conjunto de nomes base (sem extensão) para comparação
        base_output = {splitext(f)[0] for f in files_folder_output}
        base_anotacao = {splitext(f)[0] for f in files_folder_anotacao}

        # Manter apenas arquivos com nomes base comuns
        common_bases = base_output & base_anotacao  # Interseção dos conjuntos
        uncommon_bases = (base_output - base_anotacao) | (base_anotacao - base_output)


        # Filtrar listas originais para manter apenas arquivos correspondentes
        files_folder_output = [
            f for f in files_folder_output if splitext(f)[0] in common_bases
        ]
        files_folder_anotacao = [
            f for f in files_folder_anotacao if splitext(f)[0] in common_bases
        ]

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

    def mostra_metricas(self):
        erro, acuracia, precisao, recall, f1_score = self.calcula_metricas()
        print(
            f"Acertos: {self.acertos}\nErros: {self.falsos_positivos + self.falsos_negativos}\nFalsos Positivos: {self.falsos_positivos}\nFalsos Negativos: {self.falsos_negativos}\n"
        )
        print(f"Erro: {round(erro * 100, 2)}%")
        print(f"Acurácia: {round(acuracia * 100, 2)}%")
        print(
            f"Precisão: {round(precisao * 100, 2)}% (Proporção de positivos classificados corretamente)"
        )
        print(
            f"Recall: {round(recall * 100, 2)}% (Proporção de positivos corretamente identificados)"
        )
        print(f"F1-Score: {f1_score} (Média harmônica de precisão e recall)")

    def old_model_score(self):
        files_folder_anotacao, files_folder_output = self.obtem_dados(
            self.old_model_path
        )

        for file_ot, file_an in zip(files_folder_output, files_folder_anotacao):
            ot = read_file(self.old_model_path + file_ot)
            an = read_file(self.anotacao_path + file_an)

            if(splitext(file_ot)[0] != splitext(file_an)[0]):
                print("ERRO: Arquivos diferentes")


            self.total_dentes += 32

            for dente in self.dentes:  # Para cada dente do corpo humano (32 dentes)
                dente_ot = [
                    entity for entity in ot["entities"] if entity["class_name"] == dente
                ]
                dente_an = [entity for entity in an if entity["label"] == dente]

                if dente_ot:  # Existe um dente no modelo
                    if dente_an:  # Existe um dente na anotação
                        if dente_ot[0]["score"] > 0.1:  # Acerto
                            self.acertos += 1
                            self.verdadeiros_positivos += 1
                        elif dente_ot[0]["score"] < 0.1:  # Falso negativo
                            self.falsos_negativos += 1
                    else:  # Não existe dente na anotação
                        if dente_ot[0]["score"] > 0.1:  # Falso positivo
                            self.falsos_positivos += 1
                        elif dente_ot[0]["score"] < 0.1:  # Acerto
                            self.acertos += 1
                else:
                    print(f"ERRO: O modelo (ANTIGO) não retornou o dente {dente}!")
                    return

    def new_model_score(self):
        files_folder_anotacao, files_folder_output = self.obtem_dados(
            self.new_model_path
        )

        for file_ot, file_an in zip(files_folder_output, files_folder_anotacao):
            ot = read_file(self.new_model_path + file_ot)
            an = read_file(self.anotacao_path + file_an)

            self.total_dentes += 32

            for dente in self.dentes:  # Para cada dente do corpo humano (32 dentes)
                dente_ot = [
                    entity for entity in ot["entities"] if entity["class_name"] == dente
                ]
                dente_an = [entity for entity in an if entity["label"] == dente]

                if dente_ot:  # Existe um dente no modelo
                    if dente_an:  # Existe um dente na anotação
                        # Acerto
                        self.acertos += 1
                        self.verdadeiros_positivos += 1
                    else:  # Não existe dente na anotação
                        # Falso positivo
                        self.falsos_positivos += 1
                else:  # Não existe um dente no modelo
                    if dente_an:  # Existe um dente na anotação
                        # Falso negativo
                        self.falsos_negativos += 1
                    else:  # Não existe dente na anotação
                        # Acerto
                        self.acertos += 1


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


if __name__ == "__main__":
    print("------------- OLD Model ---------------")
    old = Avaliador()
    old.old_model_score()
    old.mostra_metricas()

    print("\n------------- NEW Model ---------------")
    new = Avaliador()
    new.new_model_score()
    new.mostra_metricas()

    visualiza_modelos(old, new)
