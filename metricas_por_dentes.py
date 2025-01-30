import json
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join


class Avaliador:
    def __init__(self):
        self.anotacao_path = "test/anotacao/"
        self.old_model_path = "test/output_longaxis_old/"
        self.new_model_path = "test/output_longaxis_standard/"
        self.dentes = gera_dentes()
        self.metricas_por_dente = {dente: {"acertos": 0, "verdadeiros_positivos": 0, "falsos_positivos": 0, "falsos_negativos": 0} for dente in self.dentes}

    def obtem_dados(self, model_path):
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
        return files_folder_anotacao, files_folder_output

    def calcula_metricas(self):
        metricas = {}
        for dente, valores in self.metricas_por_dente.items():
            total = valores["acertos"] + valores["falsos_positivos"] + valores["falsos_negativos"]
            if total == 0:
                continue
            erro = (valores["falsos_positivos"] + valores["falsos_negativos"]) / total
            acuracia = valores["acertos"] / total
            precisao = valores["verdadeiros_positivos"] / (valores["verdadeiros_positivos"] + valores["falsos_positivos"])
            recall = valores["verdadeiros_positivos"] / (valores["verdadeiros_positivos"] + valores["falsos_negativos"])
            f1_score = (2 * precisao * recall) / (precisao + recall) if precisao + recall > 0 else 0
            metricas[dente] = {"erro": erro, "acuracia": acuracia, "precisao": precisao, "recall": recall, "f1_score": f1_score}
        return metricas

    def mostra_metricas(self):
        metricas = self.calcula_metricas()
        for dente, valores in metricas.items():
            print(f"Dente {dente}: Erro: {valores['erro']:.2f}, Acurácia: {valores['acuracia']:.2f}, Precisão: {valores['precisao']:.2f}, Recall: {valores['recall']:.2f}, F1-Score: {valores['f1_score']:.2f}")

    def avalia_modelo(self, model_path, modelo="old"):
        files_folder_anotacao, files_folder_output = self.obtem_dados(model_path)
        
        for file_ot, file_an in zip(files_folder_output, files_folder_anotacao):
            ot = read_file(model_path + file_ot)
            an = read_file(self.anotacao_path + file_an)
            
            for dente in self.dentes:
                dente_ot = [entity for entity in ot["entities"] if entity["class_name"] == dente]
                dente_an = [entity for entity in an if entity["label"] == dente]
                
                if modelo=="old":
                    if dente_ot:
                        if dente_an:
                            if dente_ot[0]["score"] > 0.1:
                                self.metricas_por_dente[dente]["acertos"] += 1
                                self.metricas_por_dente[dente]["verdadeiros_positivos"] += 1
                            elif dente_ot[0]["score"] < 0.1:
                                self.metricas_por_dente[dente]["falsos_negativos"] += 1
                        else:
                            if dente_ot[0]["score"] > 0.1:
                                self.metricas_por_dente[dente]["falsos_positivos"] += 1
                            elif dente_ot[0]["score"] < 0.1:
                                self.metricas_por_dente[dente]["acertos"] += 1
                else:
                    if dente_ot:
                        if dente_an:
                            self.metricas_por_dente[dente]["acertos"] += 1
                            self.metricas_por_dente[dente]["verdadeiros_positivos"] += 1
                        else:
                            self.metricas_por_dente[dente]["falsos_positivos"] += 1
                    else:
                        if dente_an:
                            self.metricas_por_dente[dente]["falsos_negativos"] += 1
                        else:
                            self.metricas_por_dente[dente]["acertos"] += 1


def read_file(file_path):
    with open(file_path, "r") as file:
        return json.loads(file.read())


def gera_dentes():
    return [f"{quadrante}{posicao}" for quadrante in range(1, 5) for posicao in range(1, 9)]


def visualiza_metricas(avaliador, titulo):
    metricas = avaliador.calcula_metricas()
    dentes = list(metricas.keys())

    erro = [m["erro"] for m in metricas.values()]
    acuracia = [m["acuracia"] for m in metricas.values()]
    precisao = [m["precisao"] for m in metricas.values()]
    recall = [m["recall"] for m in metricas.values()]
    f1_score = [m["f1_score"] for m in metricas.values()]

    x = np.arange(len(dentes))  # Índices para os dentes
    width = 0.2  # Largura das barras

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Gráfico de Precisão, Recall e F1-Score
    axs[0].bar(x - width, precisao, width, label="Precisão")
    axs[0].bar(x, recall, width, label="Recall")
    axs[0].bar(x + width, f1_score, width, label="F1-Score")
    axs[0].set_ylabel("Valor")
    axs[0].set_title(f"Métricas por dente - {titulo}")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(dentes, rotation=90)
    axs[0].legend()

    # Gráfico de Erro e Acurácia
    axs[1].bar(x - width / 2, erro, width, label="Erro", color="red")
    axs[1].bar(x + width / 2, acuracia, width, label="Acurácia", color="green")
    axs[1].set_ylabel("Valor")
    axs[1].set_title(f"Erro e Acurácia por dente - {titulo}")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(dentes, rotation=90)
    axs[1].legend()

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    print("------------- OLD Model ---------------")
    old = Avaliador()
    old.avalia_modelo(old.old_model_path)
    old.mostra_metricas()
    visualiza_metricas(old, "Modelo Antigo")

    print("\n------------- NEW Model ---------------")
    new = Avaliador()
    new.avalia_modelo(new.new_model_path, "new")
    new.mostra_metricas()
    visualiza_metricas(new, "Modelo Novo")
