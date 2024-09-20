# Biblioteca graficos:
import matplotlib.pyplot as plt
import numpy as np

def plot_expected(regression:"Regression", data:[list]) -> None:
    """
    Plot que compara os valores preditos e esperados
    """
    assert len(regression.regressors) == 1, "This graph only works if you have only one regressors"
    assert type(data) == list or type(data) == tuple, "The <data> must be a list"
    assert type(data[0]) == list or type(data[0]) == tuple, "The <data[n]> must be a list, <data> is list of lists"
    assert len(data[0]) == 2, "The <data[n]> has to be 2 elements"
    
    x = [values[0] for values in data]
    y1 = [values[1] for values in data]
    y2 = [regression.prediction(**{regression.regressors[0]: value}) for value in x]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y2, label = "Valores esperados", color = "blue", linestyle = "-")
    ax.scatter(x, y1, label = "Dados observados", color = "red")
    ax.grid(True, which = "both", linestyle = "--", linewidth = 0.7)
    ax.set_title("Dados Observados vs Valores Esperados", fontsize = 16, weight = "bold")
    ax.set_xlabel("X", fontsize = 14)
    ax.set_ylabel("Y", fontsize = 14)
    ax.legend()
    plt.show()

def plot_residual(regression:"Regression", data:[list]) -> None:
    """
    Plot que mostra a distribuição dos residuos
    """
    assert len(regression.regressors) == 1, "This graph only works if you have only one regressors"
    assert type(data) == list or type(data) == tuple, "The <data> must be a list"
    assert type(data[0]) == list or type(data[0]) == tuple, "The <data[n]> must be a list, <data> is list of lists"
    assert len(data[0]) == 2, "The <data[n]> has to be 2 elements"
    
    x = [values[0] for values in data]
    y1 = [values[1] for values in data]
    y2 = [regression.prediction(**{regression.regressors[0]: value}) for value in x]
    y_dif = [y1[i] - y2[i] for i in range(len(data))]

    fig, ax = plt.subplots(figsize=(8, 6))
    n, bins, patches = ax.hist(y_dif, bins = int(len(y_dif)**(1/2)*1.5), color = "skyblue", edgecolor = "gray", alpha = 0.7)
    mean_residual = np.mean(y_dif)
    ax.axvline(mean_residual, color = "red", linestyle = "--", linewidth = 2, label = f"Média dos erros dos resíduos: {mean_residual:.2f}")
    ax.grid(True, which = "both", linestyle = "--", linewidth = 0.7)
    ax.set_title("Distribuição dos Resíduos", fontsize = 16, weight = "bold")
    ax.set_xlabel("Resíduos", fontsize = 14)
    ax.set_ylabel("Frequência", fontsize = 14)
    ax.set_xlim([min(bins), max(bins)])
    ax.legend()
    plt.show()

if __name__ == "__main__":
    from free_regression import Regression
    from random import random

    def regressao_2(x:float, a:float, b:float, c:float):
        return a*x**2 + b*x + c

    dado = [[x, regressao_2(x, a = 15, b = -7, c = -4) + random()*100-50] for x in range(30)]
    teste = Regression(regressao_2)
    teste.run(dado)
    print(teste)

    plot_expected(teste, dado)
    plot_residual(teste, dado)
