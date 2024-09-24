# Biblioteca graficos:
import matplotlib.pyplot as plt
import numpy as np

def plot_expected(regression:"Regression", data:[list], size:list = (8, 6)) -> None:
    """
    Plot que compara os valores preditos e observados.
    Só funciona quando existe apenas um regressor e um valor esperado.

    Args:
        regression (Regression): Classe 'Regression' da função a ser plotada como preditora.
        data (list(list)): Dados, lista de listas sendo do tamanho nx2.
    """
    assert len(regression.regressors) == 1, "This graph only works if you have only one regressors"
    assert type(data) == list or type(data) == tuple, "The <data> must be a list"
    assert type(data[0]) == list or type(data[0]) == tuple, "The <data[n]> must be a list, <data> is list of lists"
    assert len(data[0]) == 2, "The <data[n]> has to be 2 elements"
    
    x = [values[0] for values in data]
    y1 = [values[1] for values in data]
    x_new = [min(x)]
    dif = sorted(x)
    dif = min([dif[i+1] - dif[i] for i in range(len(dif) - 1)])
    while x_new[-1] < max(x):
        x_new.append(x_new[-1] + dif)
    y2 = [regression.prediction(**{regression.regressors[0]: value}) for value in x_new]

    fig, ax = plt.subplots(figsize = size)
    if sorted(list(set(x))) == x:
        ax.plot(x, y1, label = "Dados Observados", color = "blue", linestyle = "-")
    else:
        ax.scatter(x, y1, label = "Dados observados", color = "blue")
    ax.plot(x_new, y2, label = "Dados Observados", color = "red", linestyle = "-")
    ax.grid(True, which = "both", linestyle = "--", linewidth = 0.7)
    ax.set_title("Dados Observados vs Valores Preditos", fontsize = 16, weight = "bold")
    ax.set_xlabel("X", fontsize = 14)
    ax.set_ylabel("Y", fontsize = 14)
    ax.legend()
    plt.show()

def plot_residual(regression:"Regression", data:[list], size:list = (8, 6), percentile:list = [2.5, 97.5]) -> None:
    """
    Plot que mostra a distribuição dos resíduos.

    Args:
        regression (Regression): Classe 'Regression' da função a ser plotada como preditora.
        data (list(list)): Dados, lista de listas sendo do tamanho nx2.
    """
    assert type(data) == list or type(data) == tuple, "The <data> must be a list"
    assert type(data[0]) == list or type(data[0]) == tuple, "The <data[n]> must be a list, <data> is list of lists"
    assert len(regression.regressors) == len(data[0]) - 1, f"{len(regression.regressors)} regressors were indicated but {len(data[0]) - 1} appears in the data"
    
    x = [values[:-1] for values in data]
    y1 = [values[-1] for values in data]
    y2 = [regression.prediction([value]) for value in x]
    y_dif = [y1[i] - y2[i][0] for i in range(len(data))]

    fig, ax = plt.subplots(figsize = size)
    
    n, bins, patches = ax.hist(y_dif, bins=int(len(y_dif)**(1/2)*1.5), color = "skyblue", edgecolor = "gray", alpha = 0.7, density=True)
    n_percent = n * 100 / np.sum(n)
    ax.clear()
    ax.bar(bins[:-1], n_percent, width=np.diff(bins), align = "edge", color = "skyblue", edgecolor = "gray", alpha = 0.7)
    lower_bound = np.percentile(y_dif, percentile[0])
    upper_bound = np.percentile(y_dif, percentile[1])
    mean_residual = np.mean(y_dif)
    ax.axvline(mean_residual, color = "red", linestyle = "--", linewidth=2, label=f"Média dos resíduos: {mean_residual:.2f}")
    ax.axvline(lower_bound, color = "green", linestyle = "--", linewidth=2, label=f"Percentil {percentile[0]}%: {lower_bound:.2f}")
    ax.axvline(upper_bound, color = "green", linestyle = "--", linewidth=2, label=f"Percentil {percentile[1]}%: {upper_bound:.2f}")
    ax.grid(True, which = "both", linestyle = "--", linewidth=0.7)
    ax.set_title("Distribuição dos Resíduos", fontsize = 16, weight = "bold")
    ax.set_xlabel("Resíduos", fontsize = 14)
    ax.set_ylabel("Frequência (%)", fontsize = 14)
    ax.set_xlim([min(bins), max(bins)])
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    from free_regression import Regression
    from random import random

    def regressao_2(x:float, a:float, b:float, c:float) -> float:
        return a*x**2 + b*x + c

    def reg_2b(x1:float, x2:float, b1:float, b2:float) -> float:
        return x1*b1 + x2*b2

    dado = [[x, regressao_2(x, a = 15, b = -7, c = -4) + random()*100-50] for x in range(30)]
    dado = [[random()*i/100, random()*i/100] for i in range(40)]
    teste = Regression(regressao_2)
    teste.run(dado)
    print(teste)

    plot_expected(teste, dado)
    plot_residual(teste, dado)

    dado = []
    for i in range(20):
        a = int(random()*i*5)
        b = int(random()*i*5)
        dado.append([a, b, a*7.5 + b*(-2.4) + random()*i - i/2])
    teste_2 = Regression(reg_2b, ["x1", "x2"])
    teste_2.run(dado)
    print(teste_2)

    plot_residual(teste_2, dado)
