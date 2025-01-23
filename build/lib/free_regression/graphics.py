# Biblioteca graficos:
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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
    
    x:list = [values[0] for values in data]
    y1:list = [values[1] for values in data]
    x_new:list = [min(x)]
    dif:list = sorted(x)
    dif:float = min([dif[i+1] - dif[i] for i in range(len(dif) - 1)])
    while x_new[-1] < max(x):
        x_new.append(x_new[-1] + max(dif, (max(x) - min(x))/200))
    y2:list = [regression.prediction(**{regression.regressors[0]: value}) for value in x_new]

    fig, ax = plt.subplots(figsize = size)
    if sorted(list(set(x))) == x:
        ax.plot(x, y1, label = "Dados Observados", color = "blue", linestyle = "-")
    else:
        ax.scatter(x, y1, label = "Dados observados", color = "blue")
    ax.plot(x_new, y2, label = "Valores Preditos", color = "red", linestyle = "--")
    ax.grid(True, which = "both", linestyle = "--", linewidth = 0.7)
    ax.set_title("Dados Observados vs Valores Preditos", fontsize = 16, weight = "bold")
    ax.set_xlabel("X", fontsize = 14)
    ax.set_ylabel("Y", fontsize = 14)
    ax.legend()
    plt.show()

def plot_residual(regression:"Regression", data:[list], size:list = (8, 6), percentile:list = [2.5, 97.5]) -> list:
    """
    Plot que mostra a distribuição dos resíduos.

    Args:
        regression (Regression): Classe 'Regression' da função a ser plotada como preditora.
        data (list(list)): Dados, lista de listas sendo do tamanho nx2.

    Return:
        Residuos (list): Lista de resíduos dos erros
    """
    assert type(data) == list or type(data) == tuple, "The <data> must be a list"
    assert type(data[0]) == list or type(data[0]) == tuple, "The <data[n]> must be a list, <data> is list of lists"
    assert len(regression.regressors) == len(data[0]) - 1, f"{len(regression.regressors)} regressors were indicated but {len(data[0]) - 1} appears in the data"
    
    x:list = [values[:-1] for values in data]
    y1:list = [values[-1] for values in data]
    y2:list = [regression.prediction([value]) for value in x]
    y_dif:list = [y1[i] - y2[i][0] for i in range(len(data))]

    fig, ax = plt.subplots(figsize = size)
    
    n, bins, patches = ax.hist(y_dif, bins = int(len(y_dif)**(1/2)*1.5), color = "skyblue", edgecolor = "gray", alpha = 0.7, density=True)
    n_percent = n * 100 / np.sum(n)
    ax.clear()
    ax.bar(bins[:-1], n_percent, width = np.diff(bins), align = "edge", color = "skyblue", edgecolor = "gray", alpha = 0.7)
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
    return y_dif

def plot_prediction_bands(regression:"Regression", data:[list], size:list = (8, 6), sigma:float = 1.64, amplitude:float = None):
    """
    Plot de um gráfico da série com os intervalos de confiança passado por sigma

    Args:
        regression (Regression): Classe 'Regression' da função a ser plotada como preditora.
        data (list(list)): Dados, lista de listas sendo do tamanho nx2.

    sigmas:
    1,64	90,00
    1,96	95,00
    2,33	98,00
    2,58	99,00
    3,00	99,74
    """

    def mov_var(x_real, x_predict, y_real:list, y_predict:list, amplitude:float):
        def var(y_real, y_predict):
            return sum([(y_real[i] - y_predict[i])**2 for i in range(len(y_predict))])/max(len(y_predict), 1)
        mov_var:list = []
        for i in range(len(x_predict)):
            temp_y_real, temp_y_predict = [], []
            for j in range(len(x_real)):
                if x_real[j] - amplitude <= x_predict[i] <= x_real[j] + amplitude:
                    temp_y_real.append(y_real[j])
                    temp_y_predict.append(y_predict[j])
            mov_var.append(var(temp_y_real, temp_y_predict))
        return mov_var
    
    assert type(data) == list or type(data) == tuple, "The <data> must be a list"
    assert type(data[0]) == list or type(data[0]) == tuple, "The <data[n]> must be a list, <data> is list of lists"
    assert len(regression.regressors) == len(data[0]) - 1, f"{len(regression.regressors)} regressors were indicated but {len(data[0]) - 1} appears in the data"
    assert sigma > 0, f"sigma has to be larger than 0, sigma now: {sigma}"

    x:list = [values[0] for values in data]
    if amplitude == None:
        amplitude:float = (max(x) - min(x))/20
    
    y1:list = [values[1] for values in data]
    x_new:list = [min(x)]
    dif:list = sorted(x)
    dif:float = min([dif[i+1] - dif[i] for i in range(len(dif) - 1)])
    while x_new[-1] < max(x):
        x_new.append(x_new[-1] + max(dif, (max(x) - min(x))/200))
    y2:list = [regression.prediction(**{regression.regressors[0]: value}) for value in x_new]

    try:
        var_mov:list = mov_var(x, x, y1, y2, amplitude)
        lim_sup:list = [y2[i] + var_mov[i]**(1/2) * sigma for i in range(len(var_mov))]
        lim_inf:list = [y2[i] - var_mov[i]**(1/2) * sigma for i in range(len(var_mov))]
    except IndexError:
        #print(f"x: {len(x)}, x_new: {len(x_new)}, y1: {len(y1)}, y2: {len(y2)}")
        y2_new:list = [regression.prediction(**{regression.regressors[0]: value}) for value in x]
        var_mov:list = mov_var(x, x, y1, y2_new, amplitude)
        #print(f"var_mov: {len(var_mov)}, x: {len(x)}, x_new: {len(x_new)}, y1: {len(y1)}, y2_new: {len(y2_new)}")
        lim_sup:list = [y2_new[i] + var_mov[i]**(1/2) * sigma for i in range(len(var_mov))]
        lim_inf:list = [y2_new[i] - var_mov[i]**(1/2) * sigma for i in range(len(var_mov))]

    fig, ax = plt.subplots(figsize = size)
    if sorted(list(set(x))) == x:
        linear:bool = True
        ax.plot(x, y1, label = "Dados Observados", color = "blue", linestyle = "-")
    else:
        linear:bool = False
        ax.scatter(x, y1, label = "Dados observados", color = "blue")
        
    ax.plot(x_new, y2, label = "Valores Preditos", color = "red", linestyle = "--")
        
    ax.grid(True, which = "both", linestyle = "--", linewidth = 0.7)
    if linear:
        ax.fill_between(x, lim_inf, lim_sup, color = "lightgreen", alpha = 0.5, label = f"Banda de Confiança (±{sigma}σ)")
    else:
        combined = sorted(zip(x, lim_inf, lim_sup), key = lambda x:x[0])
        x, lim_inf, lim_sup = zip(*combined)
        ax.fill_between(x, lim_inf, lim_sup, color = "lightgreen", alpha = 0.5, label = f"Banda de Confiança (±{sigma}σ)")
    ax.set_title("Dados Observados vs Valores Preditos", fontsize = 16, weight = "bold")
    ax.set_xlabel("X", fontsize = 14)
    ax.set_ylabel("Y", fontsize = 14)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    from free_regression import Regression
    from random import random
    from models_regression import *

    def regressao_2(x:float, a:float, b:float, c:float) -> float:
        return a*x**2 + b*x + c

    def reg_2b(x1:float, x2:float, b1:float, b2:float) -> float:
        return x1*b1 + x2*b2

    def lin_reg(x:float, a:float, b:float) -> float:
        return x*a + b

##    dado = [[x, regressao_2(x, a = 15, b = -7, c = -4) + random()*100-50] for x in range(30)]
##    dado = [[random()*i/100, random()*i/100] for i in range(40)]
##    teste = Regression(regressao_2)
##    teste.run(dado)
##    print(teste)
##
##    plot_expected(teste, dado)
##    plot_residual(teste, dado)

    dado = []
    for i in range(20):
        a = int(random()*i*5)
        b = int(random()*i*5)
        dado.append([a, b, a*7.5 + b*(-2.4) + random()*i - i/2])
    dado = [[i, i*0.5 + 4 + random()*random()*random()*20] for i in range(100)]

    teste_2 = Regression(lin_reg)
    teste_2.run(dado)
    print(teste_2)

    #plot_expected(teste_2, dado)
    #plot_residual(teste_2, dado)
    plot_prediction_bands(teste_2, dado, sigma = 1.64)

    def reg_log(x, b0, b1) -> float:
      return 1/(1 + 2.71**(-(b0*x+b1)))

    dados_ = [[i/100, i/100] for i in range(100)]
    modelo = Regression(reg_log)#Regression(*generate_mlp_classifier(1,1))
    modelo.set_seed(2024)
    modelo.run(dados_, precision = 0.1)
    #modelo.change(b0 = 1, b1 = -0.5)
    print(modelo)
    plot_expected(modelo, dados_)
    plot_prediction_bands(modelo, dados_)
    
    def reg_mult(x, b0, b1, b2, b3):
      if x < 20:
        return (b2*x + b3)
      elif x > 40:
        return (b0*x + b1)
      else:
        return ((x-20)/20) * (b0*x + b1) + (1 - (x-20)/20) * (b2*x + b3)

    from data import MedidasDeMassa, transpose
    dados = MedidasDeMassa()
    dados = transpose([dados[1], dados["TotalHeight"]])
    modelo = Regression(reg_mult)
    modelo.set_seed(2024)
    modelo.change(b0 = 0, b1 = 90, b2 = 0, b3 = 30)
    modelo.run(dados, precision = 0.1)
    print(modelo)
    plot_expected(modelo, dados)
    plot_prediction_bands(modelo, dados, amplitude = 1)

##    
##    teste_1 = Regression(*generate_mlp_normals(regressors = 1, neurons = 2, max_ = 1))
##    teste_1.set_seed(1)
##    teste_1.change(b = 0)
##    print(teste_1)
##    dados = [[0, 0], [1, 1], [2, 0], [2.1, 1], [2.12, 1], [2.5, 1], [3, 0]]
##    teste_1.run(dados)
##    print(f"{teste_1}\n")
##    print(f"{teste_1.prediction([[0], [1], [2], [2.3], [2.5], [3]])}")
##    plot_expected(teste_1, dados)
##    plot_residual(teste_1, dados)
    
##    teste_1 = Regression(*generate_distribuction(regressors = 1, normals = 2))
##    teste_1.set_seed(1)
##    teste_1.change(mean_0 = 1, mean_1 = 3, var_1 = 0.3)
##    print(teste_1)
##    dados = [[0, 0], [1, 0.4], [2, 0], [2.1, 0.1], [2.12, 0.20], [2.5, 0.40], [3, 0]]
##    plot_expected(teste_1, dados)
##    teste_1.run(dados)
##    print(f"{teste_1}\n")
##    print(f"{teste_1.prediction([[0], [1], [2], [2.3], [2.5], [3]])}")
##    plot_expected(teste_1, dados)
##    plot_residual(teste_1, dados)
