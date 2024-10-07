import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import tempfile
import os

def plot_expected_and_save(regression:"Regression", data:[list], name:str, size:list = (8, 6)) -> None:
    """
    Plot que compara os valores preditos e observados.
    Só funciona quando existe apenas um regressor e um valor esperado.

    Args:
        regression (Regression): Classe 'Regression' da função a ser plotada como preditora.
        data (list(list)): Dados, lista de listas sendo do tamanho nx2.
    """
    assert len(regression.regressors) == 1, "This graph only works if you have only one regressor"
    assert type(data) == list or type(data) == tuple, "The <data> must be a list"
    assert type(data[0]) == list or type(data[0]) == tuple, "The <data[n]> must be a list, <data> is list of lists"
    assert len(data[0]) == 2, "The <data[n]> has to be 2 elements"
    
    # Extrair os dados
    x = [values[0] for values in data]
    y1 = [values[1] for values in data]
    x_new = [min(x)]
    dif = sorted(x)
    dif = min([dif[i+1] - dif[i] for i in range(len(dif) - 1)])
    while x_new[-1] < max(x):
        x_new.append(x_new[-1] + max(dif, (max(x) - min(x))/200))
    y2 = [regression.prediction(**{regression.regressors[0]: value}) for value in x_new]

    # Criar o gráfico
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

    # Criar pasta 'temporary' se ela não existir
    save_dir = "temporary"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    name = os.path.join("temporary", name)
    plt.savefig(f"{name}.png")


def make_animation(obj:"Regression", data:list, precision:float = 0.01) -> None:
    """
    Faz a animação desde que o objeto passado seja da classe passada
    """
    obj._Regression__animation_run(data, precision = precision)
    
if __name__ == "__main__":
    from free_regression import Regression
    from models_regression import *

##    teste_1 = Regression(*generate_mlp_normals(regressors = 1, neurons = 2, max_ = 1))
##    teste_1.set_seed(1)
##    teste_1.change(b = 0)
##    print(teste_1)
##    dados = [[0, 0], [1, 1], [2, 0], [2.1, 1], [2.12, 1], [2.5, 1], [3, 0]]
##
##    make_animation(teste_1, dados)

##    teste_1 = Regression(*generate_distribuction(regressors = 1, normals = 2))
##    teste_1.set_seed(1)
##    teste_1.change(mean_0 = 1, mean_1 = 3, var_1 = 0.3)
##    print(teste_1)
##    dados = [[0, 0], [1, 0.4], [2, 0], [2.1, 0.1], [2.12, 0.20], [2.5, 0.40], [3, 0]]
##    make_animation(teste_1, dados)

    dados_ = [[i/100, i/100] for i in range(100)]
    modelo = Regression(*generate_mlp_classifier(1,5))
    modelo.set_seed(2024)
    make_animation(modelo, dados_, precision = 0.1)
