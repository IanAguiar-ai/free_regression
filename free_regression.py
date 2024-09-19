"""
Regressão que aceita qualquer função para uma variável regressora

Metodo:
    Acha aleatóriamente os parâmetros

"""
from inspect import signature # Para pegar os argumentos de uma função
from copy import deepcopy
from random import random, seed

class Regression:
    """
    Classe de regressão

    Input(s):
    - function: A função qual o usuário quer fazer a regressão, essa função deve sempre ter a variável regressora chamada de x
    
    """
    __slots__ = ("function", "iterations", "params", "regressor", "__args_function", "__seed", "__lock")
    
    def __init__(self, function:"function", regressor:list = None) -> None:
        self.function:"function" = function
        
        temp = tuple(signature(function).parameters.keys())

        # Definindo regressora
        if regressor == None:
            assert "x" in temp, "The passed function must have the parameter 'x' or explicitly specify the regressor with the parameter 'regressor'"
            self.regressor = ["x"]
        elif type(regressor) == int or type(regressor) == float:
            self.regressor = [regressor]
        else:
            self.regressor = regressor

        # Argumentos da função
        self.__args_function:dict = {}
        self.params:list = []
        for parameter in temp:
            if parameter not in self.regressor:
                self.params.append(parameter)
                self.__args_function[parameter] = 1
        
        self.__seed = None
        self.iterations:int = 50 * len(self.__args_function.keys()) # Quanto mais parâmetros mais iterações eu precisso para que o valor mude

        # Variáveis bloqueadas
        self.__lock = {}

    def __repr__(self) -> str:
        """
        Mostra os argumentos
        """
        return f"FUNCTION: {self.function.__name__}\nREGRESSOR: {self.regressor}\nPARAMS: {self.__args_function}\nLOCK PARAMS: {self.__lock}"

    def __getitem__(self, index:str) -> float:
        """
        Mostra o argumento especifico pedido
        """
        assert type(index) == str, "The index must be a character(chr)"
        assert index in self.params, f"The index '{index}' must exist in params '{', '.join(self.params)}'"
        return self.__args_function[index]

    def set_seed(self, seed:int) -> None:
        """
        Coloca uma seed
        """
        assert type(seed) == int, "The seed must be an integer(int)!"
        self.__seed = seed

    def lock(self, **args) -> None:
        """
        Atualiza as variáveis que devem estar travadas
        """
        for arg in args:
            assert arg in self.__args_function.keys(), f"'{arg}' not in parameters of the function {self.function.__name__}"
            self.__lock[arg] = args[arg]

    def run(self, data:[list], precision:float = 0.01) -> None:
        """
        Faz a regressão

        Recebe:
        - data: lista de listas com x e y
        - step: Numero da precisão para achar os parâmetros esperados
        """

        assert type(data) == list, "The data must be a list of lists"
        assert type(data[0]) == list, "The data must be a list of lists"
        assert len(data[0]) == len(self.regressor) + 1, "The list of lists must have an x_n and a y parameter, for example [[x_0, x_1, ..., y], [x_0, x_1, ..., y], ...]"
        assert (k := list(map(len, data))) and max(k) == min(k), "The data list must be the same size in all itens"

        # Iniciando a seed
        if self.__seed is not None:
            seed(seed)
        
        # Pegando y esperado
        y_expected = [data[i][-1] for i in range(len(data))]

        # Salvando argumentos iniciais para a função
        args_temp:dict = {}
        for parameter in self.__args_function.keys():
            if parameter not in self.__lock.keys():
                args_temp[parameter] = self.__args_function[parameter]
            else:
                args_temp[parameter] = self.__lock[parameter] # Caso a variável deva estar travada

        precision_final, precision = precision/2, max(precision * 10, 1)
        while precision >= precision_final: #Vai diminuindo a variação da busca
            #print(f"Precisão atual: {precision}")
            with_no_iteration = 0
            while with_no_iteration < self.iterations:
                with_no_iteration += 1
                
                # y predito
                y_predicted:list = []
                for *x, _ in data:

                    # Separando as variáveis regressoras
                    x_args:dict = {}
                    for i in range(len(x)):
                        x_args[self.regressor[i]] = x[i]
                    #print(x_args, args_temp)

                    # Fazendo a predição
                    y_predicted.append(self.function(**x_args, **args_temp))

                # Resultado dos minimos quadrados
                result = least_squares(y_predicted, y_expected)

                # Atualizando melhores parâmetros para regressora
                if not "best_result" in locals():
                    best_result:float = result
                    best_args = deepcopy(args_temp)

                if result < best_result:
                    with_no_iteration = 0
                    best_result:float = result
                    best_args = deepcopy(args_temp)
                    #print(f"ERRO: {best_result} | ARGUMENTOS {best_args}")
                    #print(f"{y_predicted}\n{y_expected}")
                else:
                    args_temp = deepcopy(best_args)

                for parameter in self.__args_function.keys():
                    if parameter not in self.__lock.keys():
                        #print(f"{parameter}: {args_temp[parameter]} -> ", end = "")
                        args_temp[parameter] += random()*precision - precision/2
                        #print(f"{args_temp[parameter]}")

            # Aumenta a precisão
            precision /= 2

        # Salva o resultado
        self.__args_function = best_args

def variance(vector:list) -> float:
    """
    Calcula a variância de forma mais enxuta e otimizada
    Não usa do calculo da variância normal, sem a raiz no final
    """
    
    def mean(vector:list) -> float:
        """
        Calcula a média
        """
        return sum(vector)/len(vector)

    m = mean(vector)
    return sum([(m - x)*(m - x) for x in vector])/len(vector)

def least_squares(vector_1:list, vector_2:list) -> float:
    """
    Função de minimos quadrados
    """
    return sum([(vector_1[i] - vector_2[i])*(vector_1[i] - vector_2[i]) for i in range(len(vector_1))])        
