from inspect import signature # Para pegar os argumentos de uma função
from copy import deepcopy
from random import random, seed

def least_squares(vector_1:list, vector_2:list) -> float:
    """
    Função de minimos quadrados
    """
    return sum([(vector_1[i] - vector_2[i])*(vector_1[i] - vector_2[i]) for i in range(len(vector_1))])

class Regression:
    """
    Classe de regressão que aceita qualquer função como regressora

    Metodo:
        Acha aleatóriamente os parâmetros da função de regressão passada usando algum tipo de função de perda.
        1. Começa com os parâmetros modificaveis sendo igual a 1;
        2. Altera esses parâmetros em -<precision>/2 até <precision>/2 para cima ou para baixo;
        3. Se precision <presision> minima definida pelo usuário ele passa para o passo 5;
        4. Se ele não melhora em <iterations> iterações, ele divide precision por 2 e volta para o passo (2);
        5. Salva o erro e os parâmetros ótimos que podem ser acessados pelo usuário. 

    Input(s):
    - function: A função qual o usuário quer fazer a regressão, essa função deve sempre ter a variável regressora chamada de x
    
    """
    __slots__ = ("iterations", "params", "regressor", "__function", "__args_function", "__seed", "__lock", "__loss_function", "__error")
    
    def __init__(self, function:"function", regressor:list = None, loss_function:"function" = least_squares) -> None:
        """
        Inicializa a classe, as etapas são:

        -
        - 
        """
        self.__function:"function" = function
        self.__loss_function:"function" = loss_function
        self.__error:float = None
        
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

    def __eq__(self, obj) -> bool:
        """
        Confere se as funções de regressões são as mesmas
        """
        assert type(obj) == Regression, "Only Regression objects can be compared"
        return True if self.__function == obj.__function else False

    def __repr__(self) -> str:
        """
        Mostra os argumentos
        """
        output:str = f"FUNCTION: {self.__function.__name__}"
        if self.__error != None:
            output += f"\nLOSS FUNCTION({self.__loss_function.__name__}): {self.__error:0.08f}"
        output += f"\nREGRESSOR: {', '.join(self.regressor)}"
        if len(self.__lock) > 0:
            output += f"\nLOCK PARAMS: {', '.join(self.__lock)}"
        output += "\nPARAMS:"
        for arg in self.__args_function.keys():
            output += f"\n  {arg} = {self.__args_function[arg]:0.08f}"
        return output
        

    def __getitem__(self, index:str) -> float:
        """
        Mostra o argumento especifico pedido
        """
        assert type(index) == str, "The index must be a character(chr)"
        assert index in self.params, f"The index '{index}' must exist in params '{', '.join(self.params)}'"
        return self.__args_function[index]

    def __add__(self, obj) -> "Regression":
        """
        + para mistura
        """
        globals()[f"{self.__function.__name__}"] = self.__function
        globals()[f"{obj.__function.__name__}"] = obj.__function
        print(globals())
        return eval(self.generic_function(obj, operator = "+"))

    def generic_function(self, obj:"Regression", operator:str) -> str:
        all_parameters = list(set(self.regressor) | set(obj.regressor) | set(self.params) | set(obj.params))
        
        inputs_1 = ""
        for input_ in list(set(self.regressor) | set(self.params)):
            inputs_1 += f"{input_} = {input_},"

        inputs_2 = ""
        for input_ in list(set(obj.regressor) | set(obj.params)):
            inputs_2 += f"{input_} = {input_},"
        
        return f"Regression(lambda {', '.join(all_parameters)} : {self.__function.__name__}({inputs_1}) + {obj.__function.__name__}({inputs_2}))"

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
            assert arg in self.__args_function.keys(), f"'{arg}' not in parameters of the function {self.__function.__name__}"
            self.__lock[arg] = args[arg]

    def prediction(self, list_prediction:list = None, **x_args) -> float:
        """
        Faz a previsão de f(...) = y
        x_args: Parametros regressores
        """

        if type(list_prediction) == list or type(list_prediction) == tuple: # Caso o usuário tenha passado uma série de valores para a predição
            assert type(list_prediction[0]) == list or type(list_prediction[0]) == tuple, "If you want to pass a series of values​to predict, you should pass the list of lists of values with the regressor parameters"
            assert min(map(len, list_prediction)) == max(map(len, list_prediction)) == len(self.regressor), f"Your list of lists must be {len(list_prediction)} by {len(self.regressor)} in size"

            results = []
            x_args = {}
            for values in list_prediction:
                for i in range(len(self.regressor)):
                    x_args[self.regressor[i]] = values[i]
                results.append(self.__function(**x_args, **self.__args_function))
                
            return results
        
        else: # Caso o usuário tenha passado valores específicos para a predição
            assert len(set(x_args.keys()) & set(self.__args_function.keys())) == 0, f"You cannot pass a parameter as a regressor that is already being used as a prediction parameter.\n  Regressor parameters: {', '.join(x_args.keys())}\n  Predictor parameters: {', '.join(self.__args_function.keys())}"
            assert set(x_args.keys()) == set(self.regressor), f"Pass regressor parameters correctly\n  Regressor parameters passed: {', '.join(x_args.keys())}\n  Expected regressor parameters: {', '.join(self.regressor)}"

            return self.__function(**x_args, **self.__args_function)

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
        assert type(precision) == int or type(precision) == float, "Precision has to be a float or int"

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
        while precision >= precision_final: # Vai diminuindo a variação da busca
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

                    # Fazendo a predição
                    y_predicted.append(self.__function(**x_args, **args_temp))

                # Resultado dos minimos quadrados
                result = self.__loss_function(y_predicted, y_expected)

                # Atualizando melhores parâmetros para regressora
                if not "best_result" in locals():
                    best_result:float = result
                    best_args = deepcopy(args_temp)

                if result < best_result:
                    with_no_iteration = 0
                    best_result:float = result
                    best_args = deepcopy(args_temp)
                else:
                    args_temp = deepcopy(best_args)

                for parameter in self.__args_function.keys():
                    if parameter not in self.__lock.keys():
                        args_temp[parameter] += random()*precision - precision/2

            # Aumenta a precisão
            precision /= 2

        # Salva o resultado
        self.__args_function = best_args
        self.__error = best_result
