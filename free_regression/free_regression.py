from inspect import signature, getsource # Para pegar os argumentos de uma função e a própria função
from copy import deepcopy
from random import random, seed

def least_squares(vector_1:list, vector_2:list) -> float:
    """
    Função de minimos quadrados
    """
    return sum([(vector_1[i] - vector_2[i])*(vector_1[i] - vector_2[i]) for i in range(len(vector_1))])

def least_squares_multivariate(vector_1:list, vector_2:list) -> float:
    """
    Função de mínimos quadrados para listas de listas (espaço n-dimensional).
    """
    return sum(sum((vector_1[i][j] - vector_2[i][j]) * (vector_1[i][j] - vector_2[i][j]) for j in range(len(vector_1[i]))) for i in range(len(vector_1)))

class Regression:
    """
    Classe de regressão que aceita qualquer função como regressora.

    Metodo:
        Acha aleatóriamente os parâmetros da função de regressão passada usando algum tipo de função de perda.
        1. Começa com os parâmetros modificaveis sendo igual a 1;
        2. Altera esses parâmetros em -<precision>/2 até <precision>/2 para cima ou para baixo;
        3. Se precision <presision> minima definida pelo usuário ele passa para o passo 5;
        4. Se ele não melhora em <iterations> iterações, ele divide precision por 2 e volta para o passo (2);
        5. Salva o erro e os parâmetros ótimos que podem ser acessados pelo usuário. 

    Args:
        function (function): A função qual o usuário quer fazer a regressão, essa função deve sempre ter a variável regressora chamada de x.
        regressors (list): Lista de regressores, não é precisso passar se a função tiver apenas um parâmetro regressor e ele se chame 'x'.
        loss_function (function): Função de perda, é a função 'least squares' mas pode ser qualquer uma passada pelo usuário.
    """
    __slots__ = ("iterations", "params", "regressors", "weights", "__len_y", "__function", "__args_function", "__seed", "__lock", "__loss_function", "__error", "__robust", "__limiar", "__print")
    
    def __init__(self, function:"function" = None, regressors:list = None, loss_function:"function" = least_squares, print:bool = False) -> None:
        """
        Inicializa a classe.

        Args:
            function:(função | def) Função qual será aplicada a regressão, por exemplo, f = lambda x, a, b: x*a + b.
            regressors:[str] Lista de strings qual indica os regressores, por padrão é apenas 'x' mas se tiver mais do que um ou uma letra diferente ela deve ser indicada.
            loss_function:(função | def) função que é minimizada ao fazer a regressão.
            print:(bool) Booleano que indica se deve ser printado o estado da regressão.
        """
        if function == None:
            function = lambda x, a, b : x*a + b
        elif type(function) == list or type(function) == tuple:
            if len(function) == 2 and regressors == None:
                function, regressors = function[0], function[1]

        assert callable(function), f"<function> is a {type(function)} not a function"
        assert callable(loss_function), f"<loss_function> is a {type(loss_function)} not a function"

        self.__function:"function" = function
        self.__error:float = None
        self.__robust:bool = False
        self.__limiar:float = 1.92
        self.__print:float = print
        
        temp:tuple = tuple(signature(function).parameters.keys())
        assert len(temp) >= 2, "Your function must have at least two parameters. Example f(x, b) = x*b = y"

        # Definindo regressora
        if sum([True if args_.lower().find("x") == 0 else False for args_ in temp]) > 1:
            self.regressors = []
            for args_ in temp:
                if args_.lower().find("x") == 0:
                    self.regressors.append(args_)
        elif regressors == None:
            assert "x" in temp, "The passed function must have the parameter 'x' or explicitly specify the regressors with the parameter 'regressors'"
            self.regressors = ["x"]
        elif type(regressors) == int or type(regressors) == float:
            self.regressors = [regressors]
        else:
            self.regressors = regressors

        # Argumentos da função
        self.__args_function:dict = {}
        self.params:list = []
        for parameter in temp:
            if parameter not in self.regressors:
                self.params.append(parameter)
                self.__args_function[parameter] = 0.1
        
        # Variáveis bloqueadas
        self.__lock:dict = {}

        # Pesos dos parâmetros
        self.weights:dict = {key:1 for key in self.__args_function}

        # Conferir quantos parâmetros de volta existem
        self.__len_y:int = None
        for i in [0.1, 1, 2, 5]:
            if self.__len_y == None:
                try:
                    self.__len_y:int = len(self.__function(*[i for _ in range(len(self.regressors) + len(self.__args_function))]))
                except:
                    pass
        if self.__len_y == None:
            self.__len_y:int = 1

        # Loss function
        if self.__len_y > 1:
            self.__loss_function:"function" = least_squares_multivariate
        else:
            self.__loss_function:"function" = loss_function

        # Definindo seed e critério de parada
        self.__seed = None
        self.iterations:int = min(50 * len(self.__args_function.keys()) * self.__len_y, 500 * self.__len_y) # Quanto mais parâmetros mais iterações eu precisso para que o valor mude

    def __call__(self, list_prediction:list = None, **x_args):
        return self.prediction(list_prediction, **x_args)

    def __eq__(self, obj) -> bool:
        """
        Confere se as funções de regressões são as mesmas.
        """
        assert type(obj) == Regression, "Only Regression objects can be compared"
        return True if self.__function == obj.__function else False

    def __repr__(self) -> str:
        """
        Mostra os argumentos da classe.
        """
        output:str = f"FUNCTION: {self.__function.__name__}"
        if self.__error != None:
            output += f"\nLOSS FUNCTION({self.__loss_function.__name__}): {self.__error:0.08f}"
        else:
            output += f"\nLOSS FUNCTION: {self.__loss_function.__name__}"
        output += f"\nREGRESSORS: {', '.join(self.regressors)}"
        output += f"\nLENTH OUTPUT: {self.__len_y} ({'MULTIVARIATE' if self.__len_y > 1 else 'UNIVARIATE'})"
        if len(self.__lock) > 0:
            output += f"\nLOCK PARAMS: {', '.join(self.__lock)}"
        output += "\nPARAMS:"
        for arg in self.__args_function.keys():
            output += f"\n  {arg} = {self.__args_function[arg]:0.08f} (w: {self.weights[arg]:0.08f})"
        return output

    def __len__(self) -> list:
        """
        Retorna a dimensão sendo:
            dim_regressorses
        """
        return len(self.regressors)

    def __getitem__(self, index:str) -> float:
        """
        Mostra o argumento especifico pedido.
        """
        assert type(index) == str, "The index must be a character(chr)"
        assert index in self.params, f"The index '{index}' must exist in params '{', '.join(self.params)}'"
        return self.__args_function[index]

    def __setitem__(self, index:str, value:float) -> float:
        """
        Inicia um valor em um ponto especifico.
        """
        assert type(index) == str, "The index must be a character(chr)"
        assert index in self.params, f"The index '{index}' must exist in params '{', '.join(self.params)}'"
        assert type(value) == int or type(value) == float, f"<value> must to be a int or float not a {type(value)}"
        self.change(**{index:value})

    def __add__(self, obj) -> "Regression":
        """
        + para mistura.
        self + obj
        """
        modified_class = eval(self.__generic_function(obj, operator = "+"))
        self.__rshift__(modified_class)
        if type(self) == type(obj):
            modified_class.__function.__name__ = f"{self.__function.__name__}_add_{obj.__function.__name__}"
        else:
            modified_class.__function.__name__ = f"{self.__function.__name__}_add_number"
        return modified_class

    def __sub__(self, obj) -> "Regression":
        """
        - para mistura.
        self - obj
        """
        modified_class = eval(self.__generic_function(obj, operator = "-"))
        self.__rshift__(modified_class)
        if type(self) == type(obj):
            modified_class.__function.__name__ = f"{self.__function.__name__}_sub_{obj.__function.__name__}"
        else:
            modified_class.__function.__name__ = f"{self.__function.__name__}_sub_number"
        return modified_class

    def  __mul__(self, obj) -> "Regression":
        """
        * para mistura.
        self * obj
        """
        modified_class = eval(self.__generic_function(obj, operator = "*"))
        self.__rshift__(modified_class)
        if type(self) == type(obj):
            modified_class.__function.__name__ = f"{self.__function.__name__}_mul_{obj.__function.__name__}"
        else:
            modified_class.__function.__name__ = f"{self.__function.__name__}_mul_number"
        return modified_class

    def __truediv__(self, obj) -> "Regression":
        """
        / para mistura.
        self / obj
        """
        modified_class = eval(self.__generic_function(obj, operator = "/"))
        self.__rshift__(modified_class)
        if type(self) == type(obj):
            modified_class.__function.__name__ = f"{self.__function.__name__}_truediv_{obj.__function.__name__}"
        else:
            modified_class.__function.__name__ = f"{self.__function.__name__}_truediv_number"
        return modified_class

    def __pow__(self, obj) -> "Regression":
        """
        ** para mistura.
        self ** obj
        """
        modified_class = eval(self.__generic_function(obj, operator = "**"))
        self.__rshift__(modified_class)
        if type(self) == type(obj):
            modified_class.__function.__name__ = f"{self.__function.__name__}_pow_{obj.__function.__name__}"
        else:
            modified_class.__function.__name__ = f"{self.__function.__name__}_pow_number"
        return modified_class

    def __rshift__(self, obj) -> None:
        """
        Passa os parâmetros de:
        self para obj
        """
        assert type(obj) == Regression, "This operation only allows another instance of the Regression class"
        for arg in self.__args_function.keys():
            if arg in obj.__args_function.keys():
                obj.__args_function[arg] = self.__args_function[arg]
        obj._Regression__robust = self.__robust

    def __lshift__(self, obj) -> None:
        """
        Passa os parâmetros de:
        obj para self
        """
        obj.__rshift__(self)

    def save(self, name:str) -> bool:
        """
        Salva os argumentos de memória em um arquivo chamado <name>.memory
        """

        function:str = getsource(self.__function)
        loss_function:str = getsource(self.__loss_function)

        # Tratando identação
        function:list = function.split("\n")
        while "\t" in function[0][0] or " " in function[0][0]:
            for i in range(len(function)):
                function[i] = function[i][1:]
        function = "\n".join(function)

        loss_function:list = loss_function.split("\n")
        while "\t" in loss_function[0][0] or " " in loss_function[0][0]:
            for i in range(len(loss_function)):
                loss_function[i] = loss_function[i][1:]
        loss_function = "\n".join(loss_function)

        # Salvar tudo
        with open(f"{name.replace('.memory', '')}.memory", "w") as arq:
            arq.write(f"{self.__args_function}\n|||\n{self.regressors}\n|||\n{function}\n|||\n{loss_function}\n|||\n{self.__error}\n|||\n{self.__seed}\n|||\n{self.iterations}\n|||\n{self.__len_y}")
            
        return True

    def open(self, name:str) -> bool:
        """
        Abre os argumentos de memória em um arquivo chamado <name>.memory
        """

        with open(f"{name.replace('.memory', '')}.memory", "r") as arq:
            arq_final:list = arq.read().split("\n|||\n")

        self.__args_function:dict = eval(arq_final[0])
        self.weights:dict = {key:1 for key in self.__args_function}

        self.regressors:list = eval(arq_final[1])

        func_code:str = arq_final[2]
        local_vars = {}
        exec(func_code, globals(), local_vars)
        self.__function = next(iter(local_vars.values()))

        func_code:str = arq_final[3]
        local_vars = {}
        exec(func_code, globals(), local_vars)
        self.__loss_function = next(iter(local_vars.values()))

        self.__error:float = float(arq_final[4]) if arq_final[4] != "None" else None
        self.__seed:int = int(arq_final[5]) if arq_final[5] != "None" else None
        self.iterations:int = int(arq_final[6])
        self.__len_y:int = int(arq_final[7])

        return True

    def loss_function(self, function:"function") -> None:
        """
        Muda a função de perda usada para a regressão.
        """
        self.__loss_function:"function" = function

    def operation(self, obj, operator:str) -> "Regression":
        """
        <operator> para mistura.
        self <operador> obj
        """
        modified_class = eval(self.__generic_function(obj, operator = operator))
        if type(self) == type(obj):
            modified_class.__function.__name__ = f"{self.__function.__name__}_generic_{obj.__function.__name__}"
        else:
            modified_class.__function.__name__ = f"{self.__function.__name__}_generic_number"
        return modified_class


    def __generic_function(self, obj:"Regression", operator:str) -> str:
        """
        Cria a função genérica e deixa como variável global as funções necessárias.
        """

        if type(obj) == int or type(obj) == float:
            globals()[f"{self.__function.__name__}"] = self.__function
            
            all_parameters = list(set(self.regressors) | set(self.params))
            all_regressorss = list(self.regressors)
            
            inputs_1 = ""
            for input_ in list(set(self.regressors) | set(self.params)):
                inputs_1 += f"{input_} = {input_},"
            return f"Regression(lambda {', '.join(all_parameters)} : {self.__function.__name__}({inputs_1}) {operator} {obj}, regressors = {all_regressorss})"

        else:
            assert type(obj) == type(self), f"{obj} must be of type class 'Regression'"
            
            globals()[f"{self.__function.__name__}"] = self.__function
            globals()[f"{obj.__function.__name__}"] = obj.__function
            
            all_parameters = list(set(self.regressors) | set(obj.regressors) | set(self.params) | set(obj.params))
            all_regressorss = list(set(self.regressors) | set(obj.regressors))
            
            inputs_1 = ""
            for input_ in list(set(self.regressors) | set(self.params)):
                inputs_1 += f"{input_} = {input_},"

            inputs_2 = ""
            for input_ in list(set(obj.regressors) | set(obj.params)):
                inputs_2 += f"{input_} = {input_},"
            
            return f"Regression(lambda {', '.join(all_parameters)} : {self.__function.__name__}({inputs_1}) {operator} {obj.__function.__name__}({inputs_2}), regressors = {all_regressorss})"

    def set_seed(self, seed:int) -> None:
        """
        Coloca uma seed.
        """
        assert type(seed) == int, "The seed must be an integer(int)!"
        self.__seed = seed

    def lock(self, **args) -> None:
        """
        Atualiza as variáveis que devem estar travadas.
        """
        for arg in args:
            assert arg in self.__args_function.keys(), f"'{arg}' not in parameters of the function {self.__function.__name__}"
            self.__lock[arg] = args[arg]

    def change(self, **args) -> None:
        """
        Troca um valor para que o chute inicial dele seja diferente.
        """
        for arg in args:
            assert arg in self.__args_function.keys(), f"'{arg}' not in parameters of the function {self.__function.__name__}"
            self.__args_function[arg] = args[arg]

    def change_all(self, value:float) -> None:
        """
        Troca todos os valores para que o chute inicial dele seja diferente.
        """
        assert type(value) == int or type(value) == float or type(value) == list or type(value) == tuple, f"<value> must to be a float or int not {type(value)}"

        if type(value) == list or type(value) == tuple:
            assert len(value) == 2, f"If <value> has to be 2 values, [min, max]"

            if self.__seed is not None:
                seed(self.__seed)
            
            for arg in self.__args_function.keys():
                self.__args_function[arg] = random()*(max(value) - min(value)) - min(value)

        else:
            for arg in self.__args_function.keys():
                    self.__args_function[arg] = value

    def prediction(self, list_prediction:list = None, **x_args) -> float:
        """
        Faz a previsão de f(...) = y.

        Args:
            list_prediction (list): É uma lista de listas, faz a predição com esses valores.
            x_args (**dict): Faz a predição de acordo com os valores pedidos.

        Returns:
            float: Valor predito.
        """

        if type(list_prediction) == list or type(list_prediction) == tuple: # Caso o usuário tenha passado uma série de valores para a predição
            assert type(list_prediction[0]) == list or type(list_prediction[0]) == tuple, "If you want to pass a series of values​to predict, you should pass the list of lists of values with the regressors parameters"
            assert min(map(len, list_prediction)) == max(map(len, list_prediction)) == len(self.regressors), f"Your list of lists must be {len(list_prediction)} by {len(self.regressors)} in size\nlen min: {min(map(len, list_prediction))}\nlen max: {max(map(len, list_prediction))}"

            results = []
            x_args = {}
            for values in list_prediction:
                for i in range(len(self.regressors)):
                    x_args[self.regressors[i]] = values[i]
                results.append(self.__function(**x_args, **self.__args_function))
                
            return results
        
        else: # Caso o usuário tenha passado valores específicos para a predição
            assert len(set(x_args.keys()) & set(self.__args_function.keys())) == 0, f"You cannot pass a parameter as a regressors that is already being used as a prediction parameter.\n  regressors parameters: {', '.join(x_args.keys())}\n  Predictor parameters: {', '.join(self.__args_function.keys())}"
            assert set(x_args.keys()) == set(self.regressors), f"Pass regressors parameters correctly\n  regressors parameters passed: {', '.join(x_args.keys())}\n  Expected regressors parameters: {', '.join(self.regressors)}"

            return self.__function(**x_args, **self.__args_function)

    def variance_error(self, data:[list]) -> float:
        """
        Calcula a variância do erro
        """
        y_:list = regression.prediction([data_i[:-1] for data_i in data])
        y:list = [data_i[-1] for data_i in data]
        return sum([(yi - y_i)**2 for yi, y_i in zip(y_, y)])/len(y)

    def run(self, data:[list], precision:float = 0.01, booster:float = 100, especific_precision:list = None, adaptive:bool = True) -> None:
        """
        Faz a regressão.

        Args:
            data(list(list)): Lista de listas com x e y.
            precision(float): Numero da precisão para achar os parâmetros esperados.
            booster(float): Numero que é multiplicado pela precisão para decidir o limite superior de treino.
            especific_precision(list): Lista de valores específicos para precisão específica.
            adaptive(bool): Se o método deve dar pesos diferentes para cada parâmetro (recomendado).
        """

        assert type(data) == list, f"The data must be a list of lists not {type(data)}"
        assert type(data[0]) == list, f"The data must be a list of lists not {type(data[0])}"
        
        for i in range(len(data)):
            if type(data[i][-1]) == list:
                data[i] = [*data[i][:-1], *data[i][-1]]
            
        assert len(data[0]) == len(self.regressors) + self.__len_y, f"The list of lists must have an x_n and a y parameter\n For example [[x_0, x_1, ..., y_1, ...], [x_0, x_1, ..., y_1, ...], ...] or [[x_0, x_1, ..., [y1, ...]], [x_0, x_1, ..., [y1, ...]], ...]\n\tSize of the passed list: {len(data[0])} | {data[0]}\n\tExpected size: {len(self.regressors) + self.__len_y} | {self.regressors} + [y's]"
        assert (k := list(map(len, data))) and max(k) == min(k), "The data list must be the same size in all itens"
        assert type(precision) == int or type(precision) == float, "Precision has to be a float or int"

        # Iniciando a seed
        if self.__seed is not None:
            seed(self.__seed)
        if self.__print:
            print(f"\rseed: {self.__seed}", end = "")

        # Método adaptativo
        if adaptive:
            self.adjust_weights(data = data, value = precision)
        if self.__print:
            print(f"\rAdaptive mode: {adaptive}", end = "")
        
        # Pegando y esperado
        if self.__len_y > 1:
            y_expected = [data[i][-self.__len_y:] for i in range(len(data))]
        else:
            y_expected = [data[i][-1] for i in range(len(data))]
        if self.__print:
            print(f"\ry_expected: True", end = "")

        # Salvando argumentos iniciais para a função
        args_temp:dict = {}
        for parameter in self.__args_function.keys():
            if parameter not in self.__lock.keys():
                args_temp[parameter] = self.__args_function[parameter]
            else:
                args_temp[parameter] = self.__lock[parameter] # Caso a variável deva estar travada
                if self.__print:
                    print(f"\rLock {parameter}: True", end = "")

        if type(especific_precision) == list:
            precision_final, precision = 1, len(especific_precision)
            index_precision = 0
            if self.__print:
                print(f"\r|{' ' * len(especific_precision)}| (Precision: {especific_precision[index_precision]}) (Model: {self.__function.__name__})", end = "")
        else:
            precision_final, precision = precision/2, precision * booster
            precision_k = 0
            if self.__print:
                print(f"\r|{' '*9}| (Precision: {precision} | Final Precision: {precision_final}) (Model: {self.__function.__name__})", end = "")

        all_iterations:int = 1
        while precision >= precision_final: # Vai diminuindo a variação da busca
            with_no_iteration = 0
            if type(especific_precision) == list:
                precision:float = especific_precision[index_precision]
                index_precision += 1

            while with_no_iteration < self.iterations:
                with_no_iteration += 1
                
                # y predito
                y_predicted:list = [self.__function(**{self.regressors[i]: x[i] for i in range(len(x) - self.__len_y)}, **args_temp) for x in data]

                # Resultado dos minimos quadrados
                result:float = self.__loss_function(y_predicted, y_expected)

                # Atualizando melhores parâmetros para regressora
                if not "best_result" in locals():
                    best_result:float = result
                    best_args:dict = deepcopy(args_temp)

                if result < best_result:
                    with_no_iteration = 0
                    best_result:float = result
                    best_args:dict = deepcopy(args_temp)
                else:
                    args_temp:dict = deepcopy(best_args)

                for parameter in self.__args_function.keys():
                    if parameter not in self.__lock.keys():
                        args_temp[parameter] += (random()*precision - precision/2)*self.weights[parameter]

                if all_iterations % 1_000 == 0:
                    if type(especific_precision) == list:
                        if self.__print:
                            precision:int = len(especific_precision)
                            print(f"\r|{'#' * precision_final}{' ' * (len(especific_precision) - precision_final)}| (Precision: {especific_precision[precision_final - 1]}) (Model: {self.__function.__name__})", end = "")
                    else:
                        if self.__print:
                            print(f"\r|{'#' * precision_k}{' '*(8 - precision_k)}| (Precision: {precision} | Final Precision: {precision_final}) (Model: {self.__function.__name__})", end = "")
                    if self.__print:
                        if len(list(self.__args_function.keys())) <= 5:
                            values:str = [f"{key}: {values:7.04f}" for key, values in zip(best_args.keys(), best_args.values())]
                            print(f" || {' | '.join(values)}", end = "")
                    self.adjust_weights(data = data, value = precision)
                        
                all_iterations += 1
                        
            # Aumenta a precisão
            if type(especific_precision) == list:
                if self.__print:
                    print(f"\r|{'#' * precision_final}{' ' * (len(especific_precision) - precision_final)}| (Precision: {especific_precision[precision_final - 1]}) (Model: {self.__function.__name__})", end = "")
                precision_final += 1
                precision:int = len(especific_precision)
            else:
                precision_k += 1
                precision /= 2
                if self.__print:
                    print(f"\r|{'#' * precision_k}{' '*(8 - precision_k)}| (Precision: {precision} | Final Precision: {precision_final}) (Model: {self.__function.__name__})", end = "")
            if self.__print:
                if len(list(self.__args_function.keys())) <= 5:
                    values:str = [f"{key}: {values:7.04f}" for key, values in zip(best_args.keys(), best_args.values())]
                    print(f" || {' | '.join(values)}", end = "")

            if adaptive:
                self.adjust_weights(data = data, value = precision)

        if (self.__print) and (not self.__robust):
            print(" (end)")                

        # Salva o resultado
        self.__args_function = best_args
        self.__error = best_result/len(data)
        self.__robust:bool = False

    def adjust_weights(self, data:[list], value:float = 1) -> None:
        """
        Função que ajusta pesos das mudanças dos parâmetros

        Args:
            data(list(list)): lista de listas com x e y.
            value(float): Valor que define quanto será o salto para teste
        """
        if self.__len_y > 1:
            X:[list] = [data_i[:-self.__len_y] for data_i in data]
            y:list = [data_i[self.__len_y:] for data_i in data]
        else:
            X:[list] = [data_i[:-1] for data_i in data]
            y:list = [data_i[-1] for data_i in data]

        # Confere erro inicial:
        initial_error:float = self.__loss_function(self.prediction(X), y)

        sum_errors:list = []
        errors:dict = {}
        for key in self.__args_function: # Ver erros marginais de cada variável
            self.__args_function[key] += value
            errors[key] = self.__loss_function(self.prediction(X), y)
            self.__args_function[key] -= value

            # Calculando novos pesos
            self.weights[key] = 1/(abs(errors[key] - initial_error) + 1)
            sum_errors.append(1/(abs(errors[key] - initial_error) + 1))

        self.weights = {key: value/max(sum_errors) for key, value in self.weights.items()} #Normalizando pesos

    def __new_data(self, data:[list], limiar:float) -> [list]:        
        if self.__len_y == 1:
            y_:list = self.prediction([data_i[:-1] for data_i in data])
            y:list = [data_i[-1] for data_i in data]
            variance:float = sum([(yi - y_i)**2 for yi, y_i in zip(y_, y)])/len(y)
            sd:list = variance**(1/2)

            new_data:list = []
            for y_i, yi, i in zip(y_, y, range(len(y))):
                if abs(yi - y_i) < limiar * sd:
                    new_data.append(data[i])

        else:
            def distancia(ponto1:list, ponto2:list):
                return sum((p1 - p2) ** 2 for p1, p2 in zip(ponto1, ponto2))
                
            y_:list = self.prediction([data_i[:-self.__len_y] for data_i in data])
            y:list = [data_i[:self.__len_y] for data_i in data]
            variance:float = sum([distancia(y_i, yi) for y_i, yi in zip(y_, y)])/len(y)
            sd:list = variance**(1/2)

            new_data:list = []
            for y_i, yi, i in zip(y_, y, range(len(y))):
                if distancia(yi, y_i) < limiar * sd:
                    new_data.append(data[i])

        return new_data
        

    def run_robust(self, data:[list], precision:float = 0.001, booster:float = 100, especific_precision:list = None, limiar:float = 1.92, adaptive:bool = True) -> [list]:
        """
        Faz uma regressão robusta usando o valor de limiar para o erro dos dados.

        Args:
            data(list(list)): lista de listas com x e y.
            precision(float): Numero da precisão para achar os parâmetros esperados.
            booster(float): Numero que é multiplicado pela precisão para decidir o limite superior de treino.
            especific_precision(list): Lista de valores específicos para precisão específica.
            adaptive(bool): Se o método deve dar pesos diferentes para cada parâmetro (recomendado).
        Return:
            [list]: Nova lista com valores robustos (não anomalos)
        """
        
        assert type(data) == list, f"The data must be a list of lists not {type(data)}"
        assert type(data[0]) == list, f"The data must be a list of lists not {type(data[0])}"
        
        for i in range(len(data)):
            if type(data[i][-1]) == list:
                data[i] = [*data[i][:-1], *data[i][-1]]
            
        assert len(data[0]) == len(self.regressors) + self.__len_y, f"The list of lists must have an x_n and a y parameter, for example [[x_0, x_1, ..., y], [x_0, x_1, ..., y], ...]\n\tSize of the passed list: {len(data[0])}\n\tExpected size: {len(self.regressors) + 1}"
        assert (k := list(map(len, data))) and max(k) == min(k), "The data list must be the same size in all itens"
        assert type(precision) == int or type(precision) == float, "Precision has to be a float or int"
        assert limiar > 0, f"the limit must be greater than 0"           

        len_old_data = len(data)
        iteration:int = 1
        while True:
            self.__robust:bool = True
            self.run(data = data,
                     precision = precision,
                     booster = booster,
                     especific_precision = especific_precision,
                     adaptive = adaptive)

            data:[list] = self.__new_data(data = data, limiar = limiar)

            if self.__print:
                print(f" (Iteration: {iteration} | len(data): {len(data)})")
            iteration += 1

            if len_old_data == len(data):
                break

            len_old_data = len(data)

        self.__limiar:float = limiar
        self.__robust:bool = True
        return data            
        

    def __animation_run(self, data:[list], precision:float = 0.001, booster:float = 100, especific_precision:list = None) -> None:
        """
        Função modificada para fazer animações.
        
        Faz a regressão.

        Args:
            data(list(list)): lista de listas com x e y.
            precision(float): Numero da precisão para achar os parâmetros esperados.
            booster(float): Numero que é multiplicado pela precisão para decidir o limite superior de treino.
            especific_precision(list): Lista de valores específicos para precisão específica.
        """
        from make_animation import plot_expected_and_save

        assert type(data) == list, f"The data must be a list of lists not {type(data)}"
        assert type(data[0]) == list, f"The data must be a list of lists not {type(data[0])}"
        assert len(data[0]) == len(self.regressors) + 1, f"The list of lists must have an x_n and a y parameter, for example [[x_0, x_1, ..., y], [x_0, x_1, ..., y], ...]\n\tSize of the passed list: {len(data[0])}\n\tExpected size: {len(self.regressors) + 1}"
        assert (k := list(map(len, data))) and max(k) == min(k), "The data list must be the same size in all itens"
        assert type(precision) == int or type(precision) == float, "Precision has to be a float or int"

        # Iniciando a seed
        if self.__seed is not None:
            seed(self.__seed)

        # Pegando y esperado
        y_expected = [data[i][-1] for i in range(len(data))]

        # Salvando argumentos iniciais para a função
        args_temp:dict = {}
        for parameter in self.__args_function.keys():
            if parameter not in self.__lock.keys():
                args_temp[parameter] = self.__args_function[parameter]
            else:
                args_temp[parameter] = self.__lock[parameter] # Caso a variável deva estar travada

        iteration_:int = 0
        qnt_:int = 0
        qnt_plot:list = [int(i + 1.04**i) for i in range(5_000)]
        
        if type(especific_precision) == list:
            precision_final, precision = 1, len(especific_precision)
            index_precision = 0
        else:
            precision_final, precision = precision/2, precision * booster
        
        while precision >= precision_final: # Vai diminuindo a variação da busca
            with_no_iteration = 0
            if type(especific_precision) == list:
                precision:float = especific_precision[index_precision]
                index_precision += 1
                
            while with_no_iteration < self.iterations:
                iteration_ += 1
                with_no_iteration += 1
                
                # y predito
                y_predicted:list = []
                for *x, _ in data:

                    # Separando as variáveis regressoras
                    x_args:dict = {}
                    for i in range(len(x)):
                        x_args[self.regressors[i]] = x[i]

                    # Fazendo a predição
                    y_predicted.append(self.__function(**x_args, **args_temp))

                # Resultado dos minimos quadrados
                result:float = self.__loss_function(y_predicted, y_expected)

                # Atualizando melhores parâmetros para regressora
                if not "best_result" in locals():
                    best_result:float = result
                    best_args = deepcopy(args_temp)
                    self.__args_function = best_args
                    plot_expected_and_save(self, data, name = f"img_{int(iteration_):04.00f}")
                    qnt_ += 1

                if result < best_result:
                    with_no_iteration = 0
                    best_result:float = result
                    best_args = deepcopy(args_temp)
                    self.__args_function = best_args
                    if qnt_ in qnt_plot:
                        plot_expected_and_save(self, data, name = f"img_{int(iteration_):04.00f}")
                    qnt_ += 1
                    
                else:
                    args_temp = deepcopy(best_args)

                for parameter in self.__args_function.keys():
                    if parameter not in self.__lock.keys():
                        args_temp[parameter] += (random()*precision - precision)*self.weights[parameter]
                        
            # Aumenta a precisão
            if type(especific_precision) == list:
                precision_final += 1
                precision:int = len(especific_precision)
            else:
                precision /= 2

        # Salva o resultado
        self.__args_function = best_args
        self.__error = best_result/len(data)
