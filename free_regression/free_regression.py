def run(self, data:[list], precision:float = 0.001) -> None:
    """
    Função modificada para fazer animações.
    
    Faz a regressão.

    Args:
        data(list(list)): lista de listas com x e y.
        precision(float): Numero da precisão para achar os parâmetros esperados.
    """

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

    precision_final, precision = precision/2, precision * 100
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
                    x_args[self.regressors[i]] = x[i]

                # Fazendo a predição
                y_predicted.append(self.__function(**x_args, **args_temp))

            # Resultado dos minimos quadrados
            result = self.__loss_function(y_predicted, y_expected)

            # Atualizando melhores parâmetros para regressora
            if not "best_result" in locals():
                best_result:float = result
                best_args = deepcopy(args_temp)
                self.__memory.append(best_args)

            if result < best_result:
                with_no_iteration = 0
                best_result:float = result
                best_args = deepcopy(args_temp)
                self.__memory.append(best_args)
            else:
                args_temp = deepcopy(best_args)

            for parameter in self.__args_function.keys():
                if parameter not in self.__lock.keys():
                    args_temp[parameter] += random()*precision - precision/2
                    
        # Aumenta a precisão
        precision /= 2

    # Salva o resultado
    self.__args_function = best_args
    self.__error = best_result/len(data)

def make_animation(obj:"Regression", data:list, name:str = "animation") -> None:
    """
    Faz a animação desde que o objeto passado seja da classe passada
    """
    obj.run = run
    obj.run(data)

    # FUNÇÃO QUE SALVA AS IMAGENS DADO os parâmetros em obj.__memory

    # FUNÇÃO QUE MONTA O MP4
    
