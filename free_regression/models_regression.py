"""
Funções para criação de regressões
"""

def generate_regression(regressors:int, degree:int = 1) -> ("function", list):
    """
    Gera uma regressão com o grau pedido.
    Retorna a função e os regressores.

    Args:
        regressors (list): Lista de nomes dos regressores.
        degree (int): Grau da regressão, padrão é 1.

    Returns:
        function: Função de regressão
    """
    assert type(regressors) == int, "<regressors> must be a integer"
    assert type(degree) == int, "<degree> must be a integer"
    assert regressors > 0, "<regressors> must be at least 1"
    assert degree > 0, "<degree> must be at least 1"
 
    function:str = f"lambda {', '.join(['x_' + str(i) for i in range(1, regressors + 1)])}, {', '.join(['b_' + str(i) for i in range(1, regressors*degree + 1)])}, b: "
    final_beta = 1
    for beta_degree in range(1, degree + 1):
        for beta in range(1, regressors + 1):
            function += f"b_{final_beta}*x_{beta}**{beta_degree} + "
            final_beta += 1
    function:str = function + "b"
    final_function = eval(function)
    final_function.__name__ = f"regression_with_{regressors}_regressors_and_{degree}_degrees"
    
    return final_function, ['x_' + str(i) for i in range(1, regressors + 1)]

def generate_mlp(regressors:int, neurons:int = 1) -> ("function", list):
    """
    Gera um MLP com a quantidade de regressors e neurons pedidos.
    Retorna a função MLP e os nomes dos regressores.

    Args:
        regressors (list): Lista de nomes dos regressores.
        neurons (int): Quantidade de neurônios na camada intermediaria, padrão é 1.

    Returns:
        function: Função MLP (Mult Layer Perceptron)
    """
    assert type(regressors) == int, "<regressors> must be a integer"
    assert type(neurons) == int, "<neurons> must be a integer"
    assert regressors > 0, "<regressors> must be at least 1"
    assert neurons > 0, "<neurons> must be at least 1"

    all_parameters:set = set()
    function:str = ""
    # Camada intemediaria
    for i in range(neurons):
        function += f"x_{i} = max("
        temp_function = f"("
        for j in range(regressors):
            temp_function += f"b_{i}_{j}*x{j} + "
            all_parameters.add(f"b_{i}_{j}")
            all_parameters.add(f"x{j}")
        temp_function += f"b_{i})"
        all_parameters.add(f"b_{i}")
        function += f"{temp_function}, {temp_function}/100)\n\t"

    # Solução
    function += f"return "
    for i in range(neurons):
        function += f"b__{i}*x_{i} + "
        all_parameters.add(f"b__{i}")
    function += f"b"
    all_parameters.add(f"b")

    function:str = f"def mlp_relu_with_{regressors}_regressors_and_{neurons}_neurons({', '.join(sorted(list(all_parameters)))}):\n\t" + function
    function += f"\nglobals()['mlp_relu_with_{regressors}_regressors_and_{neurons}_neurons'] = mlp_relu_with_{regressors}_regressors_and_{neurons}_neurons"
    final_function = exec(function)

    return globals()[f'mlp_relu_with_{regressors}_regressors_and_{neurons}_neurons'], [f"x{i}" for i in range(regressors)]

def generate_mlp_classifier(regressors:int, neurons:int = 1) -> ("function", list):
    """
    Gera um MLP com a quantidade de regressors e neurons pedidos.
    Retorna a função MLP com funções de ativação sigmoide e os nomes dos regressores.

    Args:
        regressors (list): Lista de nomes dos regressores.
        neurons (int): Quantidade de neurônios na camada intermediaria, padrão é 1.

    Returns:
        function: Função MLP (Mult Layer Perceptron)
    """
    assert type(regressors) == int, "<regressors> must be a integer"
    assert type(neurons) == int, "<neurons> must be a integer"
    assert regressors > 0, "<regressors> must be at least 1"
    assert neurons > 0, "<neurons> must be at least 1"

    all_parameters:set = set()
    function:str = "try:\n\t"
    # Camada intemediaria
    for i in range(neurons):
        function += f"\tx_{i} = "
        temp_function = f"("
        for j in range(regressors):
            temp_function += f"b_{i}_{j}*x{j} + "
            all_parameters.add(f"b_{i}_{j}")
            all_parameters.add(f"x{j}")
        temp_function += f"b_{i})"
        all_parameters.add(f"b_{i}")
        function += f"1/(1 + 2.7182818**(-{temp_function}))\n\t"

    # Solução
    function += f"\treturn 1/(1 + 2.7182818**(-("
    for i in range(neurons):
        function += f"b__{i}*x_{i} + "
        all_parameters.add(f"b__{i}")
    function += f"b)))\n\texcept:\n\t\treturn 0"
    all_parameters.add(f"b")

    function:str = f"def mlp_sigmoid_with_{regressors}_regressors_and_{neurons}_neurons({', '.join(sorted(list(all_parameters)))}):\n\t" + function
    function += f"\nglobals()['mlp_sigmoid_with_{regressors}_regressors_and_{neurons}_neurons'] = mlp_sigmoid_with_{regressors}_regressors_and_{neurons}_neurons"
    final_function = exec(function)

    return globals()[f'mlp_sigmoid_with_{regressors}_regressors_and_{neurons}_neurons'], [f"x{i}" for i in range(regressors)]

def generate_mlp_semi_classifier(regressors:int, neurons:int = 1) -> ("function", list):
    """
    Gera um MLP com a quantidade de regressors e neurons pedidos.
    Retorna a função MLP e os nomes dos regressores.

    Args:
        regressors (list): Lista de nomes dos regressores.
        neurons (int): Quantidade de neurônios na camada intermediaria, padrão é 1.

    Returns:
        function: Função MLP (Mult Layer Perceptron)
    """
    assert type(regressors) == int, "<regressors> must be a integer"
    assert type(neurons) == int, "<neurons> must be a integer"
    assert regressors > 0, "<regressors> must be at least 1"
    assert neurons > 0, "<neurons> must be at least 1"

    all_parameters:set = set()
    function:str = "try:\n\t"
    # Camada intemediaria
    for i in range(neurons):
        function += f"\tx_{i} = max("
        temp_function = f"("
        for j in range(regressors):
            temp_function += f"b_{i}_{j}*x{j} + "
            all_parameters.add(f"b_{i}_{j}")
            all_parameters.add(f"x{j}")
        temp_function += f"b_{i})"
        all_parameters.add(f"b_{i}")
        function += f"{temp_function}, {temp_function}/100)\n\t"

    # Solução
    function += f"\treturn 1/(1 + 2.7182818**(-("
    for i in range(neurons):
        function += f"b__{i}*x_{i} + "
        all_parameters.add(f"b__{i}")
    function += f"b)))\n\texcept:\n\t\treturn 0"
    all_parameters.add(f"b")

    function:str = f"def mlp_relu_sigmoid_with_{regressors}_regressors_and_{neurons}_neurons({', '.join(sorted(list(all_parameters)))}):\n\t" + function
    function += f"\nglobals()['mlp_relu_sigmoid_with_{regressors}_regressors_and_{neurons}_neurons'] = mlp_relu_sigmoid_with_{regressors}_regressors_and_{neurons}_neurons"
    final_function = exec(function)
    
    return globals()[f'mlp_relu_sigmoid_with_{regressors}_regressors_and_{neurons}_neurons'], [f"x{i}" for i in range(regressors)]

def generate_mlp_sigmoid_sum(regressors:int, neurons:int = 1) -> ("function", list):
    """
    Gera um MLP com a quantidade de regressors e neurons pedidos.
    Retorna a função MLP com funções de ativação sigmoide e os nomes dos regressores.

    Args:
        regressors (list): Lista de nomes dos regressores.
        neurons (int): Quantidade de neurônios na camada intermediaria, padrão é 1.

    Returns:
        function: Função MLP (Mult Layer Perceptron)
    """
    assert type(regressors) == int, "<regressors> must be a integer"
    assert type(neurons) == int, "<neurons> must be a integer"
    assert regressors > 0, "<regressors> must be at least 1"
    assert neurons > 0, "<neurons> must be at least 1"

    all_parameters:set = set()
    function:str = ""
    # Camada intemediaria
    for i in range(neurons):
        function += f"\tx_{i} = "
        temp_function = f"("
        for j in range(regressors):
            temp_function += f"b_{i}_{j}*x{j} + "
            all_parameters.add(f"b_{i}_{j}")
            all_parameters.add(f"x{j}")
        temp_function += f"b_{i})"
        all_parameters.add(f"b_{i}")
        function += f"1/(1 + 2.7182818**(-{temp_function}))\n\t"

    # Solução
    function += f"\treturn "
    for i in range(neurons):
        function += f"b__{i}*x_{i} + "
        all_parameters.add(f"b__{i}")
    function += f"b"
    all_parameters.add(f"b")

    function:str = f"def mlp_sigmoidsum_with_{regressors}_regressors_and_{neurons}_neurons({', '.join(sorted(list(all_parameters)))}):\n\t" + function
    function += f"\nglobals()['mlp_sigmoidsum_with_{regressors}_regressors_and_{neurons}_neurons'] = mlp_sigmoidsum_with_{regressors}_regressors_and_{neurons}_neurons"
    final_function = exec(function)

    return globals()[f'mlp_sigmoidsum_with_{regressors}_regressors_and_{neurons}_neurons'], [f"x{i}" for i in range(regressors)]

def generate_mlp_normals(regressors:int, neurons:int = 1, max_:float = 999_999_999) -> ("function", list):
    """
    Gera um MLP com a quantidade de regressors e neurons pedidos.
    Retorna a função MLP com funções de ativação Radial Basis Function (RBF) e os nomes dos regressores.

    Args:
        regressors (list): Lista de nomes dos regressores.
        neurons (int): Quantidade de neurônios na camada intermediaria, padrão é 1.
        max (float): Valor máximo para a resposta

    Returns:
        function: Função MLP (Mult Layer Perceptron)
    """
    assert type(regressors) == int, "<regressors> must be a integer"
    assert type(neurons) == int, "<neurons> must be a integer"
    assert regressors > 0, "<regressors> must be at least 1"
    assert neurons > 0, "<neurons> must be at least 1"

    all_parameters:set = set()
    function:str = ""
    # Camada intemediaria
    for i in range(neurons):
        function += f"\tx_{i} = "
        temp_function = f"("
        for j in range(regressors):
            temp_function += f"b_{i}_{j}*x{j} + "
            all_parameters.add(f"b_{i}_{j}")
            all_parameters.add(f"x{j}")
        temp_function += f"b_{i})"
        all_parameters.add(f"b_{i}")
        all_parameters.add(f"mean_{i}")
        all_parameters.add(f"var_{i}")
        function += f"2.718**-((({temp_function}-mean_{i})*({temp_function}-mean_{i}))/abs(var_{i}))\n\t"

    # Solução
    function += f"\treturn min({max_}, "
    for i in range(neurons):
        function += f"b__{i}*x_{i} + "
        all_parameters.add(f"b__{i}")
    function += f"b)"
    all_parameters.add(f"b")

    function:str = f"def mlp_rbfsum_with_{regressors}_regressors_and_{neurons}_neurons({', '.join(sorted(list(all_parameters)))}):\n\t" + function
    function += f"\nglobals()['mlp_rbfsum_with_{regressors}_regressors_and_{neurons}_neurons'] = mlp_rbfsum_with_{regressors}_regressors_and_{neurons}_neurons"
    final_function = exec(function)

    return globals()[f'mlp_rbfsum_with_{regressors}_regressors_and_{neurons}_neurons'], [f"x{i}" for i in range(regressors)]

def generate_distribuction(regressors:int, normals:int = 1) -> ("function", list):
    """
    Gera um MLP com a quantidade de regressors e neurons pedidos.
    Retorna a função MLP com funções de ativação Radial Basis Function (RBF) e os nomes dos regressores.

    Args:
        regressors (list): Lista de nomes dos regressores.
        normals (int): Quantidade de neurônios na camada intermediaria, padrão é 1.
        max (float): Valor máximo para a resposta

    Returns:
        function: Função geradora de somas de normais normalizadas
    """
    assert type(regressors) == int, "<regressors> must be a integer"
    assert type(normals) == int, "<neurons> must be a integer"
    assert regressors > 0, "<regressors> must be at least 1"
    assert normals > 0, "<normals> must be at least 1"

    all_parameters:set = set()
    function:str = ""
    # Camada intemediaria
    for i in range(normals):
        function += f"\tx_{i} = "
        temp_function = f"("
        for j in range(regressors):
            temp_function += f"x{j} + "
            all_parameters.add(f"x{j}")
        temp_function += f"0)"
        all_parameters.add(f"mean_{i}")
        all_parameters.add(f"var_{i}")
        function += f"2.718281**-((({temp_function}-mean_{i})*({temp_function}-mean_{i}))/abs(var_{i}))\n\t"

    # Solução
    function += f"\treturn ("
    for i in range(normals):
        function += f"x_{i} + "
    function += f" 0)/{normals}"

    function:str = f"def normals_sum_with_{regressors}_regressors_and_{normals}_neurons({', '.join(sorted(list(all_parameters)))}):\n\t" + function
    function += f"\nglobals()['normals_sum_with_{regressors}_regressors_and_{normals}_neurons'] = normals_sum_with_{regressors}_regressors_and_{normals}_neurons"
    final_function = exec(function)
    print(function)

    return globals()[f'normals_sum_with_{regressors}_regressors_and_{normals}_neurons'], [f"x{i}" for i in range(regressors)]

def asymmetric_normal(x:list, m:float, v1:float, v2:float, c:float) -> list:
    e:float = 2.7182
    return (e**(-1/2 * (x-m)**2/v1) if x > m else e**(-1/2 * (x-m)**2/v2)) * c


def generate_neural_network(struct:list = [2, 3, 2], activation:list = ["relu", "sigmoid"]) -> "function":
    """
    Gera uma rede neural genérica.
    Retorna a função de rede neural com a quantidade de inputs, neuronios, camadas e outputs indicados em struct.
    Com funções de ativação ditas em activation.

    Args:
        struct (list): Lista com inteiros de [outputs, neuronios_camada_1, ..., outputs]. (tamanho de N)
        activation (list) : Lista com [ativacao_camada_1, ..., ativacao_output]. (tamanho de N-1)

    Returns:
        function: Função de rede neural.

    Funções de ativação:
    - relu
    - lrelu
    - sigmoid
    - abs
    - float
    - int
    """
    
    assert type(struct) == list, f"struct must be a list"
    assert len(struct) > 1, f"len(struct) must be at least 2\nlen(struct): {len(struct)} < 2"
    assert len(activation) + 1 == len(struct), f" len(activation) + 1 must be len(struct)\n{len(activation)} + 1 != {len(struct)}"    

    var:str = [f"x_{i+1}" for i in range(struct[0])]
    name:str = f"new_generated_neural_network_{'_'.join(list(map(str, struct)))}"
    function_string:str = f"def new_generated_neural_network_{'_'.join(list(map(str, struct)))}(|params|):\n"
    function_string += "\tdef relu(x):\n\t\treturn x if x > 0 else 0\n"
    function_string += "\tdef lrelu(x):\n\t\treturn x if x > x/10 else 0\n"
    function_string += "\tdef sigmoid(x):\n\t\treturn 1/(1 + 2**(-x))\n"

    for neuron in range(struct[0]):
        function_string += f"\tc_0_{neuron} = x_{neuron + 1}\n"
        var.append(f"c_0_{neuron}")

    for camade in range(1, len(struct)):
        for neuron in range(struct[camade]):
            temporary:list = [f"c_{camade - 1}_{i}*b_{camade}_{neuron}_{i}" for i in range(struct[camade - 1])]
            function_string += f"\tc_{camade}_{neuron} = {activation[camade - 1]}(i_{camade}_{neuron} + {' + '.join(temporary)})\n"
            var.append(f"c_{camade}_{neuron}")
            var.extend([f"b_{camade}_{neuron}_{i}" for i in range(struct[camade - 1])])
            var.append(f"i_{camade}_{neuron}")

    if struct[-1] == 1:
        function_string += f"\treturn c_{len(struct) - 1}_0"
    else:
        temporary:list = [f"c_{len(struct) - 1}_{i}" for i in range(struct[-1])]
        function_string += f"\treturn [{', '.join(temporary)}]"

    function_string = function_string.replace("|params|", f"{', '.join(var)}")

    function_string += f"\nglobals()['{name}'] = {name}"
    final_function = exec(function_string)
    #print(function_string)
    return globals()[f"{name}"]

def generate_linear_spline(lines:int) -> "function":

    name:str = f"new_generated_linear_spline_{lines}"
    function_string = f"def {name}(|params|):\n"

    function_string += f"\tf_0 = lambda x : b_0 + b_1*x\n"
    for i in range(lines - 1):
        if i >= 1:
            function_string += f"\tl_{i} = abs(l_{i}) + l_{i-1}\n"
        function_string += f"\tf_{i+1} = lambda x : b_{i+2}*(x - l_{i}) if x > l_{i} else 0\n"
        
    all_functions:list = [f"f_{i}(x)" for i in range(lines)]

    function_string += f"\treturn {' + '.join(all_functions)}"

    var:list = ["x",
                *[f"b_{i}" for i in range(lines + 1)],
                *[f"l_{i}" for i in range(lines - 1)]]
    function_string:str = function_string.replace("|params|", ", ".join(var))

    #print(function_string)

    function_string += f"\nglobals()['{name}'] = {name}"
    final_function = exec(function_string)
    return globals()[f"{name}"]


if __name__ == "__main__":
    from free_regression import Regression
    from graphics import plot_expected

    modelo = Regression(generate_linear_spline(lines = 5), print = True)
    print(modelo)
    modelo.lock(l_0 = 20, l_1 = 20, l_2 = 20, l_3 = 20)

    dados = [[i, 0.02*i**2 - 2*i + 5] for i in range(100)]

    modelo.run(dados, especific_precision = [1, 0.5, 0.1])
    print(modelo)
    plot_expected(modelo, dados)

    1/0
    modelo = Regression(generated_neural_network(struct = [2, 3, 3, 1], activation = ['relu', 'relu', 'sigmoid']))

    print(modelo)

    1/0
##    teste_1 = Regression(*generate_distribuction(regressors = 2, normals = 2))
##    teste_1.set_seed(1)
##    teste_1.run([[1, 4, 1], [6, 3, 0]])
##    print(f"{teste_1}\n")
##    print(f"{teste_1.prediction([[1, 4], [6, 3]])}")
##
##    from free_regression import Regression
##    teste_1 = Regression(*generate_distribuction(regressors = 1, normals = 3))
##    teste_1.set_seed(2)
##    teste_1.change_all([0.5, 3])
##    teste_1.run([[0, 0], [1, 1], [2, 0], [2.3, 1], [2.5, 1], [3, 0]], precision = 0.01)
##    print(f"{teste_1}\n")
##    print(f"{teste_1.prediction([[0], [1], [2], [2.3], [2.5], [3]])}")
##
##    mlp_func, regressors = generate_mlp_sigmoid_sum(regressors = 2, neurons = 2)
##    teste_2 = Regression(mlp_func, regressors)
##    teste_2.change_all(0.1)
##    teste_2.run([[1, 4, 0], [6, 3, 1]])
##    print(teste_2.prediction([[1, 4], [6, 3]]))

    from random import random, seed
    from data import *
    from graphics import plot_series
    
    seed(1)

    def ar_2(y1, y2, b1, b2, b3):
        return y1 * b1 + y2 * b2 + b3

    phi = [0.45, -0.95]
    data = [1, 2]
    for i in range(200):
        data.append(phi[0] * data[-1] + phi[1] * data[-2] + 5 + (random() + random() + random()) )

        
    data = data_series(data, p = 2)
    model_ar_2 = Regression(ar_2, regressors = ["y1", "y2"])
    model_ar_2.set_seed(1)
    model_ar_2.iterations = 1000
    model_ar_2.run(data, especific_precision = [1, 0.1, 0.01])
    print(model_ar_2)
    a = model_ar_2.prediction([data[i][:-1] for i in range(len(data))])
    for i in range(len(data)):
        print(data[i][-1], a[i])

    plot_series(model_ar_2, data, size = (14, 8))

        

