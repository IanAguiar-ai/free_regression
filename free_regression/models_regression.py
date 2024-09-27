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
    assert type(regressors) == int, "<regressors> must be an integer"
    assert type(degree) == int, "<degree> must be an integer"
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
    assert type(regressors) == int, "<regressors> must be an integer"
    assert type(neurons) == int, "<neurons> must be an integer"
    assert regressors > 0, "<regressors> must be at least 1"
    assert neurons > 0, "<neurons> must be at least 1"

    all_parameters:set = set()
    function:str = ""
    # Camada intemediaria
    for i in range(neurons):
        function += f"x_{i} = max("
        all_parameters.add(f"x_{i}")
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
    Retorna a função MLP e os nomes dos regressores.

    Args:
        regressors (list): Lista de nomes dos regressores.
        neurons (int): Quantidade de neurônios na camada intermediaria, padrão é 1.

    Returns:
        function: Função MLP (Mult Layer Perceptron)
    """
    assert type(regressors) == int, "<regressors> must be an integer"
    assert type(neurons) == int, "<neurons> must be an integer"
    assert regressors > 0, "<regressors> must be at least 1"
    assert neurons > 0, "<neurons> must be at least 1"

    all_parameters:set = set()
    function:str = "try:\n\t"
    # Camada intemediaria
    for i in range(neurons):
        function += f"\tx_{i} = "
        all_parameters.add(f"x_{i}")
        temp_function = f"("
        for j in range(regressors):
            temp_function += f"b_{i}_{j}*x{j} + "
            all_parameters.add(f"b_{i}_{j}")
            all_parameters.add(f"x{j}")
        temp_function += f"b_{i})"
        all_parameters.add(f"b_{i}")
        function += f"1/(1 - 2.7182818**(-{temp_function}))\n\t"

    # Solução
    function += f"\treturn 1/(1 - 2.7182818**(-("
    for i in range(neurons):
        function += f"b__{i}*x_{i} + "
        all_parameters.add(f"b__{i}")
    function += f"b)))\n\texcept:\n\t\treturn 0"
    all_parameters.add(f"b")

    function:str = f"def mlp_sigmoid_with_{regressors}_regressors_and_{neurons}_neurons({', '.join(sorted(list(all_parameters)))}):\n\t" + function
    function += f"\nglobals()['mlp_sigmoid_with_{regressors}_regressors_and_{neurons}_neurons'] = mlp_sigmoid_with_{regressors}_regressors_and_{neurons}_neurons"
    final_function = exec(function)

    return globals()[f'mlp_sigmoid_with_{regressors}_regressors_and_{neurons}_neurons'], [f"x{i}" for i in range(regressors)]

if __name__ == "__main__":
    from free_regression import Regression
    teste_1 = Regression(*generate_regression(regressors = 2, degree = 2))
    teste_1.run([[1, 4, 5], [6, 3, 2]])
    print(f"{teste_1}\n")
    print(teste_1.prediction(x_1 = 1, x_2 = 4))
    print(teste_1.prediction(x_1 = 6, x_2 = 3))

    mlp_func, regressors = generate_mlp_classifier(regressors = 2, neurons = 2)
    teste_2 = Regression(mlp_func, regressors)
    teste_2.change_all(0.1)
    teste_2.run([[1, 4, 0], [6, 3, 1]])
    print(teste_2.prediction([[1, 4], [6, 3]]))
