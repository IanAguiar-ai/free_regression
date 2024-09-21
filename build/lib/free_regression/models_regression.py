"""
Funções para criação de regressões
"""

def generate_regression(regressors:int, degree:int = 1) -> ("function", list):
    """
    Gera uma regressão com o grau pedido

    Retorna a função e os regressores
    """
    assert type(regressors) == int, "<regressors> must be an integer"
    assert type(degree) == int, "<degree> must be an integer"
    
    function:str = f"lambda {', '.join(['x_' + str(i) for i in range(1, regressors + 1)])}, {', '.join(['b_' + str(i) for i in range(1, regressors*degree + 1)])}: "
    final_beta = 1
    for beta_degree in range(1, degree + 1):
        for beta in range(1, regressors + 1):
            function += f"b_{final_beta}*x_{beta}**{beta_degree} + "
            final_beta += 1
    function:str = function[:-3]
    final_function = eval(function)
    final_function.__name__ = f"regression_with_{regressors}_regressors_and_{degree}_degrees"
    return final_function, ['x_' + str(i) for i in range(1, regressors + 1)]

if __name__ == "__main__":
    from free_regression import Regression

    teste_1 = Regression(*generate_regression(regressors = 2, degree = 2))
    teste_1.run([[1, 4, 5], [6, 3, 2]])
    print(f"{teste_1}\n")
    print(teste_1.prediction(x_1 = 1, x_2 = 4))
    print(teste_1.prediction(x_1 = 6, x_2 = 3))
