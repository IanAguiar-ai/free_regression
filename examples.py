from free_regression import Regression

def regressao_simples(x:float, a:float, b:float):
    return a*x + b

def regressao_2(x:float, a:float, b:float, c:float):
    return a*x**2 + b*x + c

def regressao_nao_continua(x:float, a:float):
    if x < 10:
        return -1 * x * a
    elif x > 15:
        return  x * a
    else:
        return x

def funcao_errada(a:float, b:float, c:float):
    return a. b, c

def regressao_2_regressores(x_1:float, x_2:float, a:float, b:float):
    return x_1*a + x_2*b**2

if __name__ == "__main__":
##    dado = [[0, 3], [1, 2], [1, 4], [2, 3]]
##    teste_1 = Regression(regressao_simples)
##    teste_1.run(dado)
##    print(teste_1)
##    
##    dado = [[x, regressao_simples(x, a = 15, b = 1)] for x in range(10)]
##    teste_1.run(dado)
##    print(teste_1)
##
##    dado = [[x, regressao_2(x, a = 15, b = -7, c = -4)] for x in range(10)]
##    teste_2 = Regression(regressao_2)
##    teste_2.iterations = 100
##    teste_2.run(dado, precision = 0.0001)
##    print(teste_2)
##
##    dado = [[x, regressao_nao_continua(x, a = 7)] for x in range(20)]
##    teste_3 = Regression(regressao_nao_continua)
##    teste_3.run(dado)
##    print(teste_3)

    dado = [[x, int(x*3 % 11), regressao_2_regressores(x, int(x*3 % 11), a = 3, b = 10)] for x in range(5)]
    teste_4 = Regression(regressao_2_regressores, regressor = ["x_1", "x_2"])
    teste_4.run(dado, precision = 0.001)
    print(teste_4)

    
