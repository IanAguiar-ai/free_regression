from free_regression import Regression

def regressao_simples(x:float, a:float, b:float):
    return a*x + b

def regressao_grande(x:float, a:float, b:float, c:float, d:float):
    return a*x**3 + b*x**2 + c*x + d

def regressao_nao_continua(x:float, a:float):
    if x < 10:
        return -1 * x * a
    elif x > 15:
        return  x * a
    else:
        return x

def funcao_errada(a:float, b:float, c:float):
    return a. b, c

if __name__ == "__main__":
    dado = [[0, 3], [1, 2], [1, 4], [2, 3]]
    teste_1 = Regression(regressao_simples)
    teste_1.run(dado)
    print(teste_1)
    
    dado = [[x, regressao_simples(x, a = 15, b = 1)] for x in range(10)]
    teste_1.run(dado)
    print(teste_1)

##    dado = [[x, regressao_grande(x, a = 15, b = -1, c = -4, d = 2)] for x in range(10)]
##    teste_2 = Regression(regressao_grande)
##    teste_2.iterations = 100
##    teste_2.run(dado, precision = 0.0001)
##    print(teste_2)

    dado = [[x, regressao_nao_continua(x, a = 7)] for x in range(20)]
    teste_3 = Regression(regressao_nao_continua)
    teste_3.run(dado)
    print(teste_3)
