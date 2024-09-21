# Biblioteca autoral
from free_regression import Regression

# Biblioteca de teste
import unittest
from sklearn.datasets import load_iris

def regressao_simples(x:float, a:float, b:float) -> float:
    return a*x + b

def regressao_2(x:float, a:float, b:float, c:float) -> float:
    return a*x**2 + b*x + c

def regressao_nao_continua(x:float, a:float) -> float:
    if x < 10:
        return -1 * x * a
    elif x > 15:
        return  x * a
    else:
        return x

def funcao_errada(a:float, b:float, c:float) -> float:
    return a. b, c

def regressao_2_regressores(x_1:float, x_2:float, a:float, b:float) -> float:
    return x_1*a + x_2*b**2

def reg_1(x:float, a:float, b:float) -> float:
    return a*x**3 + b*x**2

def reg_2(x:float, c:float) -> float:
    return c*x

def reg_1_2(x:float, a:float, b:float, c:float, d:float) -> float:
    return a*x**3 + b*x**2 + c*x + d

def regressao_2_betas(x1:float, x2:float, b1:float, b2:float) -> float:
    return b1*x1 + b2*x2

def regressao_4_betas(x1:float, x2:float, b1:float, b2:float, b3:float, b4:float) -> float:
    return b1*x1 + b2*x2 + b3*x1**2 + b4*x2**2

def relu(x:float) -> float:
    if x > 0:
        return x
    return x/10

def sigmoide(x:float) -> float:
    return 1/(1 + 2.71**(-x))

def mlp(x1:float, x2:float, b_0:float, b_1:float, b_2:float, b1:float, b2:float, b3:float, b4:float, b5:float, b6:float) -> float:
    x1_1 = relu(b_0 + x1*b1 + x2*+b2)
    x2_1 = relu(b_1 + x1*b3 + x2*+b4)
    return relu(b_2 + x1_1*b5 + x2_1*b6)

# Classe de testes
class Teste(unittest.TestCase):
    def teste_classe(self):
        # Testando classe e métodos
        print("Testes de regressão:")
        print("\nTeste 1.1:")
        try:
            dado = [[0, 3], [1, 2], [1, 4], [2, 3]]
            teste_1 = Regression(regressao_simples)
            print(f"{teste_1}\n")
            teste_1.run(dado)
            print(teste_1)
        except Exception as e:
            self.fail(f"Teste 1 com regressão simples: {e}")

        print("\nTeste 1.2:")
        try:
            dado = [[x, regressao_simples(x, a = 15, b = 1)] for x in range(10)]
            teste_1.run(dado)
            print(teste_1)
        except Exception as e:
            self.fail(f"Teste 2 com regressão simples: {e}")

        print("\nTeste 1.2:")
        try:
            dado = [[x, regressao_2(x, a = 15, b = -7, c = -4)] for x in range(10)]
            teste_2 = Regression(regressao_2)
            teste_2.iterations = 100
            teste_2.run(dado, precision = 0.0001)
            print(teste_2)
        except Exception as e:
            self.fail(f"Teste com regressão de 3 parâmetros: {e}")

        print("\nTeste 1.3:")
        try:
            dado = [[x, regressao_nao_continua(x, a = 7)] for x in range(20)]
            teste_3 = Regression(regressao_nao_continua)
            teste_3.run(dado)
            print(teste_3)
        except Exception as e:
            self.fail(f"Teste com regressão não continua: {e}")

        print("\nTeste 1.4:")
        try:
            print(teste_3['a'])
            print(teste_2['c'])
        except Exception as e:
            self.fail(f"Teste do __getitem__: {e}")

        print(f"\n\n{'='*50}\nTestes de lock:")
        print("\nTeste 2.1:")
        try:
            dado = [[x, int(x*3 % 11), regressao_2_regressores(x, int(x*3 % 11), a = 3, b = 10)] for x in range(5)]
            teste_4 = Regression(regressao_2_regressores, regressors = ["x_1", "x_2"])
            teste_4.lock(a = 3)
            teste_4.run(dado, precision = 0.001)
            print(teste_4)
        except Exception as e:
            self.fail(f"Teste com regressão com 2 regressores e lock: {e}")

        print("\nTeste 2.2:")
        try:
            dado = [[x, int(x*3 % 11), regressao_2_regressores(x, int(x*3 % 11), a = 3, b = 10)] for x in range(5)]
            teste_5 = Regression(regressao_2_regressores, regressors = ["x_1", "x_2"])
            teste_5.run(dado, precision = 0.001)
            print(teste_5)
        except Exception as e:
            self.fail(f"Teste com regressão com 2 regressores: {e}")

        print(f"\n\n{'='*50}\nTestes de predição:")
        print("\nTeste 3.1:")
        try:
            print(dado[3])
            print(teste_5.prediction(x_1 = 3, x_2 = 9))
        except Exception as e:
            self.fail(f"Teste da predição com regressão com 2 regressores: {e}")

        print("\nTeste 3.2:")
        try:
            print([x[-1] for x in dado])
            print(teste_5.prediction([x[:-1] for x in dado]))
        except Exception as e:
            self.fail(f"Teste da predição com regressão com 2 regressores e lista: {e}")

        print(f"\n\n{'='*50}\nTestes de excessões Regression/run:")
        print("\nTeste 4.1...")
        with self.assertRaises(AssertionError):
            teste_6 = Regression(regressao_2_regressores)

        print("\nTeste 4.2...")
        with self.assertRaises(AssertionError):
            teste_6 = Regression(regressao_2_regressores, regressors = ["x_1", "x_2"])
            teste_6.run([1, 2, 3])

        print("\nTeste 4.3...")
        with self.assertRaises(AssertionError):
            teste_6.run(1)

        print("\nTeste 4.4...")
        with self.assertRaises(AssertionError):
            teste_6.run(dado, precision = [2])

        print("\nTeste 4.5...")
        with self.assertRaises(AssertionError):
            Regression(lambda x: x + 1)

        print(f"\n\n{'='*50}\nTestes de excessões lock:")
        print("\nTeste 5.1...")
        teste_7 = Regression(regressao_2_regressores, regressors = ["x_1", "x_2"])
        with self.assertRaises(AssertionError):
            teste_7.lock(x_1 = 2)

        print("\nTeste 5.2...")
        with self.assertRaises(AssertionError):
            teste_7.lock(x_1 = 2, x_2 = 1, b = 1)

        print(f"\n\n{'='*50}\nTestes de excessões prediction:")
        print("\nTeste 6.1...")
        with self.assertRaises(AssertionError):
            teste_5.prediction(dado)

        print("\nTeste 6.2...")
        with self.assertRaises(AssertionError):
            teste_5.prediction(x_1 = 3, x_2 = 7, a = 1)

        print("\nTeste 6.3...")
        with self.assertRaises(AssertionError):
            teste_5.prediction(x_1 = 3)

        print(f"\n\n{'='*50}\nTestes de excessões __getitem__:")
        print("\nTeste 7.1...")
        with self.assertRaises(AssertionError):
            teste_5[5]

        print("\nTeste 7.2...")
        with self.assertRaises(AssertionError):
            teste_5['param_nao_existente']

        print(f"\n\n{'='*50}\nTestes de métodos especiais:")
        print("\nTeste 8.1...")
        self.assertEqual(Regression(reg_1) == Regression(reg_1), True)

        print("\nTeste 8.2...")
        self.assertEqual(Regression(reg_1) == Regression(reg_2), False)

        print("\nTeste 8.3...")
        with self.assertRaises(AssertionError):
            teste_5['param_nao_existente'] = 3

        print("\nTeste 8.4...")
        t_1 = Regression(reg_1)
        self.assertEqual(len(t_1), 1)

        print("\nTeste 8.5...")
        t_2 = Regression(regressao_2_regressores, regressors = ["x_1", "x_2"])
        self.assertEqual(len(t_2), 2)

        print(f"\n\n{'='*50}\nTestes de metodos especiais para modelos de mistura:")
        print("\nTeste 9.1:")
        try:
            a1 = Regression(reg_1)
            a2 = Regression(reg_2)
            f = a1 + a2 - 20
            print(f"\nModelo mistura:\n{f}")

            dado = [[x, reg_1_2(x, a = 1, b = -3, c = -10, d = -20)] for x in range(10)]
            f_esperado = Regression(reg_1_2)
            f_esperado.run(dado, precision = 0.01)
            print(f"\nEsperado:\n{f_esperado}")

            f['b'] = -3.5
            f.change(a = 1.5, c = -10.5)
            f.run(dado, precision = 0.01)
            print(f"\nModelo:\n{f}")

            print(f"\n{f_esperado.prediction([x[:-1] for x in dado])}")
            print(f.prediction([x[:-1] for x in dado]))
        except Exception as e:
            self.fail(f"Teste com mistura: {e}")

        print(f"\n\n{'='*50}\nTestes com o banco de dados iris:")
        print("\nTeste 10.1:")
        try:
            iris = load_iris()
            iris = [list(x[:3]) for x in iris.data]
            
            modelo_iris = Regression(regressao_2_betas, regressors = ['x1', 'x2'])
            modelo_iris.run(iris)

            print(f"\n{modelo_iris}")

            modelo_iris_2 = Regression(regressao_4_betas, regressors = ['x1', 'x2'])
            modelo_iris_2.run(iris, precision = 0.1)

            print(f"\n{modelo_iris_2}")

            modelo_iris_3 = Regression(mlp, regressors = ['x1', 'x2'])
            modelo_iris_3.run(iris, precision = 0.1)

            print(f"\n{modelo_iris_3}")

            print(f"\nComparando com os valores reais:\n{'MODELO 1':12}|{'MODELO 2':12}|{'MODELO 3':12}|{'ESPERADO':12}")
            for *x, y in iris:
                print(f"{modelo_iris.prediction([x])[0]:12.04f}|{modelo_iris_2.prediction([x])[0]:12.04f}|{modelo_iris_3.prediction([x])[0]:12.04f}|{y:12.04f}")
         
        except Exception as e:
            self.fail(f"Não foi possível completar o teste com o banco de dados iris: {e}")

        
if __name__ == "__main__":
    unittest.main()
       
