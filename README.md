# free_regression

Construtor de Regressão Genérico. Com esta biblioteca, o usuário é capaz de:

1. Criar regressões 'tradicionais';
2. Criar regressões personalizadas;
3. Manipular as regressões.


# Download

```
pip install git+https://github.com/IanAguiar-ai/free_regression
```

# Importação

Para importar a classe principal:

```
from free_regression import Regression
```

Para importar os geradores de funções:

```
from free_regression import generate_regression, generate_mlp, generate_mlp_classifier
```

Para importar as funções geradoras de imagens:

```
from free_regression import plot_expected, plot_residual
```

Para importar as funções de manipulação de dados:

```
from free_regression import to_dummy
```

# Documentação das classes e funções

A documentação gerada pelo Doxygen pode ser acessada [aqui](https://github.com/IanAguiar-ai/free_regression/blob/main/free_regression/documentacao/html/annotated.html).

A classe principal que o usuário deve utilizar é a **Regression**, que recebe como argumentos:

- **function**: Função de regressão que o usuário deseja utilizar;
- **regressors**: Lista de strings que especifica os regressores utilizados. Não é necessário passar este parâmetro se a função possuir apenas o parâmetro *x* como regressor, por exemplo, *def f(x, b1, b2)*;
- **loss_function**: Função de perda para encontrar o mínimo local. Por padrão, utiliza-se a função de mínimos quadrados.

Além disso, na classe principal, as variáveis às quais o usuário tem acesso são:

- **.params**: Parâmetros que serão encontrados;
- **.regressors**: Variáveis regressoras;
- **.iterations**: Número mínimo de iterações para convergência.


## Criando regressores

O usuário pode criar regressores de duas formas: utilizando as funções prontas ou desenvolvendo seus próprios regressores.

### Geração de regressores (geradores)

Se o usuário desejar utilizar um regressor próprio, ele possui algumas opções, como:

- **generate_regression**: Para regressão linear, recebe dois parâmetros: *regressors* e *degree*;
- **generate_mlp**: Para gerar um Multi Layer Perceptron, recebe dois parâmetros: *regressors* e *neurons*, utilizando ReLU para gerar a descontinuidade;
- **generate_mlp_classifier**: Para gerar um Multi Layer Perceptron classificador, recebe dois parâmetros: *regressors* e *neurons*, utilizando uma função sigmoide para gerar a descontinuidade.

Todas essas funções retornam dois elementos: a função em si e a lista de regressores.


Exemplo de uso:

```
funcao, regressores = generate_regression(regressors = 3, degree = 2)
meu_modelo_regressor = Regression(function = funcao, regressors = regressores)
meu_modelo.run(seus_dados) # Para começar a regressão
```

```
meu_modelo_regressor = Regression(*generate_regression(regressors = 3, degree = 2))
meu_modelo.run(seus_dados) # Para começar a regressão
```

### Regressores próprios

Você pode criar regressores próprios para realizar a regressão, basta definir a função. Por exemplo:

```
def minha_funcao(x:float, b1:float, b2:float) -> float:
	return x*b1 + b2
	
meu_modelo_regressor = Regression(minha_funcao)
print(meu_modelo_regressor) # Para ver as propriedades do modelo
meu_modelo_regressor.run(seus_dados) # Para começar a regressão
```

Essa versão corrige e formaliza a explicação de como criar regressores personalizados, mantendo a clareza e a simplicidade.

```
def minha_funcao(h:float, b1:float, b2:float) -> float:
	return h*b1 + b2
	
meu_modelo_regressor = Regression(minha_funcao, regressors = ["h"])
print(meu_modelo_regressor) # Para ver as propriedades do modelo
meu_modelo_regressor.run(seus_dados) # Para começar a regressão
```

## Regressão

Comandos para fazer a regressão.

### Regressão direta

Para realizar uma regressão com a biblioteca, o usuário deve seguir os seguintes passos:

1. Definir ou criar a função de regressão que será utilizada;
2. Preparar os dados que serão usados na regressão. Os dados devem ser organizados em uma lista de listas, onde a última coluna corresponde ao valor esperado e as outras colunas são reservadas para os *x_n* (variáveis regressoras);
3. Executar o comando `.run(seus_dados)` na instância da classe **Regression**.

Em código, o passo a passo seria assim:

```
# Passo 1
funcao_regressora, argumentos = generate_regression(regressors = 2, degree = 1)
meu_modelo_regressor = Regression(funcao_regressora, argumentos)

# Passo 2
meus_dados = [[3, 2, 9], [2, 3, 1], [3, 5, 4], [2, 3, 1.3], [1, 3, 2], [7, 4, 20]]

# Passo 3
meu_modelo_regressor.run(meus_dados)

# Conferindo parâmetros atualizados
print(meu_modelo_regressor)
```

### Parâmetros de regressão

Além da regressão simples, é possível realizar diversas alterações nos parâmetros a fim de controlar melhor o processo de regressão. Algumas das funções embutidas na classe *Regression* são:

1. **.change**: Recebe cada parâmetro a ser modificado e o valor correspondente a essa alteração.

2. **.change_all**: Recebe um valor do tipo float que é aplicado a todos os parâmetros válidos.

3. **.lock**: Recebe cada parâmetro a ser bloqueado e o valor em que ele deve ser fixado. Após essa operação, esses parâmetros não são mais considerados na regressão, pois estão bloqueados.

A seguir, apresentamos um exemplo da função **.change**:

```
# Criando a função
def minha_funcao(x:float, b1:float, b2:float, b3:float) -> float:
	return b1*x**2 + x*b2 + b3

# Criando a instancia da classe
meu_modelo_regressor = Regression(minha_funcao)

# Antes da modificação
print(f"Antes:\n{meu_modelo_regressor}\n")

# Mudando os betas
meu_modelo_regressor.change(b1 = 2, b2 = 1.1, b3 = 15)
print(f"Depois do change:\n{meu_modelo_regressor}\n")
```

Exemplo do **.change_all**:

```
# Criando a função
def minha_funcao(x:float, b1:float, b2:float, b3:float) -> float:
	return b1*x**2 + x*b2 + b3

# Criando a instancia da classe
meu_modelo_regressor = Regression(minha_funcao)

# Antes da modificação
print(f"Antes:\n{meu_modelo_regressor}\n")

# Mudando os betas
meu_modelo_regressor.change_all(10)
print(f"Depois do change:\n{meu_modelo_regressor}\n")
```

Exemplo do **.lock**:

```
# Criando a função
def minha_funcao(x:float, b1:float, b2:float, b3:float) -> float:
	return b1*x**2 + x*b2 + b3

# Criando a instancia da classe
meu_modelo_regressor = Regression(minha_funcao)

# Antes da modificação
print(f"Antes:\n{meu_modelo_regressor}\n")

# Mudando os betas
meu_modelo_regressor.lock(b3 = 20)
print(f"Depois do lock:\n{meu_modelo_regressor}\n")

# Fazendo a regressão
meu_modelo_regressor.run(seus_dados)
print(f"Depois do lock e do run:\n{meu_modelo_regressor}\n")
# Perceba que as variáveis que estão bloqueadas não mudam mesmo depois da regressão
```

O **.change** e o **.change_all** servem para iniciar a regressão a partir de um ponto diferente, sendo que, por padrão, todos os valores começam em 0.1. Caso o usuário fique preso em um mínimo local indesejado, ele pode modificar tanto todos os parâmetros quanto parâmetros específicos.

O **.lock** pode ser utilizado para travar, por exemplo, um intercepto.

O uso dessas técnicas requer um conhecimento um pouco mais aprofundado sobre a função de regressão utilizada.


## Predição

Após o usuário realizar a regressão, é interessante que ele possa prever valores desconhecidos. Para isso, ele deve utilizar a função **.prediction** da classe *Regression*. Um exemplo de uso pode ser visto a seguir:

```
# Criando a função
def minha_funcao(x:float, b1:float, b2:float, b3:float) -> float:
	return b1*x**2 + x*b2 + b3

# Criando a instancia da classe
meu_modelo_regressor = Regression(minha_funcao)

# Fazendo a regressão
meu_modelo_regressor.run(seus_dados)

# Predição
resposta = meu_modelo_regressor.prediction(x = 4)
print(f"Predição para x = 4: {resposta}")
```

```
# Criando a função
def minha_funcao(x1:float, x2:float, b1:float, b2:float, b3:float) -> float:
	return x1*b1 + x2*b2 + b3

# Criando a instancia da classe
meu_modelo_regressor = Regression(minha_funcao, regressors = ["x1", "x2"])

# Fazendo a regressão
meu_modelo_regressor.run(seus_dados)

# Predição
resposta = meu_modelo_regressor.prediction(x1 = 2, x2 = 1.2)
print(f"Predição para x1 = 2 e x2 = 1.2: {resposta}")
```

## Misturas

É possível fazer a mistura de funções de regressão, por exemplo:

```
# Criando as funções para a mistura

def parte_1(x, a, b):
	return a*x**3 + b*x**2
	
def parte_2(x, c):
	return c*x
	
# Fanzendo a mistura das regressões
meu_modelo_parte_1 = Regressao(parte_1)
meu_modelo_parte_2 = Regressao(parte_2)

meu_modelo_final = meu_modelo_parte_1 + meu_modelo_parte_1 - 20

print(f"Modelo mistura: {meu_modelo_final}")
```

Perceba que, no caso acima, obtemos \( ax^3 + bx^2 + cx - 20 \).

É possível criar modelos de mistura utilizando os operadores '+', '-', '*', '/', e '**'. Se o usuário desejar usar um operador diferente, basta utilizar a função **.operation** da classe *Regression*. Por exemplo:


```
# Criando as funções para a mistura

def parte_1(x, a, b):
	return a*x**3 + b*x**2
	
def parte_2(x, c):
	return c*x
	
# Fanzendo a mistura das regressões
meu_modelo_parte_1 = Regressao(parte_1)
meu_modelo_parte_2 = Regressao(parte_2)

meu_modelo_final = meu_modelo_parte_1.operation(meu_modelo_parte_1, operator = "%")

print(f"Modelo mistura: {meu_modelo_final}")
```

Atenção: observe que a mistura é feita da esquerda para a direita. Por exemplo, \( a + b / c \) é sempre \( (a + b) / c \) e não \( a + (b / c) \).

## Função de perda

A função de perda é, por padrão, uma função de mínimos quadrados, sendo essa a sua forma específica:

```
def least_squares(vector_1:list, vector_2:list) -> float:
    """
    Função de minimos quadrados
    """
    return sum([(vector_1[i] - vector_2[i])*(vector_1[i] - vector_2[i]) for i in range(len(vector_1))])
```

Contudo, se o usuário seguir o padrão de criar uma função de perda que utiliza os dois vetores passados—sendo o vetor dos valores observados e o vetor dos valores preditos nessa ordem—é possível alterar a função de perda com a função **.loss_function** da classe *Regression*. Por exemplo:

```
# Criando função de perda
def perda_modificada(v_1:list, v_2:list) -> float:
	"""
	Função de perda que da mais pesso para perdas para valores abaixo do observado
	"""
	perda = 0
	for i in range(len(v_1)):
		if v_1 > v2:
			perda += (v1[i] - v2[i])**2
		else:
			perda += (v2[i] - v1[i])/10
	return perda    	


# Definindo a função criada como a loss_function
meu_modelo.loss_function(perda_modificada)

# Fazer a regressão com a função de perda modificada
meu_modelo.run(seus_dados)
```

## Reutilização

### Salvando e reutilizando parâmetros

Assim que a regressão é realizada, é possível salvar os parâmetros. Para isso, basta utilizar a função **.save** da classe *Regression*. Por exemplo:

```
# Depois de treinar o modelo...

# Salvando
meu_modelo.save(name = "nome_do_modelo")
```

Isso salvará, na mesma pasta em que o código Python estiver sendo executado, um arquivo *.memory*, que pode ser acessado posteriormente por uma instância da classe *Regression* construída da mesma forma:

```
# Depois de reconstruir o modelo...

# Puxar os parâmetros
meu_modelo.open(name = "nome_do_modelo") # Não é preciso colocar o ".memory" no final do nome do modelo
```

### Passando parâmetros de uma instancia para a outra

É possível transferir parâmetros comuns de uma instância para outra. Por exemplo:

```
def func_1(x, a, b):
	return a*x**3 + b*x**2
	
def func_2(x, b, c):
	return c*x + b

# Fanzendo a mistura das regressões
meu_modelo_1 = Regressao(func_1)
meu_modelo_2 = Regressao(func_2)

meu_modelo_1 << meu_modelo_2
```

No caso acima, o valor do parâmetro *b* do *meu_modelo_2* é transferido para o *meu_modelo_1*.

### Setar *seed*

A função **.set_seed** da classe *Regression* recebe um número inteiro e define a *seed* do processo, garantindo reprodutibilidade. Exemplo:

```
meu_modelo = Regressao(func_1)
meu_modelo.set_seed(1)
```

## Manipulação de dados

Existem funções que auxiliam o usuário no tratamento dos dados. Por exemplo, a função **to_dummy**. Essa função recebe uma lista de listas, onde algumas colunas são categóricas, e, desde que todas as categorias estejam presentes nas variáveis categóricas, ela cria novas colunas dummy para cada categoria, em ordem alfabética, por coluna categórica. Exemplo:

```
dados = [[1, "a", 5, "1"],
         [3, "b", 6, "2"],
         [0, "a", 4, "1"],
         [6, "c", 3, "2"],
         [4, "a", 4, "2"],]

print(to_dummy(dados))
```

Retorna:

```
[[1, 5, 1, 0, 0, 1, 0],
 [3, 6, 0, 1, 0, 0, 1],
 [0, 4, 1, 0, 0, 1, 0],
 [6, 3, 0, 0, 1, 0, 1],
 [4, 4, 1, 0, 0, 0, 1]]
```

## Gráficos

### *plot_expected*

Função que recebe:

- **regression**: Instância da classe *Regression*;
- **data**: Dados em lista de listas;
- **size**: Tamanho da figura.

Esta função gera um plot simples que compara os valores observados com os valores preditos. Se os valores observados puderem ser representados como uma série, a função o fará; caso contrário, apresentará os dados observados como pontos.

### *plot_residual*

Função que recebe:

- **regression**: Instância da classe *Regression*;
- **data**: Dados em lista de listas;
- **size**: Tamanho da figura;
- **percentile**: Lista com dois percentis que serão apresentados no plot.

Esta função gera um histograma com a distribuição dos resíduos.

# Limitação da biblioteca

Dado o método de convergência, é muito fácil que a regressão fique presa em mínimos locais. Portanto, tenha atenção ao usar o gerador de MLP.

# Dicas

## Pré-regressão/treinamento

Se uma regressão não está apresentando boa precisão ou demora demais para chegar ao resultado esperado, é possível realizar uma regressão com maior precisão e, em seguida, com menor precisão. Por exemplo:

```
meu_modelo = Regression(*generate_regression(regressors = 2, degree = 1))

meu_modelo.run(seus_dados, precision = 1)
meu_modelo.run(seus_dados, precision = 0.001)
```

## Regressão/treinamento de forma dividida

O usuário ainda pode dividir os dados e treinar um modelo de cada vez. Isso faz sentido se o usuário deseja, por exemplo, modelar uma ação, aprendendo todo o histórico dela, e depois refazer a regressão com os últimos 30 dias, a fim de obter um viés mais recente da ação. Por exemplo:

```
meu_modelo = Regression(*generate_regression(regressors = 2, degree = 1))

meu_modelo.run(seus_dados[:])
meu_modelo.run(seus_dados[-30:])
```

