# free_regression

Construtor de regressão genérico.

Com está biblioteca o usuário é capaz de:

1. Criar regressões 'tradicionais';

2. Criar regressões próprias;

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

A classe principal que o usuário deve usar é a *Regression* que recebe como argumentos:

- **function**: Função de regressão que o usuário quer usar;

- **regressors**: Lista de strings que explicita os regressores usados, não precisa ser passada se a função tem apenas o parâmetro *x* como regressor, por exemplo, *def f(x, b1, b2)*;

- **loss_function**: Função de perda para achar o mínimo local, por padrão é uma função de mínimos quadrados.

Ainda na classe principal, as variáveis que o usuário tem acesso são:

- **.params**: Parametros que serão achados;

- **.regressors**: Variáveis regressoras;

- **.iterations**: Número de iterações mínimas para convergência.

## Criando regressores

O usuário pode criar regressores de duas formas, ou usando as funções prontas ou criando seus próprios regressores.

### Geração de regressores (geradores)

Se o usuário quiser usar um regressor próprio ele tem algumas opções como:

- **generate_regression**: Para regressão linear, recebe dois parâmetros, *regressors* e *degree*;

- **generate_mlp**: Para gerar Multi Layer Perceptron, recebe dois parâmetros, *regressors* e *neurons*, usa de uma ReLU para gerar a descontinuidade;

- **generate_mlp_classifier**: Para gerar Multi Layer Perceptron, recebe dois parâmetros, *regressors* e *neurons*, usa uma sigmoide para gerar a descontinuidade;

Todas essas funções retornam dois parâmetros, a própria função e a lista de regressores.

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

Você pode criar regressores próprios para fazer a regressão, basta definir a função, por exemplo:

```
def minha_funcao(x:float, b1:float, b2:float) -> float:
	return x*b1 + b2
	
meu_modelo_regressor = Regression(minha_funcao)
print(meu_modelo_regressor) # Para ver as propriedades do modelo
meu_modelo_regressor.run(seus_dados) # Para começar a regressão
```

Nesse exemplo acima ele entende que a variável regressora é *x*, mas se não for *x* você deve explicitar ela, por exemplo:

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

Para fazer uma regressão com a biblioteca, o usuário precisa seguir os seguintes passos:

1. Ter decidido ou criado a função de regressão que será usada;

2. Ter os dados que serão usados para a regressão, os dados devem ser passados em uma lista de listas onde a ultima coluna é o valor esperado e os outros valores é reservado para os x_n (variáveis regressoras).

3. Dar o comando *.run(seus_dados)* na instancia da classe *Regression*.

Em código, o passo a passo se traduz em:

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

Além da regressão simples, pode ser feito várias mudanças nos parâmetros afim de controlar mais a regressão. Algumas dessas funções imbutidas na classe *Regression* são:

1. **.change**: Recebe cada parâmetro que deve ser mudado e o valor a ser mudado;

2. **.change_all**: Recebe um valor float que é colocado em todos os parâmetros válidos;

3. **.lock**: Recebe cada parâmetro que deve ser travado e o valor em que ele deve ser travado, depois disso, esses parâmetros não são mais trabalhados na regressão pois eles estão bloqueados.

Um exemplo do **.change**:

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

Um exemplo do **.change_all**:

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

Um exemplo do **.lock**:

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

O **.change** e o **.change_all** servem para iniciar a regressão de outro lugar, por padrão todos os valores começam em 0.1. O usuário ficando travado em um minimo local indesejado pode mudar tanto todos os parâmetros quanto parâmetros específicos.

O **.lock** pode ser usado para travar um intercepto por exemplo.

O uso dessas técnicas requer um conhecimento um pouco maior da função de regressão usada.

## Predição

Depois do usuário fazer a regressão, é interessante que ele possa prever valores desconhecidos, para isso ele deve usar a função **.prediction** da classe *Regression*. Um exemplo do uso pode ser visto logo abaixo:

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

Perceba que no caso acima ficamos com $ax^3 + bx^2 + cx - 20$.

É possível fazer modelos de mistura com os operadores '+', '-', '*', '/', '**'. Se o usuário quiser usar um operador diferente basta usar a função **.operation** da classe *Regression*, por exemplo:

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

Atenção, fique atento que a mistura é feita da esquerda para a direita, por exemplo, $a + b / c$ é sempre $(a + b)/c$ e não $a + (b/c)$.

## Função de perda

A função de perda é por padrão uma função de mínimos quadrados, em específico essa:

```
def least_squares(vector_1:list, vector_2:list) -> float:
    """
    Função de minimos quadrados
    """
    return sum([(vector_1[i] - vector_2[i])*(vector_1[i] - vector_2[i]) for i in range(len(vector_1))])
```

Mas se o usuário seguir o padrão de criar uma função de perda que usa os dois vetores passados, sendo o vetor dos valores observados e preditos nessa ordem, é possível mudar a função de perda com a função **.loss_function** da classe *Regression*, por exemplo:

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

Assim que a regressão é feita, é possível salvar os parâmetros, basta usar a função **.save** da classe *Regression*, por exemplo:

```
# Depois de treinar o modelo...

# Salvando
meu_modelo.save(name = "nome_do_modelo")
```

Isso vai salvar na mesma pasta em que o código python estiver sendo usado, um arquivo *.memory* que pode ser pego posteriormente por uma instancia da classe *Regression* construida da mesma forma:

```
# Depois de reconstruir o modelo...

# Puxar os parâmetros
meu_modelo.open(name = "nome_do_modelo") # Não é preciso colocar o ".memory" no final do nome do modelo
```

### Passando parâmetros de uma instancia para a outra

É possível passar parâmetros em comum de uma instancia para a outra, por exemplo:

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

Nesse caso acima, o valor do parâmetro *b* do *meu_modelo_2* é passado para o *meu_modelo_1*.

### Setar *seed*

A função **.set_seed** da classe *Regression* recebe um número inteiro e seta a *seed* do processo, garantindo reprodutibilidade. Exemplo:

```
meu_modelo = Regressao(func_1)
meu_modelo.set_seed(1)
```

## Manipulação de dados

Existe funções, que auxiliam o usuário a tratar os dados, por exemplo, a função **to_dummy**, essa função recebe uma lista de listas onde algumas colunas são categóricas, e desde que, nas variáveis categóricas apareçam todas as categorias, ele cria colunas dummys novas, para cada categoria em ordem alfabetica por coluna categórica. Exemplo:

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

- regression: Instancia da classe *Regression*;
	
- data: Dados em lista de lista;
	
- size: Tamanho da figura.
	
Plot simples que compara os valores observado com os valores preditos. Se os valores observados puderem ser representados como uma série a função o fará, caso contrário apresentará os dados observados com pontos.

### *plot_residual*

Função que recebe:

- regression: Instancia da classe *Regression*;
	
- data: Dados em lista de lista;
	
- size: Tamanho da figura;
	
- percentile: Lista com dois percentis que serão apresentados no plot. 

Faz um histograma com a distribuição do resíduo.

# Limitação da biblioteca

Dado o método de convergência, é muito fácil a regressão ficar presa em mínimos locais, então tenha atenção ao usar o gerador de MLP.

# Dicas

...

## Pré-regressão/treinamento

...

## Regressão/treinamento de forma dividida

...
