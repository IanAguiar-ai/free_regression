# free_regression

Construtor de regressão genérico.

Com está biblioteca o usuário é capaz de:

1. Criar regressoras 'tradicionais';

2. Criar regressoras próprias;

3. Manipular as regressoras.

# Download

```
pip install git+https://github.com/IanAguiar-ai/free_regression
```

# Documentação das classes e funções

A documentação gerada pelo Doxygen pode ser acessada [aqui](https://github.com/IanAguiar-ai/free_regression/blob/main/free_regression/documentacao/html/annotated.html).

A classe principal que o usuário deve usar é a *Regression* que recebe como argumentos:

- **function**: Função de regressão que o usuário quer usar;

- **regressors**: Lista de strings que explicita os regressores usados, não precisa ser passada se a função tem apenas o parâmetro *x* como regressor, por exemplo, *def f(x, b1, b2)*;

- **loss_function**: Função de perda para achar o mínimo local, por padrão é uma função de mínimos quadrados.

## Criando regressores

O usuário pode criar regressores de duas formas, ou usando as funções prontas ou criando seus próprios regressores.

### Geração de regressores (geradores)

Se o usuário quiser usar um regressor próprio ele tem algumas opções como:

- **generate_regression**: Para regressão linear, recebe dois parâmetros, *regressors* e *degree*;

- **generate_mlp**: Para gerar Mult Layer Perceptron, recebe dois parâmetros, *regressors* e *neurons*, usa de uma ReLU para gerar a descontinuidade;

- **generate_mlp_classifier**: Para gerar Mult Layer Perceptron, recebe dois parâmetros, *regressors* e *neurons*.

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

Além da regressão simples, pode ser feito várias mudanças nos parâmetros afim de controlar mais a regressão. Alguns desses parâmetros são

1. change
2. change_all
3. lock

## Predição

## Misturas

## Função de perda

## Manipulação de dados

## Gráficos

## Reutilização

# Limitação da biblioteca

Dado o método de convergência, é muito fácil a regressão ficar presa em mínimos locais, então tenha atenção ao usar o gerador de MLP.
