"""
Dados para testes
"""

def read_csv(csv:str) -> list:
    """
    Lê o csv sem precisar usar uma biblioteca
    """
    path = f"banco_de_dados/{csv.replace('.csv', '')}.csv"
    with open(path, "r") as arq:
        dados = arq.read()

    dados:list = dados.split("\n")[:-1]
    for line in range(len(dados)):
        dados[line] = dados[line].split(",")

    return dados

def transpose(data:list) -> list:
    """
    Faz a transposição dos dados
    """
    new_data:list = []
    for i in range(len(data[0])):
        try:
            new_data.append([data[n][i] for n in range(len(data))])
        except IndexError:
            pass

    return new_data

def to_float(number:str) -> float:
    try:
        return float(number)
    except:
        return number

def to_dummy(data:list) -> list:
    """
    Função que pega uma lista de listas e passa as colunas categoricas para numéricas binárias

    Args:
        data: Lista de listas com os dados do usuário
    """
    assert type(data) == list, "<data> has to be a list"
    assert type(data[0]) == list, "<data[n]> has to be a list"
    assert (k := list(map(len, data))) and max(k) == min(k), "The <data> list must be the same size in all itens"

    # Transpões os dados
    data = transpose(data)

    # Obtem variáveis e index da lista transposta aonde serão criados variáveis dummy
    create = []
    i = 0
    for var in data:
        if str in list(map(type, var)):
            create.append([i, sorted(list(set(var)))[:-1]])
        i += 1

    # Cria as variáveis dummy
    list_create:dict = {}
    for index, vars_ in create:
        for variable in vars_:
            list_create[variable] = []
            for value in data[index]:
                if variable == value:
                    list_create[variable].append(1)
                else:
                    list_create[variable].append(0)

    # Apaga as colunas antigas erradas
    erase = [data[index] for index, _ in create]

    for to_erase in erase:
        data.remove(to_erase)

    # Adicionando no final da lista as variáveis dummy
    for key in list_create.keys():
        data.append(list_create[key])

    return transpose(data)


class Representation:
    """
    Classe de quem será herdada os métodos especiais
    """
    def __getitem__(self, index:str) -> list:
        """
        Pega uma lista da coluna
        """
        assert index in self.data.keys(), f"Not exist colum '{index}' in data"
        return self.data[index]

    def __repr__(self) -> str:
        """
        Ajuda na vizualização para o usuário
        """
        return f"NAME DATA BASE: {self.name}\nCOLUMNS: {', '.join(self.columns)}"

class MedidasDeMassa(Representation):
    """
    Dados exemplos para teste
    """
    __socket__ = ("name", "columns", "data", "data_list")
    def __init__(self):
        # Obtendo e tratando os dados
        data:list = transpose(read_csv("medidas_de_massa"))
        columns_use:tuple = (1, 7, 11, 12)
        data:list = [list(map(to_float, data[i])) for i in columns_use]

        # Criando nome e colunas
        self.name:str = "Medidas de Massa"
        self.columns:list = [line[0] for line in data]

        # Dados
        self.data:dict = {}
        for i in range(len(data)):
            self.data[self.columns[i]] = data[i][1:]

        self.data_list:list = transpose([value for value in self.data.values()])

class ProdutividadeTrabalhoRemoto(Representation):
    """
    Dados exemplos para teste
    """
    __socket__ = ("name", "columns", "data", "data_list")
    def __init__(self):
        # Obtendo e tratando os dados
        data:list = transpose(read_csv("medidas_de_massa"))
        columns_use:tuple = (1, 7, 11, 12)
        data:list = [list(map(to_float, data[i])) for i in columns_use]

        # Criando nome e colunas
        self.name:str = "Medidas de Massa"
        self.columns:list = [line[0] for line in data]

        # Dados
        self.data:dict = {}
        for i in range(len(data)):
            self.data[self.columns[i]] = data[i][1:]

        self.data_list:list = transpose([value for value in self.data.values()])

if __name__ == "__main__":
    meus_dados = MedidasDeMassa()
##    print(teste_1)
##    #print(teste_1['algo'])
##
    dados = [[1, "a", 5, "1"],
             [3, "b", 6, "2"],
             [0, "a", 4, "1"],
             [6, "c", 3, "2"],
             [4, "a", 4, "2"],]

##
##    print(to_dummy(dados))
    
