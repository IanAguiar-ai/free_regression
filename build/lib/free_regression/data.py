"""
Dados para testes
"""

def to_dummy(data:list) -> list:
    """
    Função que pega uma lista de listas e passa as colunas categoricas para numéricas binárias

    Args:
        data: Lista de listas com os dados do usuário
    """
    assert type(data) == list, "<data> has to be a list"
    assert type(data[0]) == list, "<data[n]> has to be a list"
    assert (k := list(map(len, data))) and max(k) == min(k), "The <data> list must be the same size in all itens"

    def transpose(data:list) -> list:
        """
        Faz a transposição dos dados
        """
        new_data:list = []
        for i in range(len(data[0])):
            new_data.append([data[n][i] for n in range(len(data))])            
        return new_data

    # Transpões os dados
    data = transpose(data)

    # Obtem variáveis e index da lista transposta aonde serão criados variáveis dummy
    create = []
    i = 0
    for var in data:
        if str in list(map(type, var)):
            create.append([i, sorted(list(set(var)))])
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
        return f"NOME DO BANCO DE DADOS: {self.name}\nCOLUNAS: {', '.join(self.columns)}"

class DataRegression(Representation):
    """
    Dados exemplos para teste
    """
    def __init__(self):
        self.name:str = ""
        self.columns:list = ['idade', 'altura']
        self.data:dict = {"nome_coluna_1":[],
                          "nome_coluna_2":[],
                          "nome_coluna_3":[]}

if __name__ == "__main__":
    teste_1 = DataRegression()
    print(teste_1)
    #print(teste_1['algo'])

    dados = [[1, "a", 5, "1"],
             [3, "b", 6, "2"],
             [0, "a", 4, "1"],
             [6, "c", 3, "2"],
             [4, "a", 4, "2"],]

    print(to_dummy(dados))
    
