"""
Dados para testes
"""

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
        return f"NOME DO BANCO DE DADOS: {self.name}\nCOLUNAS: {''.join(self.columns)}"

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
    print(teste_1.name)
    print(teste_1.columns)
    #print(teste_1['algo'])
    
