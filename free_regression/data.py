"""
Dados para testes
"""

class DataRegression:
    """
    Dados exemplos para teste
    """
    def __init__(self):
        self.name:str = ""
        self.columns:list = []
        self.data:dict = {"nome_coluna_1":[],
                          "nome_coluna_2":[],
                          "nome_coluna_3":[]}

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

if __name__ == "__main__":
    teste_1 = DataRegression()
    print(teste_1)
    print(teste_1.name)
    print(teste_1.columns)
    #print(teste_1['algo'])
    
