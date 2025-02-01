from copy import deepcopy
from random import random

def round_series(series:list, n:int = 10) -> list:
    return [int(x_i * n)/n for x_i in series]

def plot_time_series(generator:"Generator", sequence:list, lenth:int = 1, times:int = 100, size:tuple = (14, 8), bins:int = 20) -> list:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    
    if set(map(type, sequence)) != {str}:
        sequence:list = list(map(str, sequence))
        
    new_sequences:list = []
    for _ in range(times):
        new_sequences.append(sequence.copy())
        for __ in range(lenth):
            new_sequences[-1].append(generator.choice(list(map(str, new_sequences[-1][-generator.dependence:]))))
            
    values_y_to_histogram:list = [X_i[-1] for X_i in new_sequences]

    # Criando a figura e os eixos
    fig = plt.figure(figsize=size)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # Grid com duas colunas, 3x maior para a série temporal

    ax_time_series = plt.subplot(gs[0])  # Eixo principal para a série temporal
    ax_histogram = plt.subplot(gs[1], sharey=ax_time_series)  # Eixo para o histograma compartilhando Y

    # Plot da série temporal
    x = [i + 1 for i in range(len(sequence))]
    ax_time_series.plot(x, list(map(float, sequence)), label = "Dados Observados", color = "red", linestyle = "--")

    x_new = [i + 1 for i in range(len(sequence) + lenth)]
    alpha = 1 / len(new_sequences) ** (1 / 2)
    for Y_i in new_sequences:
        ax_time_series.plot(x_new, list(map(float, Y_i)), color = "blue", linestyle = "-", alpha = alpha, zorder = 1)

    ax_time_series.grid(True, which = "both", linestyle = "--", linewidth = 0.7)
    ax_time_series.set_title(f"Dados Observados com Predições por MCMC de dependência {generator.dependence}", fontsize = 16, weight = "bold")
    ax_time_series.set_xlabel("X", fontsize = 14)
    ax_time_series.set_ylabel("Y", fontsize = 14)
    ax_time_series.legend()

    counts, bin_edges, patches = ax_histogram.hist(
        values_y_to_histogram, bins = bins, orientation = "horizontal",
        color = "lightblue", alpha = 0.6, edgecolor="black"
    )

    # Converter para porcentagem
    counts_percentage = counts/counts.sum() * 100

    # Atualizar os rótulos no eixo X
    ax_histogram.clear()  # Limpa o eixo para atualizar corretamente
    ax_histogram.barh(bin_edges[:-1], counts_percentage, height = np.diff(bin_edges),
                      color = "lightblue", alpha = 0.6, edgecolor = "black")

    ax_histogram.set_xlabel("Frequência (%)", fontsize=12)
    ax_histogram.grid(axis = "x", linestyle = "--", linewidth = 0.7, alpha = 0.5)

    plt.subplots_adjust(left = 0.07, right = 0.98, top = 0.95, bottom = 0.07, wspace = 0.1)  # Ajuste de espaçamento
    plt.show()

    return new_sequences

class Generator:
    def __init__(self, dependence:int = 1) -> None:
        self.dependence:int = dependence
        self.chain:dict = None
        self.prob_chain:dict = None

    def train(self, input_:list) -> None:
        """
        Calculates the number of occurrences
        """
        self.chain:dict = {}
        temporary_chain:dict = {}
        
        for iten in range(len(input_) - self.dependence):
            temporary_input:list = input_[iten : iten + self.dependence]
            answer:str = input_[iten + self.dependence]

            current_level = self.chain
    
            for key in list(map(str, temporary_input)):
                if key not in list(map(str, current_level)):
                    current_level[key] = {}
                current_level:dict = current_level[key]

            if str(answer) not in list(map(str,current_level)):
                current_level[answer] = 0
            current_level[answer] += 1

        self.calculate_probability()
            
    def calculate_probability(self) -> None:
        """
        Transforms the number of occurrences into probabilities
        """
        self.prob_chain = {}

        def calculate_level_probabilities(current_level:dict) -> dict:
            level_probabilities = {}
            total_occurrences:float = sum(current_level.values())

            for key, count in current_level.items():
                level_probabilities[key] = count / total_occurrences

            return level_probabilities

        def traverse_chain(current_level:dict) -> dict:
            prob_level = {}
            for key, sub_level in current_level.items():
                if isinstance(sub_level, dict):
                    prob_level[key] = traverse_chain(sub_level)
                else:
                    return calculate_level_probabilities(current_level)
            return prob_level

        self.prob_chain:float = traverse_chain(self.chain)

    def probability(self, list_keys:list) -> dict:
        """
        Probability of event
        """
        final_choices:dict = self.prob_chain[list_keys[0]]
        list_keys:list = list_keys[1:]

        for key in list_keys:
            assert key in final_choices, f"'{key}' has no dependencies!"
            final_choices:dict = final_choices[key]
                
        return final_choices
        

    def choice(self, list_keys:list) -> str:
        """
        Choice a next value
        """
        assert len(list_keys) == self.dependence, f"List must be the same size as the number of dependencies\nCurrent list size: {len(list_keys)}\nDependency size: {self.dependence}"

        final_choices = self.probability(list_keys)
                

        value_random:float = random()
        probs:tuple = tuple(final_choices.values())
        answer:tuple = tuple(final_choices.keys())

        i:int = 0
        final_answer:str = answer[i]
        while value_random > probs[i]:
            value_random -= probs[i]
            i += 1
            final_answer:str = answer[i]

        return final_answer

    def make_text(self, initial_text:list, lenth:int = 50) -> str:
        """
        Make a text with the probability chain
        """
        final_text = initial_text.copy()
        while len(final_text) < lenth:
            final_text.append(self.choice(final_text[-self.dependence:]))
            
        return "".join(final_text)
