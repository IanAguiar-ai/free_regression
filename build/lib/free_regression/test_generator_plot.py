from generator_probabilistic import Generator, plot_time_series, round_series
from random import random, seed
from math import cos

if __name__ == "__main__":
    seed(2)
    text = [cos(i/7) + cos(i/3) - cos(cos(i/17) + random() - 0.5) for i in range(3_000)]
    text = round_series(text, n = 40)
    
    print(sorted(set(text)))
    test = Generator(dependence = 2)
    test.train(list(text))
    #print(test.chain)
    #print(test.prob_chain)
    #print(test.choice(["t", "a"]))
    #resp:str = test.make_text([1, 2, 3, ], lenth = 300)
    resp = plot_time_series(generator = test,
                            sequence = text[400:405],
                            lenth = 30,
                            times = 1000)

    
