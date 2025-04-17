import numpy as np
import json

def Uniform(min, max, step_size):
    sample = np.random.uniform(min, max, step_size)
    with open("prior_uniform.json", "w") as f:
        json.dump(sample.tolist(), f)
    with open("prior_uniform.json") as g:
        print(g.read())
    return sample

def Normal(min, max, sigma, size):
    range = np.linspace(min,max,20)
    mean = np.sum(range)/len(range)
    sample = np.random.normal(loc=mean, scale=sigma, size=size)
    with open("prior_normal.json", "w") as f:
        json.dump(sample.tolist(), f)
    with open("prior_normal.json") as g:
        print(g.read())
        
    return sample