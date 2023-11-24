import numpy as np

class RandomValuesGenerator:
    def generateUniform(self, number, lower, upper, dim):
        return np.random.uniform(lower, upper, size=(number,dim))
    
    def generateBinary(self, number):
        return np.random.randint(2, size=number)
