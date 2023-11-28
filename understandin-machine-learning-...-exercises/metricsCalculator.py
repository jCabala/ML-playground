import numpy as np

class MetricsCalculator:
    # Gets set of labels that correspond to points that vere clssified as 1
    def claculateEmpiricalTrainingError(self, y, y_pred):
        correctNum = 0
        for i in range(len(y)):
            correctNum += (y[i] != y_pred[i])
        return correctNum / len(y)
            