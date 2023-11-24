import sys
import math
import numpy as np

import randomValuesGenerator as rvg
import metricsCalculator as mc

import matplotlib
from matplotlib import patches, patches
import matplotlib.pyplot as plt
matplotlib.use('Agg')

NUM_OF_POINTS = 20
LOWER_BOUND = -200
UPPER_BBOUND = 200
MIN_INT = -sys.maxsize + 1
MAX_INT = sys.maxsize - 1

class MinRectClasifier:
    def fit(self, points, labels):
        self._lbx = MAX_INT # Left bottom
        self._lby = MAX_INT
        self._rtx = MIN_INT # Right top
        self._rty = MIN_INT

        for i in range(NUM_OF_POINTS):
            if labels[i]:
                [x, y] = points[i]
                self._lbx = min(self._lbx, x)
                self._lby = min(self._lby, y)
                self._rtx = max(self._rtx, x)
                self._rty = max(self._rty, y)

    def fit_and_draw(self, points, labels):
        self.fit(points, labels)
        # Drawing a plot
        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots()

        # Adding the rectangle
        width = self._rtx - self._lbx
        height = self._rty - self._lby

        rect = patches.Rectangle(
            (self._lbx, self._lby), width, height, linewidth=1, fill=False)
        ax.add_patch(rect)

        # Adding the points
        xs = points[:, 0]
        ys = points[:, 1]

        true_xs, true_ys, false_xs, false_ys = [], [], [], []

        for i in range(len(xs)):
            if labels[i]:
                true_xs.append(xs[i])
                true_ys.append(ys[i])
            else:
                false_xs.append(xs[i])
                false_ys.append(ys[i])

        ax.scatter(true_xs, true_ys, color="red")
        ax.scatter(false_xs, false_ys, color="green")

        # Adding the legend and labels
        true_patch = patches.Patch(color='red', label='Values labled as 1')
        false_patch = patches.Patch(color='green', label='Values labled as 0')
        plt.legend(handles=[true_patch, false_patch])
        
        plt.ylabel('Y AXIS')
        plt.xlabel('X AXIS')
        
        plt.savefig("./plots/minRectClassifier.png")

    def predict(self, points):
        labels = []
        for [x, y] in points:
            if self.__pred_true(x, y):
                labels.append(1)
            else:
                labels.append(0)
        return np.array(labels)


    def __pred_true(self, x, y):
        return ((x <= self._rtx or math.isclose(x, self._rtx)) 
                and (x >= self._lbx or math.isclose(x, self._lbx)) 
                and (y <= self._rty or math.isclose(y, self._rty)) 
                and (y >= self._lby or math.isclose(y, self._lby))) 


if __name__ == '__main__':
    generator = rvg.RandomValuesGenerator()

    points = generator.generateUniform(
        NUM_OF_POINTS, LOWER_BOUND, UPPER_BBOUND, 2)
    labels = generator.generateBinary(NUM_OF_POINTS)
    
    mrc = MinRectClasifier()
    mrc.fit_and_draw(points, labels)
    pred_labels = mrc.predict(points)

    calculator = mc.MetricsCalculator()
    ETE = calculator.claculateEmpiricalTrainingError(labels, pred_labels)

    print(f"ETE: {ETE}")