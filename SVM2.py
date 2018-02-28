import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

class Support_Verctor_Mchine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.color = {1: 'r', -1: 'b'}

        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)


    def fit(self, data):

        self.data = data
        opt_dic = {}

        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_featue_value = max(all_data)
        self.min_featue_value = min(all_data)

        all_data = None

        step_sizes = [self.max_featue_value * 0.1, self. max_featue_value * 0.01, self. max_featue_value * 0.001]

        b_range_multiple = 5
        b_multiple = 5

        latest_optimum = self.max_featue_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])

            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_featue_value * b_range_multiple),
                                   self.max_featue_value * b_range_multiple, step * b_multiple):

                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True

                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False

                        if found_option:
                            opt_dic[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print("Optimize a step")

                else:
                    w = w - step

            norms = sorted([n for n in opt_dic])

            opt_choice = opt_dic[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]

            latest_optimum = opt_choice[0][0] + step * 2





    def prediction(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c= self.color[classification])

    def visualize(self):


        return classification




data_dict = {-1: np.array([[1, 7], [2, 8], [3, 8]]), 1: np.array([[5, 1], [6, -1], [7, 3]])}

