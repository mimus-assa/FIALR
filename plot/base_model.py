import matplotlib.pyplot as plt

class SubplotBase:
    def __init__(self, fig, position):
        self.fig = fig
        self.position = position
        self.ax = plt.subplot(position)
        self.lines = []
        self.steps = []

    def update_limits(self):
        raise NotImplementedError("Must be implemented by the subclass.")

    def plot(self):
        raise NotImplementedError("Must be implemented by the subclass.")

    def reset(self):
        for line in self.lines:
            line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.lines = []  # Clear the list of lines
        self.steps = []

    def add_step(self, step):
        self.steps.append(step)

    def remove_scatter_plots(self):
        for coll in self.ax.collections:
            coll.remove()