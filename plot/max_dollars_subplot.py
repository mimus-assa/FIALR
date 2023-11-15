from plot.base_model import SubplotBase


class MaxDollarsSubplot(SubplotBase):
    def __init__(self, fig, position):
        super().__init__(fig, position)
        self.max_current_dollars = []
        self.max_dollars_line, = self.ax.plot([], [], color='magenta', label='Max Current Dollars')
        self.ax.legend()

    def update_limits(self):
        # Ajusta límites si hay suficientes datos
        if len(self.steps) > 30:
            self.ax.set_xlim(self.steps[-30], self.steps[-1])
        elif len(self.steps) > 1:
            self.ax.set_xlim(self.steps[0], self.steps[-1])

        # Ajusta límites de y
        if self.max_current_dollars:
            self.ax.set_ylim(min(self.max_current_dollars), max(self.max_current_dollars))

    def plot(self, max_dollars, step):
        self.add_step(step)
        self.max_current_dollars.append(max_dollars)
        self.max_dollars_line.set_data(self.steps, self.max_current_dollars)  # Esto actualiza la línea con los nuevos datos
        self.update_limits()  # Actualiza los límites del eje después de agregar los nuevos datos

    def reset(self):
        super().reset()
        self.max_current_dollars = []