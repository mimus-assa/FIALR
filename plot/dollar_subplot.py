from plot.base_model import SubplotBase
import numpy as np

class DollarSubplot(SubplotBase):
    def __init__(self, fig, position):
        super().__init__(fig, position)

        self.current_dollars = []
        self.dollars_line, = self.ax.plot([], [], color='brown', label='Current Dollars')
        self.lines.append(self.dollars_line)
        self.ax.legend()

    def update_limits(self):
        # Establecer límites del eje X
        limit = 100
        if len(self.steps) > limit:
            self.ax.set_xlim(self.steps[-limit], self.steps[-1])
        elif len(self.steps) > 1:
            self.ax.set_xlim(self.steps[0], self.steps[-1])
        else:
            x_min = self.steps[0] - 0.5 if self.steps else -0.5
            x_max = self.steps[0] + 0.5 if self.steps else 1.5
            self.ax.set_xlim(x_min, x_max)

        # Establecer límites del eje Y
        y_values = self.current_dollars[-limit:] if len(self.steps) > limit else self.current_dollars
        y_values = [y for y in y_values if not np.isnan(y)]
        if y_values:
            min_val, max_val = np.nanmin(y_values), np.nanmax(y_values)
            self.ax.set_ylim(min_val - 30, max_val + 30)  # Establecer límites a ±100 del rango de y_values
        else:
            self.ax.set_ylim(-30, 30)  # Valores predeterminados si no hay datos

        self.ax.relim()
        self.ax.autoscale_view()

    def plot(self, current_dollars, step):
        # Asegúrate de que esta línea se ejecute en el momento adecuado
       # print(f"Último debug - Plotting current dollars: {current_dollars} at step: {step}")
        self.add_step(step)
        self.current_dollars.append(current_dollars)
        self.dollars_line.set_data(self.steps, self.current_dollars)
        self.update_limits()

    def reset(self):
        super().reset()
        self.current_dollars = []