from plot.base_model import SubplotBase

import numpy as np

class FinalDollarSubplot(SubplotBase):
    '''this plot should show the final dollars per episode, instead of updating every step, it should update every episode'''
    def __init__(self, fig, position):
        super().__init__(fig, position)
        self.final_dollars = []
        self.episodes = []
        self.final_dollar_line, = self.ax.plot([], [], 'o-', color='orange', label='Final Dollar per Episode')
        self.lines.append(self.final_dollar_line)
        self.ax.legend()

    def plot(self, final_dollar, episode):
        self.final_dollars.append(final_dollar)
        self.episodes.append(episode)

        # Update the data for the line
        self.final_dollar_line.set_data(self.episodes, self.final_dollars)

        # Re-calculate the limits based on the new data
        self.ax.relim()
        self.ax.autoscale_view()

        # Update the plot limits
        self.update_limits()


    def update_limits(self):
        # Establecer un rango mínimo para el eje y para evitar escalas extrañas con pocos datos.
        y_min = np.min(self.final_dollars) if self.final_dollars else 0
        y_max = np.max(self.final_dollars) if self.final_dollars else 1
        y_range = y_max - y_min
        if y_range < 1:  # Establecer un rango mínimo de 1 si los datos están muy cerca entre sí.
            y_avg = (y_max + y_min) / 2
            y_min = y_avg - 0.5
            y_max = y_avg + 0.5
        self.ax.set_ylim(y_min, y_max)

        if len(self.episodes) > 1:
            self.ax.set_xlim(0, len(self.episodes) + 1)
        else:
            self.ax.set_xlim(-0.5, 1.5)




    def reset_subplots(self):

        pass


