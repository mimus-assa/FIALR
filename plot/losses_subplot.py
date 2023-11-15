from plot.base_model import SubplotBase

import numpy as np

class LossesSubplot(SubplotBase):
    def __init__(self, fig, position):
        super().__init__(fig, position)
        self.losses_0 = []
        self.losses_1 = []
        self.losses_2 = []

        self.losses_0_line, = self.ax.plot([], [], color='red', label='Loss 0')
        self.losses_1_line, = self.ax.plot([], [], color='green', label='Loss 1')
        self.losses_2_line, = self.ax.plot([], [], color='blue', label='Loss 2')
        self.episodes = []
        self.lines.extend([self.losses_0_line, self.losses_1_line, self.losses_2_line])
        self.ax.legend()

    def update_limits(self):
        # Establecer un rango mínimo para el eje y para evitar escalas extrañas con pocos datos.
        y_min = None
        y_max = None
        for losses in [self.losses_0, self.losses_1, self.losses_2]:
            if losses:
                if y_min is None or np.min(losses) < y_min:
                    y_min = np.min(losses)
                if y_max is None or np.max(losses) > y_max:
                    y_max = np.max(losses)
        if y_min is None:
            y_min = 0
        if y_max is None:
            y_max = 1
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



    def plot(self, loss_0, loss_1, loss_2, episode):

        self.losses_0.append(np.mean(loss_0))
        self.losses_1.append(np.mean(loss_1))
        self.losses_2.append(np.mean(loss_2))
        self.episodes.append(episode)
        self.losses_0_line.set_data(self.episodes, self.losses_0)
        self.losses_1_line.set_data(self.episodes, self.losses_1)
        self.losses_2_line.set_data(self.episodes, self.losses_2)

        self.ax.relim()
        self.ax.autoscale_view()
        self.update_limits()


        
    def reset(self):
        pass









        