import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime

from plot.dollar_subplot import DollarSubplot 
from plot.text_values_subplot import TextValuesSubplot
from plot.losses_subplot import LossesSubplot
from plot.price_subplot import PriceSubplot
from plot.base_model import SubplotBase
from plot.final_dollars_subplot import FinalDollarSubplot


class EpsilonSubplot(SubplotBase):
    def __init__(self, fig, position, agent):
        super().__init__(fig, position)
        self.epsilon_values = []
        self.epsilon_line, = self.ax.plot([], [], color='orange', label='Epsilon')
        self.lines.append(self.epsilon_line)
        self.ax.legend()
        self.agent=agent

    def plot(self, step):
        # Suponiendo que tienes un método para obtener el valor actual de epsilon en este punto
        current_epsilon = self.agent.exploration_explotation.epsilon
        self.add_step(step)
        self.epsilon_values.append(current_epsilon)
        
        self.epsilon_line.set_data(self.steps, self.epsilon_values)
        self.update_limits()

    def update_limits(self):
        # Establecer un rango fijo para el eje y ya que epsilon decay es conocido
        self.ax.set_ylim(0, max(self.epsilon_values))
        
        # Asegúrate de que el eje x muestre todos los pasos
        self.ax.set_xlim(0, self.steps[-1] if self.steps else 0)
        self.ax.relim()
        self.ax.autoscale_view()

    def reset(self):
        super().reset()
        self.epsilon_values = []

class PlotManager:
    def __init__(self, env, portfolio_manager, agent):
        # Create the figure and the subplots using GridSpec
        self.fig = plt.figure()
        self.gs = gridspec.GridSpec(3, 3, height_ratios=[0.5, 0.5, 0.5])

        # Initialize the subplots
        self.price_subplot = PriceSubplot(self.fig, self.gs[0, 0])
        self.dollar_subplot = DollarSubplot(self.fig, self.gs[1, 0])
        self.final_dollar_subplot = FinalDollarSubplot(self.fig, self.gs[2, 0])
        self.text_subplot = TextValuesSubplot(self.fig, self.gs[0:, 1], agent, env, portfolio_manager)
        self.losses_subplot = LossesSubplot(self.fig, self.gs[0, 2])
        self.epsilon_subplot = EpsilonSubplot(self.fig, self.gs[1, 2], agent)
    def plot_epsilon(self, step):
        self.epsilon_subplot.plot(step)

    def reset_subplots(self):
        self.price_subplot.reset()
        self.dollar_subplot.reset()
        self.losses_subplot.reset()
        self.epsilon_subplot.reset()
        self.final_dollar_subplot.reset()

    def save_plot(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.fig.savefig(f"/home/mimus/qlearning/plot/imgs/plot_{timestamp}.jpg")  # Update this path as necessary

class PlotUpdater:
    def __init__(self, plot_manager, env, portfolio_manager, agent):
        self.plot_manager = plot_manager
        self.env = env
        self.portfolio_manager = portfolio_manager
        self.agent = agent

    def add_data(self):
        step = self.env.current_step
        high = self.env.current_high
        low = self.env.current_low
        open = self.env.current_open
        close = self.env.current_close
        episode = self.agent.episode
        current_dollars = self.portfolio_manager.current_dollars
        current_ts = self.env.current_ts
        stop_loss = self.portfolio_manager.stop_loss_price
        take_profit = self.portfolio_manager.take_profit_price
        signal_o = self.portfolio_manager.signal_o  # Initialize signal_o with None
        signal_c = self.portfolio_manager.signal_c  # Initialize signal_c with None

    
        self.plot_manager.price_subplot.plot(current_ts, open, high, low, close, stop_loss, take_profit, signal_o, signal_c, step)
        self.plot_manager.dollar_subplot.plot(current_dollars, step)
        self.plot_manager.text_subplot.plot()  # This will just update the text values
        self.plot_manager.plot_epsilon(step)
        end_of_episode = self.env.current_step == self.env.max_steps - 1
        if end_of_episode:
            final_dollar = self.portfolio_manager.current_dollars
            episode_number = self.agent.episode
            losses_0, losses_1, losses_2 = self.agent.trainer.losses_0, self.agent.trainer.losses_1, self.agent.trainer.losses_2
            self.plot_manager.final_dollar_subplot.plot(final_dollar, episode_number)
            self.plot_manager.losses_subplot.plot(losses_0, losses_1, losses_2, episode)
            

    def update_subplot(self, subplot):
        subplot.ax.draw_artist(subplot.ax.patch)
        for line in subplot.lines:
            subplot.ax.draw_artist(line)

    def update(self):
        # Redraw the subplots
        self.update_subplot(self.plot_manager.price_subplot)
        self.update_subplot(self.plot_manager.dollar_subplot)
        self.update_subplot(self.plot_manager.losses_subplot)

        # Update text values (no lines to draw)
        self.plot_manager.text_subplot.plot()

        # Draw the new final dollar subplot
        self.update_subplot(self.plot_manager.final_dollar_subplot)

        # Recalculate the limits and rescale the view
        self.plot_manager.price_subplot.update_limits()
        self.plot_manager.dollar_subplot.update_limits()
        self.plot_manager.losses_subplot.update_limits()

        # Blit the changes to the canvas
        self.plot_manager.fig.canvas.draw()
        self.plot_manager.fig.canvas.flush_events()

    def plot_data(self):#esto se deberia de llamar update_plot_data o algo asi  
        self.add_data()
        self.update()

class PlotAgent:
    def __init__(self, env, portfolio_manager, agent, update_frequency=1):
        self.plot_manager = PlotManager(env, portfolio_manager, agent)
        self.plot_updater = PlotUpdater(self.plot_manager, env, portfolio_manager, agent)
        self.update_frequency = update_frequency

    def plotter(self, step):
        if step % self.update_frequency == 0:
            self.plot_updater.plot_data()

    def reset(self):
        self.plot_manager.reset_subplots()

    def save_plot(self):
        self.plot_manager.save_plot()

    def init_plot(self):
        self.plot_manager.fig.show()