

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime

class PlotAgent:
    def __init__(self, env, portfolio_manager, agent):
        self.env = env 
        self.portfolio_manager = portfolio_manager  # store the portfolio manager
        self.agent = agent  
        self.steps = []
        self.high_prices = []  
        self.low_prices = []
        self.stop_losses = []  
        self.take_profits = []  
        self.current_dollars = []  
        self.signal_o_values = []  # initialize signal_o_values
        self.signal_c_values = []  # initialize signal_c_values
        self.signal_o_markers = []  # initialize signal_o_markers
        self.signal_c_markers = []  # initialize signal_c_markers
        self.signal_o_plots = []  # initialize signal_o_plots
        self.signal_c_plots = []  # initialize signal_c_plots
        self.max_current_dollars_per_episode = []

        # Create four subplots, with the text-based one being half the width of the others
        self.fig = plt.figure()
        self.gs = gridspec.GridSpec(5, 2, height_ratios=[1, 1, 1, 1, 0.5])  # Añade una nueva fila
        self.ax1 = plt.subplot(self.gs[0, 0])  # prices and signals
        self.ax2 = plt.subplot(self.gs[1, 0])  # current dollars
        self.ax4 = plt.subplot(self.gs[2, 0])  # losses
        self.ax3 = plt.subplot(self.gs[0:, 1])  # texts and values
        self.ax5 = plt.subplot(self.gs[3, 0])  # Nueva ax para los máximos dólares actuales
        self.max_dollars_line, = self.ax5.plot([], [], color='magenta', label='Max Current Dollars')  # Nueva línea
        self.ax5.legend()  # Leyenda para la nueva subplot

        self.high_price_line, = self.ax1.plot(self.steps, self.high_prices, color='blue', label='High Price')
        self.low_price_line, = self.ax1.plot(self.steps, self.low_prices, color='purple', label='Low Price')

        self.stop_loss_line, = self.ax1.plot(self.steps, self.stop_losses, color='red', label='Stop Loss')
        self.take_profit_line, = self.ax1.plot(self.steps, self.take_profits, color='green', label='Take Profit')

        self.ax1.legend()  # add legend to first subplot

        self.dollars_line, = self.ax2.plot(self.steps, self.current_dollars, color='brown', label='Current Dollars')  # Moved to ax2
        self.ax2.legend()  # add legend to second subplot
        # Create the lines for plotting losses
        self.losses_0_line, = self.ax4.plot([], [], color='red', label='Loss 0')  # New line plot for losses_0
        self.losses_1_line, = self.ax4.plot([], [], color='green', label='Loss 1')  # New line plot for losses_1
        self.losses_2_line, = self.ax4.plot([], [], color='blue', label='Loss 2')  # New line plot for losses_2

        self.ax4.legend()  # add legend to fourth subplot

        # Initialize episode data for losses plot
        self.episodes = []
        self.losses_0 = []
        self.losses_1 = []
        self.losses_2 = []
        self.signal_o_steps = []
        self.signal_c_steps = []

    def update_max_current_dollars(self, max_dollars):
        self.max_current_dollars_per_episode.append(max_dollars)

    def add_data(self):
        self.steps.append(self.env.current_step)
        self.high_prices.append(self.env.current_high)
        self.low_prices.append(self.env.current_low)
        self.current_dollars.append(self.portfolio_manager.current_dollars)

        if self.portfolio_manager.signal_c == 'closed':
            self.stop_losses.append(self.stop_losses[-1])
            self.take_profits.append(self.take_profits[-1])
        else:
            self.stop_losses.append(self.portfolio_manager.stop_loss_price)
            self.take_profits.append(self.portfolio_manager.take_profit_price)

        close = self.env.current_close
        signal_o_value = np.nan if self.portfolio_manager.signal_o == 0 else close
        signal_c_value = np.nan if self.portfolio_manager.signal_c == 0 else close

        signal_o_marker = 'ro' if self.portfolio_manager.signal_o == 'long' else 'go' if self.portfolio_manager.signal_o == 'short' else None
        signal_c_marker = 'bo' if self.portfolio_manager.signal_c == 'closed' else None

        if signal_o_marker is not None:
            self.signal_o_steps.append(self.env.current_step)
            self.signal_o_values.append(signal_o_value)
            self.signal_o_markers.append(signal_o_marker)

        if signal_c_marker is not None:
            self.signal_c_steps.append(self.env.current_step)
            self.signal_c_values.append(signal_c_value)
            self.signal_c_markers.append(signal_c_marker)

        if len(self.steps) > 30:
            self.ax1.set_xlim(self.steps[-30], self.steps[-1])
            self.ax2.set_xlim(self.steps[-30], self.steps[-1])

            y_values1 = [v for v in (self.high_prices[-30:] + self.low_prices[-30:] + self.stop_losses[-30:] + self.take_profits[-30:]) if v is not None]
            if y_values1:
                min_val, max_val = min(y_values1), max(y_values1)
                if min_val == max_val:
                    min_val -= 0.05 * abs(max_val)  # Adjust min_val to 5% below max_val
                    max_val += 0.05 * abs(max_val)  # Adjust max_val to 5% above itself
                self.ax1.set_ylim(min_val, max_val)

            y_values2 = [v for v in self.current_dollars[-30:] if v is not None]
            if y_values2:
                min_val, max_val = min(y_values2), max(y_values2)
                if min_val == max_val:
                    min_val -= 0.05 * abs(max_val)  # Adjust min_val to 5% below max_val
                    max_val += 0.05 * abs(max_val)  # Adjust max_val to 5% above itself
                self.ax2.set_ylim(min_val, max_val)

        elif len(self.steps) > 1:
            self.ax1.set_xlim(self.steps[0], self.steps[-1])
            self.ax2.set_xlim(self.steps[0], self.steps[-1])

            y_values1 = [v for v in (self.high_prices + self.low_prices + self.stop_losses + self.take_profits) if v is not None]
            if y_values1:
                min_val, max_val = min(y_values1), max(y_values1)
                if min_val == max_val:
                    min_val -= 0.05 * abs(max_val)  # Adjust min_val to 5% below max_val
                    max_val += 0.05 * abs(max_val)  # Adjust max_val to 5% above itself
                self.ax1.set_ylim(min_val, max_val)

            y_values2 = [v for v in self.current_dollars if v is not None]
            if y_values2:
                min_val, max_val = min(y_values2), max(y_values2)
                if min_val == max_val:
                    min_val -= 0.05 * abs(max_val)  # Adjust min_val to 5% below max_val
                    max_val += 0.05 * abs(max_val)  # Adjust max_val to 5% above itself
                self.ax2.set_ylim(min_val, max_val)

        if len(self.episodes) > 100:
            self.ax4.set_xlim(self.episodes[-30], self.episodes[-1])
        elif len(self.episodes) > 1:
            self.ax4.set_xlim(self.episodes[0], self.episodes[-1])

    
    def update(self, frame):
        self.high_price_line.set_data(self.steps, self.high_prices)
        self.low_price_line.set_data(self.steps, self.low_prices)
        self.stop_loss_line.set_data(self.steps, self.stop_losses)
        self.take_profit_line.set_data(self.steps, self.take_profits)
        self.dollars_line.set_data(self.steps, self.current_dollars)
        self.losses_0_line.set_data(self.episodes, self.losses_0)
        self.losses_1_line.set_data(self.episodes, self.losses_1)
        self.losses_2_line.set_data(self.episodes, self.losses_2)
        

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax4.relim()
        self.ax4.autoscale_view()
        self.max_dollars_line.set_data(self.episodes, self.max_current_dollars_per_episode)  # Actualiza la nueva línea
        self.ax5.relim()
        self.ax5.autoscale_view()    

        # Remove previous scatter plots
        for coll in self.ax1.collections:
            coll.remove()

        # Create new scatter plots
        signal_o_steps = [step for step, marker in zip(self.steps, self.signal_o_markers) if marker]
        signal_o_values = [value for value, marker in zip(self.signal_o_values, self.signal_o_markers) if marker]
        signal_o_colors = ['r' if marker == 'ro' else 'g' for marker in self.signal_o_markers if marker]

        signal_c_steps = [step for step, marker in zip(self.steps, self.signal_c_markers) if marker]
        signal_c_values = [value for value, marker in zip(self.signal_c_values, self.signal_c_markers) if marker]
        signal_c_colors = ['b' for marker in self.signal_c_markers if marker]

        # Use the separate steps lists for scatter plots
        if signal_o_steps and signal_o_values and len(signal_o_steps) == len(signal_o_colors):
            self.ax1.scatter(self.signal_o_steps, signal_o_values, c=signal_o_colors, marker='o')

        if signal_c_steps and signal_c_values and len(signal_c_steps) == len(signal_c_colors):
            self.ax1.scatter(self.signal_c_steps, signal_c_values, c=signal_c_colors, marker='o')

        self.ax4.legend()


        return self.high_price_line, self.low_price_line, self.stop_loss_line, self.take_profit_line, self.dollars_line, self.max_dollars_line  # Añade la nueva línea a la lista de objetos devueltos
    
    def update_text(self):
        # Clear the subplot
        self.ax3.clear()

        # Add each piece of data as a new line of text
        self.ax3.text(0, 1.0, f"Episode: {self.agent.episode}", transform=self.ax3.transAxes)
        self.ax3.text(0, 0.95, f"Step: {self.env.current_step}", transform=self.ax3.transAxes)
        self.ax3.text(0, 0.9, f"Current Dollars: {round(self.portfolio_manager.current_dollars,2)}", transform=self.ax3.transAxes)
        self.ax3.text(0, 0.85, f"Max Current Dollars: {round(self.portfolio_manager.max_current_dollars,2)}", transform=self.ax3.transAxes)
        self.ax3.text(0, 0.75, f"Stop Loss Price: {'None' if self.portfolio_manager.stop_loss_price is None else round(self.portfolio_manager.stop_loss_price,2)}", transform=self.ax3.transAxes)
        self.ax3.text(0, 0.7, f"Take Profit Price: {'None' if self.portfolio_manager.take_profit_price is None else round(self.portfolio_manager.take_profit_price,2)}", transform=self.ax3.transAxes)
        self.ax3.text(0, 0.65, f"Positioquen Type: {self.portfolio_manager.position_type}", transform=self.ax3.transAxes)
        self.ax3.text(0, 0.6, f"Bonuses: {round(self.agent.reward_and_punishment.bonuses,4)}", transform=self.ax3.transAxes)
        self.ax3.text(0, 0.55, f"Penalty: {round(self.agent.reward_and_punishment.penalty,4)}", transform=self.ax3.transAxes)
        self.ax3.text(0, 0.5, f"Reward: {round(self.agent.trainer.reward,4)}", transform=self.ax3.transAxes)
        self.ax3.text(0, 0.45, f"Win Streak: {self.agent.reward_and_punishment.win_streak}", transform=self.ax3.transAxes)
        self.ax3.text(0, 0.4, f"Lose Streak: {self.agent.reward_and_punishment.lose_streak}", transform=self.ax3.transAxes)

        # Remove the x and y axes
        self.ax3.axis('off')       

    def _calculate_average_loss(self, loss_list):
        return np.mean(loss_list) if len(loss_list) > 0 else None

    def update_losses(self, episode):
        # Update episode data for losses plot
        self.episodes.append(episode)
        
        loss_0 = self._calculate_average_loss(self.agent.trainer.losses_0[-1]) if isinstance(self.agent.trainer.losses_1[-1], list) else np.mean(self.agent.trainer.losses_0) 
        loss_1 = self._calculate_average_loss(self.agent.trainer.losses_1[-1]) if isinstance(self.agent.trainer.losses_1[-1], list) else np.mean(self.agent.trainer.losses_1)
        loss_2 = self._calculate_average_loss(self.agent.trainer.losses_2[-1]) if isinstance(self.agent.trainer.losses_2[-1], list) else np.mean(self.agent.trainer.losses_2)

        self.losses_0.append(loss_0)
        self.losses_1.append(loss_1)
        self.losses_2.append(loss_2)
        self.max_current_dollars_per_episode.append(self.portfolio_manager.max_current_dollars)

    def init_plot(self):
        self.fig.show()

    def plot_data(self):
        
        self.update(len(self.steps) - 1)
        plt.pause(0.01)

    def reset(self):
        # Save the figure to a file before resetting
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.fig.savefig(f"/home/mimus/qlearning/plot/imgs/plot_{timestamp}.jpg")

        # Then reset the data
        self.steps = []
        self.high_prices = []
        self.low_prices = []
        self.current_dollars = []  # reset current_dollars
        self.stop_losses = []
        self.take_profits = []
        self.signal_o_values = []
        self.signal_c_values = []
        self.signal_o_markers = []  # reset signal_o_markers
        self.signal_c_markers = []  # reset signal_c_markers

        # And reset the plots
        self.high_price_line.set_data(self.steps, self.high_prices)
        self.low_price_line.set_data(self.steps, self.low_prices)
        self.stop_loss_line.set_data(self.steps, self.stop_losses)
        self.take_profit_line.set_data(self.steps, self.take_profits)
        self.dollars_line.set_data(self.steps, self.current_dollars)

        # Clear the scatter plots
        for plot in self.signal_o_plots:
            plot.remove()
        for plot in self.signal_c_plots:
            plot.remove()

        # Clear the lists
        self.signal_o_plots = []
        self.signal_c_plots = []

        self.signal_o_steps = []
        self.signal_c_steps = []

    

    def reset_all(self):
        # Reset the losses plot
        self.losses_0_line.set_data([], [])
        self.losses_1_line.set_data([], [])
        self.losses_2_line.set_data([], [])

        # Clear the episode data for losses plot
        self.episodes = []
        self.losses_0 = []
        self.losses_1 = []
        self.losses_2 = []

    def plotter(self):
            self.add_data()
            self.update_text()
            self.update(len(self.steps) - 1)  # add this line
            self.plot_data()


