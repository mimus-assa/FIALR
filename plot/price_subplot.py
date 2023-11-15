import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import pandas as pd
from plot.base_model import SubplotBase

class PriceSubplot(SubplotBase):
    def __init__(self, fig, position):
        super().__init__(fig, position)

        self.open_prices = []
        self.high_prices = []
        self.low_prices = []
        self.close_prices = []
        self.stop_losses = []
        self.take_profits = []
        self.signal_o_values = []
        self.signal_c_values = []
        self.signal_o_markers = []
        self.signal_c_markers = []
        self.signal_o_steps = []
        self.signal_c_steps = []

        self.entry_prices = []
        self.close_trades = []
        self.entry_ts=[]
        self.exit_ts=[]
        self.tp_trades = []
        self.sl_trades = []
        self.last_was_open=None
        self.counter = 0

        self.high_price_line, = self.ax.plot([], [], color='blue', label='High Price')
        self.low_price_line, = self.ax.plot([], [], color='purple', label='Low Price')
        self.stop_loss_line, = self.ax.plot([], [], color='red', label='Stop Loss')
        self.take_profit_line, = self.ax.plot([], [], color='green', label='Take Profit')

        self.lines.extend([self.high_price_line, self.low_price_line, self.stop_loss_line, self.take_profit_line])
        self.ax.legend()

    def update_limits(self):
        # Verifica si hay suficientes pasos para comenzar a desplazar la gráfica
        if len(self.steps) > 1:  # Asegúrate de que hay al menos dos pasos para establecer límites
            self.ax.set_xlim(self.steps[0], self.steps[-1])
        if len(self.steps) > 30:
            self.ax.set_xlim(self.steps[-30], self.steps[-1])
        else:
            self.ax.set_xlim(self.steps[0], self.steps[-1] if self.steps else 0)

        # Ignora valores NaN usando np.nanmin y np.nanmax
        y_values = (self.high_prices + self.low_prices + 
                    self.stop_losses + self.take_profits)
        y_values = [y for y in y_values if not np.isnan(y)]  # Filtra los NaN

        if y_values:  # Verifica si y_values no está vacío
            min_val, max_val = np.nanmin(y_values), np.nanmax(y_values)
            if min_val == max_val:
                min_val -= 0.05 * abs(max_val)
                max_val += 0.05 * abs(max_val)
            self.ax.set_ylim(min_val, max_val)

        self.ax.relim()
        self.ax.autoscale_view()



    def plot(self, current_ts, Open, High, Low, Close, stop_loss, take_profit, signal_o, signal_c, step):
        self.add_step(step)
        self.high_prices.append(High)
        self.low_prices.append(Low)

        # Establece np.nan para stop_loss y take_profit si no hay valores
        self.stop_losses.append(stop_loss if stop_loss is not None else np.nan)
        self.take_profits.append(take_profit if take_profit is not None else np.nan)

        # Actualiza las listas de señales solo si hay una señal
        if signal_o is not None:
            self.signal_o_values.append(signal_o)
            self.signal_o_steps.append(step)
            self.signal_o_markers.append('long' if signal_o > 0 else 'short')
        if signal_c is not None:
            self.signal_c_values.append(signal_c)
            self.signal_c_steps.append(step)
            self.signal_c_markers.append('closed')

        # Actualiza las líneas de precio, stop loss y take profit
        self.high_price_line.set_data(self.steps, self.high_prices)
        self.low_price_line.set_data(self.steps, self.low_prices)
        self.stop_loss_line.set_data(self.steps, self.stop_losses)
        self.take_profit_line.set_data(self.steps, self.take_profits)

        # Primero, elimina los scatter plots antiguos.
        self.remove_scatter_plots()

        # Luego, añade los nuevos scatter plots si hay señales
        if self.signal_o_values:
            signal_o_colors = ['r' if marker == 'long' else 'g' for marker in self.signal_o_markers]
            self.ax.scatter(self.signal_o_steps, self.signal_o_values, c=signal_o_colors, marker='o')
        if self.signal_c_values:
            signal_c_colors = ['b' for _ in self.signal_c_markers]
            self.ax.scatter(self.signal_c_steps, self.signal_c_values, c=signal_c_colors, marker='o')

        # Actualiza los límites de los ejes para el nuevo plot
        self.update_limits()
        
        
        if signal_o is not None:
            self.entry_prices.append(Close)
            self.entry_ts.append(current_ts)
            self.tp_trades.append(take_profit)
            self.sl_trades.append(stop_loss)

        
        if signal_c is not None: 
            self.exit_ts.append(current_ts)
    

    def reset(self):
        #now we create a dataframe with the trades info this is entr_price, entry_ts, exit_ts, tp_trades, sl_trades

        if len(self.entry_ts) > len(self.exit_ts):
            excess = len(self.entry_ts) - len(self.exit_ts)
            self.entry_ts = self.entry_ts[excess:]
            self.entry_prices = self.entry_prices[excess:]
            self.tp_trades = self.tp_trades[excess:]
            self.sl_trades = self.sl_trades[excess:]

        #df_trades=pd.DataFrame({'entry_price':self.entry_prices,'entry_ts':self.entry_ts,'exit_ts':self.exit_ts,'tp_trades':self.tp_trades,'sl_trades':self.sl_trades})
        #now we save it as a csv file
        #df_trades.to_csv(f'trades/trades_{self.counter}.csv')
        self.counter = self.counter + 1
        super().reset()
        self.high_prices = []
        self.low_prices = []
        self.stop_losses = []
        self.take_profits = []
        self.signal_o_values = []
        self.signal_c_values = []
        self.signal_o_markers = []
        self.signal_c_markers = []
        self.signal_o_steps = []
        self.signal_c_steps = []
        self.signal_o_steps = []
        self.signal_c_steps = []
        self.entry_prices = []
        self.close_trades = []
        self.entry_ts=[]
        self.exit_ts=[]
        self.tp_trades = []
        self.sl_trades = []
