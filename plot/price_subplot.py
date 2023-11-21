import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import pandas as pd
from plot.base_model import SubplotBase
import json

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
        self.signals_o = []
        self.last_was_open=None
        self.counter = 0

        self.high_price_line, = self.ax.plot([], [], color='blue', label='High Price')
        self.low_price_line, = self.ax.plot([], [], color='purple', label='Low Price')
        self.stop_loss_line, = self.ax.plot([], [], color='red', label='Stop Loss')
        self.take_profit_line, = self.ax.plot([], [], color='green', label='Take Profit')

        self.lines.extend([self.high_price_line, self.low_price_line, self.stop_loss_line, self.take_profit_line])
        self.ax.legend()

    def update_limits(self):
        # Establecer límites del eje X
        if len(self.steps) > 30:
            self.ax.set_xlim(self.steps[-30], self.steps[-1])
        elif len(self.steps) > 1:
            self.ax.set_xlim(self.steps[0], self.steps[-1])
        else:
            x_min = self.steps[0] - 0.5 if self.steps else -0.5
            x_max = self.steps[0] + 0.5 if self.steps else 1.5
            self.ax.set_xlim(x_min, x_max)

        # Establecer límites del eje Y
        y_values = (self.high_prices + self.low_prices + self.stop_losses + self.take_profits)
        y_values = [y for y in y_values if not np.isnan(y)]
        if y_values:
            min_val, max_val = np.nanmin(y_values), np.nanmax(y_values)
            y_range = max_val - min_val
            y_margin = max(y_range * 0.05, 10)  # Utiliza un margen del 5% o un valor mínimo fijo, lo que sea mayor
            self.ax.set_ylim(min_val - y_margin, max_val + y_margin)
        else:
            self.ax.set_ylim(-1, 1)

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
        if signal_o!=0:
            self.signal_o_values.append(-1 if signal_o == "short" else 1 if signal_o == "long" else None)
            self.signal_o_steps.append(step)
            self.signal_o_markers.append(signal_o)
        if signal_c!=0:
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
           # print(self.signal_o_values)
           # print(self.signal_o_steps)
            signal_o_colors = ['r' if marker == 'long' else 'g' for marker in self.signal_o_markers]
           # print(signal_o_colors)
            self.ax.scatter(self.signal_o_steps, self.signal_o_values, c=signal_o_colors, marker='o')
        if self.signal_c_values:
            signal_c_colors = ['b' for _ in self.signal_c_markers]
            self.ax.scatter(self.signal_c_steps, self.signal_c_values, c=signal_c_colors, marker='o')

        # Actualiza los límites de los ejes para el nuevo plot
        self.update_limits()
        
        
        if signal_o!=0:
            self.entry_prices.append(Close)
            self.entry_ts.append(current_ts)
            self.tp_trades.append(take_profit)
            self.sl_trades.append(stop_loss)
            self.signals_o.append(signal_o)

        
        if signal_c!=0: 
            print(signal_c)
            self.exit_ts.append(current_ts)
    

    def reset(self):
        #now we create a dataframe with the trades info this is entr_price, entry_ts, exit_ts, tp_trades, sl_trades

        if len(self.entry_ts) > len(self.exit_ts):
            excess = len(self.entry_ts) - len(self.exit_ts)
            self.entry_ts = self.entry_ts[excess:]
            self.entry_prices = self.entry_prices[excess:]
            self.tp_trades = self.tp_trades[excess:]
            self.sl_trades = self.sl_trades[excess:]
            self.signals_o = self.signals_o[excess:]
 
        print("entryPrices", len(self.entry_prices), "entryTs", len(self.entry_ts), "exitTs", len(self.exit_ts), "tpTrades", len(self.tp_trades), "slTrades", len(self.sl_trades), "signalsO", len(self.signals_o))
        print("entryPrices", self.entry_prices, "entryTs", self.entry_ts)
        print("exitTs", self.exit_ts)
        print("tpTrades", self.tp_trades) 
        print("slTrades", self.sl_trades)
        print("signalsO", self.signals_o)
        df_trades=pd.DataFrame({'entryPrice':self.entry_prices,'entryTS':self.entry_ts,'exitTS':self.exit_ts,'tp':self.tp_trades,'sl':self.sl_trades, 'type':self.signals_o})
        #now we save it as a csv file
        file_name = f'trades/trades_{self.counter}.json'
        list_of_dicts = df_trades.to_dict(orient='records')
        data_to_save = {"positions": list_of_dicts}
        # Guarda los datos en el archivo JSON
        with open(file_name, 'w') as file:
            json.dump(data_to_save, file, indent=4)
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
        self.signals_o = []
