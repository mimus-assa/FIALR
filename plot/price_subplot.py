import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import pandas as pd
from plot.base_model import SubplotBase
import json

class PriceSubplot(SubplotBase):
    def __init__(self, fig, position):
        super().__init__(fig, position)

        self.close_prices = []
        self.signal_o_values = []
        self.signal_c_values = []
        self.signal_o_markers = []
        self.signal_c_markers = []
        self.signal_o_steps = []
        self.signal_c_steps = []


        # Añadir nuevas listas para almacenar la información de los markers
        self.open_markers_x = []
        self.open_markers_y = []
        self.open_markers_color = []
        self.close_markers_x = []
        self.close_markers_y = []
        self.close_markers_color = []


        self.entry_prices = []
        self.close_trades = []
        self.entry_ts=[]
        self.exit_ts=[]
        self.signals_o = []

        self.close_price_line, = self.ax.plot([], [], color='blue', label='Close Price')
        self.lines.extend([self.close_price_line])
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
        y_values = self.close_prices[-limit:] if len(self.steps) > limit else self.close_prices

        y_values = [y for y in y_values if not np.isnan(y)]
        if y_values:
            min_val, max_val = np.nanmin(y_values), np.nanmax(y_values)
            self.ax.set_ylim(min_val - 30, max_val + 30) 
        else:
            self.ax.set_ylim(-30, 30)  # Valores predeterminados si no hay datos

        self.ax.relim()
        self.ax.autoscale_view()

    def plot(self, current_ts, Open, High, Low, Close, signal_o, signal_c, step):
        self.add_step(step)
        self.close_prices.append(Close)

        # Actualiza las listas de señales solo si hay una señal
        if signal_o != 0:
            self.signal_o_values.append(-1 if signal_o == "short" else 1 if signal_o == "long" else None)
            self.signal_o_steps.append(step)
            self.signal_o_markers.append(signal_o)
            # Añadir posición y color del marker de apertura
            self.open_markers_x.append(step)
            self.open_markers_y.append(Close)
            color = 'green' if signal_o == 'long' else 'red' if signal_o == 'short' else 'black'
            self.open_markers_color.append(color)

        if signal_c != 0:
            self.signal_c_values.append(signal_c)
            self.signal_c_steps.append(step)
            self.signal_c_markers.append('closed')
            # Añadir posición y color del marker de cierre
            self.close_markers_x.append(step)
            self.close_markers_y.append(Close)
            self.close_markers_color.append('blue')

        # Actualiza las líneas de precio
        self.close_price_line.set_data(self.steps, self.close_prices)

        # Primero, elimina los scatter plots antiguos.
        self.remove_scatter_plots()

        # Luego, añade los nuevos scatter plots si hay señales
        if self.signal_o_values:
            signal_o_colors = ['red' if marker == 'long' else 'green' for marker in self.signal_o_markers]
            self.ax.scatter(self.signal_o_steps, self.signal_o_values, c=signal_o_colors, marker='o')

        if self.signal_c_values:
            signal_c_colors = ['blue' for _ in self.signal_c_markers]
            self.ax.scatter(self.signal_c_steps, self.signal_c_values, c=signal_c_colors, marker='o')

        # Dibujar los markers de apertura y cierre
        self.ax.scatter(self.open_markers_x, self.open_markers_y, c=self.open_markers_color, marker='o')
        self.ax.scatter(self.close_markers_x, self.close_markers_y, c=self.close_markers_color, marker='o')

        # Actualiza los límites de los ejes para el nuevo plot

        # Registra las entradas y salidas para trades
        if signal_o != 0:
            self.entry_prices.append(Close)
            self.entry_ts.append(current_ts)
            self.signals_o.append(signal_o)

        if signal_c != 0: 
           # print(signal_c)
            self.exit_ts.append(current_ts)
        self.update_limits()
# Añade la impresión de depuración aquí para verificar el paso y el valor de cierre
       #print(f"Último debug - Plotting price at step: {step}, Close: {Close}")

      
        #if signal_c != 0:
           # print(f"Debug - Adding close marker at step: {step}, Close: {Close}, Signal: {signal_c}")
            # Añadir el código para agregar el marcador de cierre


        # ... [resto del código de reseteo y actualización] ...

    

    def reset(self):
        # Calcular cuántos trades no se han cerrado
        excess = len(self.entry_ts) - len(self.exit_ts)

        if excess > 0:
            # Eliminar los trades que están abiertos (es decir, los últimos 'excess' trades)
            self.entry_ts = self.entry_ts[:-excess]
            self.entry_prices = self.entry_prices[:-excess]
            self.signals_o = self.signals_o[:-excess]

      
        # Crear DataFrame y guardarlo como antes
        #df_trades = pd.DataFrame({'entryPrice': self.entry_prices, 'entryTS': self.entry_ts, 'exitTS': self.exit_ts, 'tp': self.tp_trades, 'sl': self.sl_trades, 'type': self.signals_o})
        
         #now we save it as a csv file
        #file_name = f'trades/trades_{self.counter}.json'
        #list_of_dicts = df_trades.to_dict(orient='records')
        #data_to_save = {"positions": list_of_dicts}
        # Guarda los datos en el archivo JSON
        #with open(file_name, 'w') as file:
         #   json.dump(data_to_save, file, indent=4)
        #self.counter = self.counter + 1
        super().reset()
        self.close_prices = []
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
        self.signals_o = []
        self.open_markers_x = []
        self.open_markers_y = []
        self.open_markers_color = []
        self.close_markers_x = []
        self.close_markers_y = []
        self.close_markers_color = []

        

