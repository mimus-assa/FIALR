# Import necessary libraries and modules
import numpy as np
import gym
from configs import BtcMarketEnvConfig
from gym.spaces import Tuple, Discrete, Box
import json
from tqdm import tqdm
import random

# Define the BtcMarketEnv class which is a subclass of gym.Env
class BtcMarketEnv(gym.Env):
    # Initialize the class with data, prices, and a config
    def __init__(self, data,  prices, config=None):
        #print("Inicializando BtcMarketEnv...")
        # Set config to BtcMarketEnvConfig if not provided
        self.config = config #if config is not None else BtcMarketEnvConfig()
        # Initialize environment
        self._initialize_environment(prices, self.config)
        # Set other properties
        self.data=data
        self.extra_features = 5
       # print("Configurando num_columns...")
        self.num_columns = data.shape[2] + self.extra_features  # Cambiado de data.shape[3]
       # print("num_columns configurado:", self.num_columns)
        self.window_size = config.window_size
        # Define observation space
       # print("Configurando observation_space...")
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.window_size, self.num_columns), dtype=np.float32)
       # print("observation_space configurado.")
       # print("Configurando data_buffer...")
        self.data_buffer = np.zeros((len(self.data), self.window_size, self.num_columns))
     #   print("data_buffer configurado.")

        #print("Llenando data_buffer...")
        self._fill_data_buffer()
        #print("data_buffer llenado.")
        # Reset environment
        #print("Reseteando el entorno...")
        self.reset(self.starting_step)
        #print("Entorno reseteado.")
       # print("BtcMarketEnv inicializado.")

    def _fill_data_buffer(self):
      #  print("Iniciando _fill_data_buffer...")
        
        # Copiar los datos existentes en las primeras 9 columnas de cada muestra
        self.data_buffer[:, :, :self.data.shape[2]] = self.data
        
        # Añadir características adicionales si es necesario
        # Por ejemplo, si tienes alguna lógica para calcular estas características adicionales, 
        # puedes añadirlas aquí. Si no, y solo necesitas inicializarlas, puedes hacerlo de la siguiente manera:
        # self.data_buffer[:, :, self.data.shape[2]:] = valor_inicial
        
        #print("Finalizando _fill_data_buffer.")


    # Method to initialize environment
    def _initialize_environment(self,  prices,  config):
       # print("Iniciando _fill_data_buffer...")
        # Set various properties
        self.o, self.h, self.l, self.c = prices[:4]
        self.ts = prices[4]
        self.max_steps= config.max_steps
        self.starting_step = self.config.starting_step
        self.ending_step= self.starting_step + self.max_steps
        # Define action space
        self.action_space = Discrete(4)  # Actions: Buy, sell, hold, close
        self.current_step = self.current_ts = self.observation = None
       # print("Finalizando _fill_data_buffer.")
    # Method to reset environment
    def reset(self,start):
        #print("Reseteando el entorno...")
        # Set various properties
        self.current_open = self.o[start]
        self.current_high = self.h[start]
        self.current_low = self.l[start]
        self.current_close = self.c[start]
        self.current_step = start 
        self.current_ts = self.ts[start]
        # Roll the data buffer
        self.data_buffer = np.roll(self.data_buffer, shift=-self.config.max_steps, axis=0)
        # Get next observation
        self.observation = self._get_next_observation() 
      #  print("shape of the observation on reset from env", self.observation.shape)
        #print("Entorno reseteado.")
        return self.observation

    # Method to get next observation
    def _get_next_observation(self):
       # print("Obteniendo siguiente observación...")
        start = self.current_step % self.config.max_steps
        obs = self.data_buffer[start:start+1, :self.window_size, :]  # Obtener una sola secuencia de observación
       # print("Siguiente observación obtenida.")
        return obs

    # Method to update step features
    def update_step_features(self, step, features):
        self.data_buffer[step, :, -len(features):] = features


    # Method to update prices and timestamps
    def update_prices_and_ts(self, o, h, l, c, ts):
        self.current_open = o
        self.current_high = h
        self.current_low = l
        self.current_close = c
        self.current_ts = ts

    # Method to perform a step in the environment
    def step(self):
       # print("Ejecutando un paso en el entorno...")
        # Update prices and timestamp
        self.update_prices_and_ts(
            self.o[self.current_step],
            self.h[self.current_step],
            self.l[self.current_step],
            self.c[self.current_step],
            self.ts[self.current_step]
        )
        # Increase current step
        if self.current_step < len(self.data) - 1:    # Check if current step is less than the last index
            self.current_step += 1
        if self.current_step >= len(self.data):
            # Roll the data buffer if the end of the data is reached
            self.data_buffer = np.roll(self.data_buffer, shift=-self.config.max_steps, axis=0)
        # Get next observation
        self.observation = self._get_next_observation()
       # print("Paso en el entorno ejecutado.")
        return self.observation
