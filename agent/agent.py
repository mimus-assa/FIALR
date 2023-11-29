# Import necessary libraries and modules
import numpy as np
import logging
from agent.memory import ExperienceReplay
from configs import AgentConfig
from agent.explo import ExplorationExploitation
from agent.portfolio_manager import PortfolioManager
from agent.reward_and_punish import RewardAndPunishment
from agent.trainer import AgentTrainer
from models.model_manager import ModelManager
import random

# Define the DQNAgent class
class DQNAgent:
    # Define action types
    HOLD = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3

    def __init__(self, environment, model_path=None, deep_model_config=None, config=None):
        # Initialize agent properties
        self.environment = environment
        self.model_path = model_path
        self.plot=False
        self.window_size = config.window_size
        self.action_size = environment.action_space.n
        self.number_of_features = self.environment.num_columns

        self.config = config
        self.episode=0
        self.init_models(deep_model_config)
        self.init_portfolio_manager()
        self.init_reward_and_punishment()
        self.init_exploration_exploitation()
        
        self.scores = []
        self.episode_rewards = []
        
        self.initial_dollars = self.config.initial_dollars if config else None

        # Set trainer and other initial parameters
        self.trainer = AgentTrainer(self)
        self.max_scale_dolars=100
        

    def init_models(self, deep_model_config):
        self.deep_model_config = deep_model_config
        self.model_manager = ModelManager(self.window_size, self.action_size, self.number_of_features,
                                          self.model_path, self.deep_model_config)
        self.target_model = self.model_manager.clone_model_architecture()
        self.batch_size = self.config.batch_size
        self.memory_size = self.config.memory_size
        self.batch_losses = []
        self.losses = []    
        
    def init_exploration_exploitation(self):
        self.experience_replay = ExperienceReplay(self)
        self.exploration_explotation = ExplorationExploitation(self)
        self.last_action = self.HOLD 

    def init_portfolio_manager(self):
        self.portfolio_manager = PortfolioManager(self.config, self)

    def init_reward_and_punishment(self):
        self.reward_and_punishment = RewardAndPunishment(self.portfolio_manager, self) 

    def record_and_print_score(self, episode, episodes):
        # Record and print the score, and append to scores and losses lists
        print(f"Episode: {episode + 1}/{episodes}, Ending step: {self.environment.current_step}, Score: {self.portfolio_manager.current_dollars}, Reward: {self.trainer.reward},Max current dollars: {round(self.portfolio_manager.max_current_dollars,2)} ,Average Training Loss: {self.model_manager.average_losses}")
        self.scores.append(self.portfolio_manager.current_dollars)
        self.losses.append(self.model_manager.average_losses)

    def is_bellow_threshold(self):#creo que esto deberia irse al portfolio manager
        # Check if current dollars are below the stop price
        return self.portfolio_manager.current_dollars <= self.config.stop_price * self.portfolio_manager.max_current_dollars
    
    def is_out_of_time(self):#y este deberia irse al environment
        # Check if current step is beyond the ending step
        return self.environment.current_step > self.environment.ending_step

    def is_done(self):
        # Check if episode is done
        return self.is_bellow_threshold() or self.is_out_of_time()

    def train_agent(self):
        # Train agent for a given number of episodes
        for episode in range(self.config.episodes):
            self.episode=episode
            
            self.trainer.train(episode)
            self.exploration_explotation.update_epsilon()#esto talves deberia ir al trainer
        self.model_manager.save_model()

    def reset_agent(self):
        self.portfolio_manager.current_dollars = self.portfolio_manager.original_initial_dollars
        self.portfolio_manager.max_current_dollars = self.portfolio_manager.original_initial_dollars
        self.portfolio_manager.in_position = False
        self.portfolio_manager.position_type = None
        self.portfolio_manager.entry_price = None

    
    def setup_for_training(self, episode):
        self.environment = self.environment
        self.config = self.config
        self.experience_replay = self.experience_replay
        self.target_model = self.model_manager.clone_model_architecture()
        self.model_manager = self.model_manager
        self.episode_rewards = self.episode_rewards
        #self.batch_losses = self.batch_losses#esto lo removi porque creo que no sirve para nada
        #self.exploration_explotation.counter = 0
        
        starting_step = 0
        #print("max current dollars", round(self.portfolio_manager.max_current_dollars,2))
        self.reset_agent() 
        current_state = self.update_observation(self.environment.reset(starting_step))
        return current_state


        
    def update_observation(self, state):
        # Actualiza con los valores actuales de dólares y los máximos dólares actuales
        state[-1, -5] = self.get_normalized_values(self.portfolio_manager.current_dollars)
        state[-1, -4] = self.get_normalized_values(self.portfolio_manager.max_current_dollars)
        # Incluye un indicador de si actualmente está en posición
        state[-1, -3] = int(self.portfolio_manager.in_position)
        # Normaliza y actualiza el precio de stop
        state[-1, -2] = self.get_normalized_values(self.portfolio_manager.max_current_dollars * self.config.stop_price)
        # Actualiza la acción como un valor único en lugar de un vector one-hot
        state[-1, -1] = self.last_action  # Aquí asumimos que action es un valor numérico. Si no, necesitarás convertirlo.
        
        return state

    def get_normalized_values(self, value):
        initial_dollars = self.portfolio_manager.original_initial_dollars
        max_value = initial_dollars * self.max_scale_dolars
        normalized_value = value / max_value
        if not (0 <= normalized_value <= 1):
            print(f"WARNING: value is not normalized: {normalized_value}")
        return normalized_value