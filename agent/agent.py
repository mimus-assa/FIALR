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
    LONG = 1
    SHORT = 2
    HOLD = 0

    def __init__(self, environment, model_path=None, deep_model_config=None, config=None):
        # Initialize agent properties
        self.environment = environment
        self.model_path = model_path
        self.plot=False
        self.window_size = config.window_size
        self.action_size = environment.action_space[0].n
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
        self.stop_loss_levels = self.portfolio_manager.stop_loss_levels if self.portfolio_manager else None
        self.ratio_levels = self.portfolio_manager.ratio_levels if self.portfolio_manager else None
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
        self.last_action = (self.HOLD ,self.portfolio_manager.stop_loss_levels[0],self.portfolio_manager.ratio_levels[0])

    def init_portfolio_manager(self):
        self.portfolio_manager = PortfolioManager(self.config, self)

    def init_reward_and_punishment(self):
        self.reward_and_punishment = RewardAndPunishment(self.portfolio_manager, self) 

    def record_and_print_score(self, episode, episodes):
        # Record and print the score, and append to scores and losses lists
        print(f"Episode: {episode + 1}/{episodes}, Ending step: {self.environment.current_step}, Score: {self.portfolio_manager.current_dollars}, Reward: {self.trainer.reward}, Average Training Loss: {self.model_manager.average_losses}")
        self.scores.append(self.portfolio_manager.current_dollars)
        self.losses.append(self.model_manager.average_losses)

    def is_bellow_threshold(self):
        # Check if current dollars are below the stop price
        return self.portfolio_manager.current_dollars <= self.config.stop_price * self.portfolio_manager.max_current_dollars
    
    def is_out_of_time(self):
        # Check if current step is beyond the ending step
        return self.environment.current_step > self.environment.ending_step

    def is_done(self):
        # Check if episode is done
        return self.is_bellow_threshold() or self.is_out_of_time()

    def train_agent(self):
        # Train agent for a given number of episodes
        for episode in range(self.config.episodes):
            self.episode=episode
            
            self.trainer.train(episode, plot=self.plot)
            self.exploration_explotation.update_epsilon()
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
        self.batch_losses = self.batch_losses
        self.exploration_explotation.current_step = 0#revisar esto
        self.exploration_explotation.counter = 0
        
        starting_step = 0
        print("Last Max on step: ", self.portfolio_manager.step_on_max_current_dollars,"max current dollars", round(self.portfolio_manager.max_current_dollars,2), "RESTART ON: ", starting_step)
        self.reset_agent() 
        current_state = self.update_observation(self.environment.reset(starting_step))
        return current_state


    
    def update_observation(self, state):
        state[-1, -11] = self.get_normalized_values(self.portfolio_manager.current_dollars)
        state[-1, -10] = self.get_normalized_values(self.portfolio_manager.max_current_dollars)
        state[-1, -9] = int(self.portfolio_manager.in_position)
        state[-1, -8] = self.get_normalized_values(self.portfolio_manager.max_current_dollars * self.config.stop_price)
        state[-1, -7] = self.portfolio_manager.ratio/100
    

        state[-1, -6] = self.portfolio_manager.stop_loss_levels[int(self.portfolio_manager.stop_loss)]/100
 
        
        state[-1, -5] = self.get_normalized_values(self.reward_and_punishment.bonuses)
        state[-1, -4] = self.get_normalized_values(self.reward_and_punishment.penalty)
        action, _, _ = self.last_action
        state[-1, -3:] = np.eye(3)[action]
        return state

    def get_normalized_values(self, value):
        initial_dollars = self.portfolio_manager.original_initial_dollars
        max_value = initial_dollars * self.max_scale_dolars
        normalized_value = value / max_value
        if not (0 <= normalized_value <= 1):
            print(f"WARNING: value is not normalized: {normalized_value}")
        return normalized_value