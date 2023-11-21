# Define the RewardAndPunishment class
import numpy as np

class RewardAndPunishment:
    # Initialize the class with a portfolio manager and an agent
    def __init__(self, portfolio_manager, agent):
        self.agent = agent
        self.steps_since_last_closed_position = 0
        self.portfolio = portfolio_manager
        self.this_win = False
        self.last_win = False
        self.penalty = 0
        self.bonuses=0
        self.steps_after_closing = 0
        self.wait_bonus = 0
        self.last_level = 0
        self.consecutive_wins = 0  # Initialize consecutive_wins
        self.steps_since_last_deficit_check = 0  # new field
        self.last_deficit_level = 1  # 100% of max_current_dollars
        self.thresholds_crossed = {0.97: False, 0.96: False, 0.95: False}
        self.win_streak = 0
        self.lose_streak = 0
        self.win_streak_bonus = 0
        self.lose_streak_penalty = 0
        self.death_penalty = 0
        self.patience_bonus = 0  # Initialize patience bonus
        self.steps_since_last_trade = 0  


    # Method to calculate reward
    def calculate_reward(self, previous_dollars):


        # Calculate profit/loss as a percentage of the previous dollars
        profit_loss = self.portfolio.current_dollars - previous_dollars
    

        # Calculate total reward
        reward = profit_loss 

        return reward

