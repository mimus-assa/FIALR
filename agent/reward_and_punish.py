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

    # Method to evaluate take profit
    def _evaluate_take_profit(self):
        # This function now also checks if a profitable position was just closed
        if self.portfolio.just_closed_profitable_position:
            return True
        elif self.portfolio.in_position:
            _, take_profit_reached = self.portfolio._evaluate_position()
            return take_profit_reached
        return False

   # Method to evaluate stop loss
    def _evaluate_stop_loss(self):
        if self.portfolio.in_position:
            stop_loss_reached, _ = self.portfolio._evaluate_position()
            return stop_loss_reached
        return False


    # Method to get current dollars
    def _get_current_dollars(self):
        return self.portfolio.current_dollars

    # Method to calculate immediate reentry penalty
    def immediate_reentry_penalty(self, steps_limit, penalty_ratio):
        if self.portfolio.just_closed_position:  # The agent just closed a position
            self.steps_since_last_closed_position = 0  # Reset counter
            self.portfolio.just_closed_position = False  # Reset flag
        if self.steps_since_last_closed_position < steps_limit and self.portfolio.in_position:
            # The agent took a new position within steps_limit of closing the last one
            penalty = self.portfolio.current_dollars * penalty_ratio
            self.steps_since_last_closed_position = steps_limit  
            return penalty
        self.steps_since_last_closed_position += 1  # Increment counter in all other cases
        return 0

    # Method to update penalty
    def update_penalty(self):
        penalties = [self.immediate_reentry_penalty(steps, ratio) for steps, ratio in [(12*3, 0.00001), (3, 0.000001), (4, 0.000005), (5, 0.00001)]]
        penalties.append(self.lose_streak_penalty)  # Add the lose streak penalty to the total penalty
        self.penalty = sum(penalties)

    def calculate_patience_bonus(self):
        if self.portfolio.in_position:
            self.patience_bonus = 0  # Reset patience bonus when in position
            self.steps_since_last_trade = 0  # Reset steps since last trade when in position
        else:
            self.steps_since_last_trade += 1  # Increment steps since last trade when not in position
            self.patience_bonus = self.steps_since_last_trade * 0.00001 * self.portfolio.current_dollars  # Calculate patience bonus (replace 0.01 with your preferred factor)
            
    # Method to calculate reward
    def calculate_reward(self, previous_dollars):
        self.calculate_death_penalty()
        self.calculate_patience_bonus()
        self.update_penalty()

        # Calculate profit/loss as a percentage of the previous dollars
        profit_loss = self.portfolio.current_dollars - previous_dollars
        
        # Convert bonuses and penalties to absolute values
        win_streak_bonus = self.win_streak_bonus
        patience_bonus = self.patience_bonus
        penalty = self.penalty
        death_penalty = self.death_penalty

        # Calculate total reward
        reward = profit_loss + win_streak_bonus + patience_bonus - penalty + death_penalty

        return reward


    def update_streaks(self):
        if self._evaluate_take_profit():  # If the last trade was profitable
            self.win_streak += 1  # Increase the win streak count
            self.lose_streak = 0  # Reset the lose streak count
            self.win_streak_bonus = self.win_streak * 0.01 * self.portfolio.current_dollars  # Calculate the win streak bonus (replace 0.01 with whatever factor you want)
            self.lose_streak_penalty = 0  # Reset the lose streak penalty
        elif self._evaluate_stop_loss():  # If the last trade was unprofitable
            self.lose_streak += 1  # Increase the lose streak count
            self.win_streak = 0  # Reset the win streak count
            self.lose_streak_penalty = self.lose_streak * 0.01 * self.portfolio.current_dollars  # Calculate the lose streak penalty (replace 0.01 with whatever factor you want)
            self.win_streak_bonus = 0  # Reset the win streak bonus

    def calculate_death_penalty(self):
        if self.agent.is_bellow_threshold():
            self.death_penalty = -0.5 * self.portfolio.current_dollars  # Change this to your desired penalty formula
        else:
            self.death_penalty = 0  # Reset death penalty when is_below_threshold is False
