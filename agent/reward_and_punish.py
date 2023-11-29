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
    def calculate_pnl(self):
        

        # Calculate profit/loss as a percentage of the previous dollars 
        if self.portfolio.position_type == "long":   
            profit_loss = self.portfolio.position_size*(self.agent.environment.current_close - self.portfolio.entry_price)/self.portfolio.entry_price
        elif self.portfolio.position_type == "short":
            profit_loss = -(self.portfolio.position_size*(self.agent.environment.current_close - self.portfolio.entry_price)/self.portfolio.entry_price)
        else:
            profit_loss = 0
        # Calculate total reward
        reward = profit_loss 
       
        return reward 
    #
    def update_pnl(self):
            fee_rate = 0.0032

            # Calcula el PnL con el precio de entrada y el precio actual
            if self.position_type == "long":
                pnl = self.position_size * (self.environment.current_close - self.entry_price) / self.entry_price
            elif self.position_type == "short":
                pnl = -(self.position_size * (self.environment.current_close - self.entry_price) / self.entry_price)
            else:
                pnl = 0
            return pnl - (self.current_dollars * fee_rate)




    def calculate_unrealized_pnl(self):
        upnl = self.calculate_pnl()
        if upnl<-5:
             return -2.5
        elif upnl<-10:
            return -5
        elif upnl<-15:
            return -7.5
        elif upnl<-20:
            return -10
        elif upnl<-25:
            return -12.5
        else:
            return 0
     


    def calculate_reward(self, pnl):
        profit = self.agent.portfolio_manager.pnl_for_reward
        
        if self.agent.portfolio_manager.in_position:
            upnl = self.calculate_unrealized_pnl()
        else:
            upnl = 0
        reward = profit + upnl*0.001
    #    print("on the reward","profit ",profit, "upnl", upnl,  "reward ",reward,  "total reward", self.agent.trainer.reward, "current dollars", self.agent.portfolio_manager.current_dollars)
        #print("reward: ", reward,"step: ", self.agent.environment.current_step)
        return reward



