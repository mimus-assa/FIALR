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
        self.upnl = 0

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
        # Obtener el PnL no realizado de PortfolioManager
        upnl = self.portfolio.get_pnl()

        # Definir los rangos y las penalizaciones correspondientes
        pnl_ranges = [(-25, -12.5), (-20, -10), (-15, -7.5), (-10, -5), (-5, -2.5)]
        
        # Iterar a través de los rangos para determinar la penalización adecuada
        for threshold, penalty in pnl_ranges:
            if upnl < threshold:
                return penalty

        # Devolver 0 si no se cumple ninguna condición
        return 0

     


    def calculate_reward(self):
        reward = 0
        # Utilizar los valores almacenados de pnl y fee del PortfolioManager
        if self.agent.last_action == self.agent.CLOSE:
            profit = self.portfolio.last_pnl
            fee = self.portfolio.last_fee
        else:
            profit = 0
            fee = 0
        
        # Calcula la recompensa total
        reward = profit - fee

        # Actualiza el PnL no realizado solo si estamos en posición
        self.upnl = self.calculate_unrealized_pnl() if self.portfolio.in_position else 0

        

        # Incluir el PnL no realizado con un factor en la recompensa
        reward += self.upnl * 0.001

        # Imprimir información de depuración si es necesario
        #print("step", self.agent.environment.current_step, "reward", reward, "profit", profit, "upnl", self.upnl, "fee", fee)
        return reward



