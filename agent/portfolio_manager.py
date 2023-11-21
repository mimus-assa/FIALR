import numpy as np

class PortfolioManager:
    def __init__(self, config, agent):
        # levels
        self.stop_loss_levels = [0.007, 0.0125, 0.018, 0.022]
        self.ratio_levels = [1.9, 2.4, 2.9, 3.9]
        # variables del config
        self.initial_dollars = config.initial_dollars
        self.current_dollars = config.initial_dollars
        self.risk_factor = config.risk_factor
        self.max_current_dollars = config.initial_dollars
        self.original_initial_dollars = config.initial_dollars
        # clases del qlearning
        self.agent = agent
        self.environment = agent.environment

        # stop loss y ratio
        self.stop_loss_level = 0
        self.ratio_level = 0
        self.stop_loss = self.stop_loss_levels[self.stop_loss_level]
        self.ratio = self.ratio_levels[self.ratio_level]
        self.take_profit = self.stop_loss * self.ratio
        self.stop_loss_price = None
        self.take_profit_price = None
        
        
        # variables de la clase
        self.in_position = False
        self.position_type = None
        self.entry_price = None
        self.signal_o = 0
        self.signal_c = 0
        self.fee_factor = 0.0025
        self.last_price = self.environment.current_close
        self.position_size = 0
        self.step_on_max_current_dollars = self.agent.environment.starting_step
        self.in_trade_current_dollars=0
        self.stop_loss_reached = False
        self.take_profit_reached = False   

        # variables que se usan en el reward o en el plot
        self.last_win = True
        self.this_win = True
        self.just_closed_position = False
        self.just_closed_profitable_position = False
        self.old_step_on_max_current_dollars = self.step_on_max_current_dollars

    def open_position(self, action):
        self.stop_loss = self.stop_loss_levels[action[1]]
        self.ratio = self.ratio_levels[action[2]]
        position_type = action[0]
        self.take_profit = self.risk_factor * self.ratio
        
        self.entry_price = self.environment.current_close
        self.position_size = (self.current_dollars * self.risk_factor) / self.stop_loss
        self.in_trade_current_dollars = self.current_dollars * (1 - self.risk_factor)
        
        if position_type in [1, 2]:  # Assuming 1 for long and 2 for short
            self.in_position = True
            self.position_type = "long" if position_type == 1 else "short"
            self.signal_o = self.position_type
            self.stop_loss_price = self.get_stop_loss_price()
            self.take_profit_price = self.get_take_profit_price()
        else:
            self.in_position = False
            self.position_type = None
            self.signal_o = 0

        self.last_action = action


    def evaluate_position(self):
        if self.position_type == "long":
            self.stop_loss_reached = self.environment.current_low < self.stop_loss_price
            self.take_profit_reached = self.environment.current_high > self.take_profit_price
        elif self.position_type == "short":
            self.stop_loss_reached = self.environment.current_high > self.stop_loss_price
            self.take_profit_reached = self.environment.current_low < self.take_profit_price
        need_to_close = self.stop_loss_reached or self.take_profit_reached
        return need_to_close

    def close_position(self):
        print("closing trade")
        self.current_dollars = self.update_current_dollars()
        self.signal_c = "close"
        self.entry_price = None
        self.position_size = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.position_type = None
        self.in_position = False
        print("current dollars after closing the trade", self.current_dollars)



    def update_max_current_dollars(self):
        if self.current_dollars > self.max_current_dollars:
            self.max_current_dollars = self.current_dollars
            self.step_on_max_current_dollars = self.environment.current_step

    def get_stop_loss_price(self):
        if self.position_type == "long":
            self.stop_loss_price = self.entry_price * (1 - self.stop_loss_levels[self.stop_loss_level])
        elif self.position_type == "short":
            self.stop_loss_price = self.entry_price * (1 + self.stop_loss_levels[self.stop_loss_level])
        return self.stop_loss_price

    def get_take_profit_price(self):
        if self.position_type == "long":
            
            self.take_profit_price = self.entry_price * (1 + self.stop_loss_levels[self.stop_loss_level] * self.ratio_levels[self.ratio_level])
        elif self.position_type == "short":
            self.take_profit_price = self.entry_price * (1 - self.stop_loss_levels[self.stop_loss_level] * self.ratio_levels[self.ratio_level])
        return self.take_profit_price

    def update_current_dollars(self):
    

        if self.stop_loss_reached:
            fee_rate = 0.0032  # Increased fee for a losing position
            new_dollars = self.current_dollars * (1 - self.risk_factor)
            self.just_closed_profitable_position = False  # The position was closed with a loss
        elif self.take_profit_reached:
            fee_rate = 0.0018  # Reduced fee for a winning position
            new_dollars = self.current_dollars * (1 + self.risk_factor * self.ratio)

            self.just_closed_profitable_position = True  # The position was closed with profit
        else:
            return   # No fee if the position is not closed
        new_dollars = new_dollars * (1 - fee_rate)
        return new_dollars 
        




