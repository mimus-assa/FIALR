import numpy as np

class PortfolioManager:
    def __init__(self, config, agent):
        # levels
 
        # variables del config
        self.initial_dollars = config.initial_dollars
        self.current_dollars = config.initial_dollars
        self.max_current_dollars = config.initial_dollars
        self.original_initial_dollars = config.initial_dollars
        # clases del qlearning
        self.agent = agent
        self.environment = agent.environment
        
        
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
  
        # variables que se usan en el reward o en el plot
        self.last_win = True
        self.this_win = True
        self.just_closed_position = False
        self.just_closed_profitable_position = False
        self.old_step_on_max_current_dollars = self.step_on_max_current_dollars
        self.pnl_for_reward =0

    def open_position(self, action):
       # print("opening trade")
        position_type = action
        
        self.entry_price = self.environment.current_close
        self.position_size = self.initial_dollars
        
        #if position_type in [1, 2]:  # Assuming 1 for long and 2 for short
        self.in_position = True
        self.position_type = "long" if position_type == 1 else "short"
      #  print(self.position_type)
        self.signal_o = self.position_type
     #   print("entry price at opening trade", self.entry_price, "position type", self.position_type)

        


    def close_position(self):
     #   print("closing trade")
     
        self.current_dollars = self.update_current_dollars()
        self.signal_c = "close"
      #  print("closing, step", self.environment.current_step , "price at closing trade", self.environment.current_close,"entry price", self.entry_price  , "current dollars", self.current_dollars)

        self.pnl_for_reward =self.entry_price-self.environment.current_close -(self.current_dollars * 0.0032)
       # print("este es el step donde esta valiendo verga", self.environment.current_step)
        self.entry_price = None
        self.position_size = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.position_type = None
        self.in_position = False
      #  print("current dollars after closing the trade", self.current_dollars)
        



    def update_max_current_dollars(self):
        if self.current_dollars > self.max_current_dollars:
            self.max_current_dollars = self.current_dollars
            self.step_on_max_current_dollars = self.environment.current_step


    def update_current_dollars(self):
        fee_rate = 0.0032
        # Calcula el PnL con el precio de entrada y el precio actual
        if self.position_type == "long":
            pnl = self.position_size * (self.environment.current_close - self.entry_price) / self.entry_price
        elif self.position_type == "short":
            pnl = -(self.position_size * (self.environment.current_close - self.entry_price) / self.entry_price)
        else:
            pnl = 0
        new_dollars = self.current_dollars + pnl - (self.current_dollars * fee_rate)
        return new_dollars

