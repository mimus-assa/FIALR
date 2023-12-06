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
    self.fee = 0
    self.pnl=0
    self.last_pnl=0
    self.last_fee=0
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
    self.position_size = self.current_dollars
    
    #if position_type in [1, 2]:  # Assuming 1 for long and 2 for short
    self.in_position = True
    self.position_type = "long" if position_type == 1 else "short"
  #  print(self.position_type)
    self.signal_o = self.position_type
  #   print("entry price at opening trade", self.entry_price, "position type", self.position_type)
  #  print(f"Abriendo posición: Tipo = {self.position_type}, Precio de entrada = {self.entry_price}")
    


  def close_position(self):
    self.pnl = self.calculate_pnl(self.position_size)
    self.fee = self.calculate_fee(self.position_size)
    self.current_dollars += self.pnl - self.fee

    

   #print(f"Cálculo de PnL: PnL = {pnl}, Comisión = {fee}")
   # print(f"Actualización de current_dollars: Después = {self.current_dollars}, PnL = {pnl}, Comisión = {fee}")
    self.last_pnl = self.pnl
    self.last_fee = self.fee
    # Restablecer los valores de la posición.
    self.entry_price = None
    self.position_size = None
    self.stop_loss_price = None
    self.take_profit_price = None
    self.position_type = None
    self.in_position = False
    self.signal_c = "close"

   # print(f"Estado después de cerrar la posición: current_dollars = {self.current_dollars}, position size = {self.position_size}, entry price = {self.entry_price}")

    

  def get_pnl(self):
    # Método para obtener el PnL actual
    return self.calculate_pnl(self.position_size)


  def get_fee(self):
      # Método para obtener la tarifa actual
      return self.calculate_fee(self.position_size)

  def update_max_current_dollars(self):
    if self.current_dollars > self.max_current_dollars:
        self.max_current_dollars = self.current_dollars
        self.step_on_max_current_dollars = self.environment.current_step


 


  def calculate_fee(self, position_size):
    fee_rate = 0.0032
    fee = position_size * fee_rate
    return fee

  def calculate_pnl(self, position_size):
      if self.position_type == "long":
          pnl = position_size * (self.environment.current_close - self.entry_price) / self.entry_price
      elif self.position_type == "short":
          pnl = -(position_size * (self.environment.current_close - self.entry_price) / self.entry_price)
      else:
          pnl = 0
      return pnl
