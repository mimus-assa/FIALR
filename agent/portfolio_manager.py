import numpy as np

class PortfolioManager:
    def __init__(self, config, agent):
        self.stop_loss_levels = [0.007, 0.0125, 0.018, 0.022]
        self.ratio_levels = [2, 2.5, 3, 4]
        self.initial_dollars = config.initial_dollars
        self.agent=agent
        self.environment=agent.environment
        self.current_dollars = config.initial_dollars
        self.risk_factor = config.risk_factor
        self.stop_loss = self.stop_loss_levels[0]
        self.max_current_dollars = config.initial_dollars
        self.in_position = False
        self.position_type = None
        self.entry_price = None
        self.signal_o = 0
        self.signal_c = 0
        self.current_open = None
        self.current_high = None
        self.current_low = None
        self.current_close = None
        self.current_ts = None
        self.original_initial_dollars = config.initial_dollars
        self.fee_factor = 0.0025
        self.ratio = self.ratio_levels[0]
        self.take_profit= self.stop_loss * self.ratio
        self.last_win=True
        self.this_win=True
        self.penalty =0
        self.bonuses =0
        self.just_closed_position = False
        self.just_closed_profitable_position = False
        self.step_on_max_current_dollars=self.agent.environment.starting_step
        self.old_step_on_max_current_dollars=self.step_on_max_current_dollars
        self.stop_loss_price = None
        self.take_profit_price = None

    def evaluate_and_close_position(self):
        if self.in_position:
            stop_loss_reached, take_profit_reached = self._evaluate_position()
            
            if stop_loss_reached or take_profit_reached:
                # Llama a plotter aquí, antes de que los valores se restablezcan a None
                if self.agent.plot:  # Asume que tienes una referencia al agente y que plot es un atributo booleano
                    self.agent.trainer.plot_agent.plotter()
                
                # Ahora cierra la posición
                self._close_position(stop_loss_reached, take_profit_reached)

    def _evaluate_position(self):
        stop_loss_price = self.stop_loss_price
        take_profit_price = self.take_profit_price

        if self.position_type == "long":
            stop_loss_reached = self.environment.current_low < stop_loss_price
            take_profit_reached = self.environment.current_high > take_profit_price
        elif self.position_type == "short":
            stop_loss_reached = self.environment.current_high > stop_loss_price
            take_profit_reached = self.environment.current_low < take_profit_price
        return stop_loss_reached, take_profit_reached


    def _update_portfolio_value(self, stop_loss_reached, take_profit_reached):
        if self.in_position and self.signal_c == "closed":
            if stop_loss_reached:
                new_dollars = self.current_dollars * (1 - self.risk_factor)
                fee_rate = 0.0036  # Increased fee for a losing position
                self.just_closed_profitable_position = False  # The position was closed with a loss
            elif take_profit_reached:
                new_dollars = self.current_dollars * (1 + self.risk_factor * self.ratio_levels[self.ratio])
                fee_rate = 0.002  # Reduced fee for a winning position
                self.just_closed_profitable_position = True  # The position was closed with profit
            else:
                return  # No update if the position is not closed
            
            # Subtract the fee
            new_dollars = new_dollars * (1 - fee_rate)
            self.current_dollars = new_dollars 

    def get_stop_loss_price(self):
        if self.in_position:
            if self.position_type == "long":
                self.stop_loss_price = self.entry_price * (1 - self.stop_loss_levels[self.stop_loss])
            elif self.position_type == "short":
                self.stop_loss_price = self.entry_price * (1 + self.stop_loss_levels[self.stop_loss])
        return self.stop_loss_price

    def get_take_profit_price(self):
        if self.in_position:
            if self.position_type == "long":
                
                self.take_profit_price = self.entry_price * (1 + self.stop_loss_levels[self.stop_loss] * self.ratio_levels[self.ratio])
            elif self.position_type == "short":
                self.take_profit_price = self.entry_price * (1 - self.stop_loss_levels[self.stop_loss] * self.ratio_levels[self.ratio])
        return self.take_profit_price

    def _close_position(self, stop_loss_reached, take_profit_reached):
        if self.in_position:
            # Add this line to update the consecutive wins counter before closing the position
            #self.agent.reward_and_punishment.lucky_strike_bonus()
            
            self.signal_c = "closed"
            self._update_portfolio_value(stop_loss_reached, take_profit_reached)
            self.in_position = False
            self.position_type = None
            self.entry_price = None
            self.stop_loss_price = None
            self.take_profit_price = None
            self.just_closed_position = True

            # Reset stop loss and take profit prices when position is closed
            self.stop_loss_price = None
            self.take_profit_price = None

    def update_max_portfolio_value(self):
        if self.current_dollars > self.max_current_dollars:
            if self.agent.environment.current_step>self.old_step_on_max_current_dollars:
                self.step_on_max_current_dollars=self.agent.environment.current_step
            self.old_step_on_max_current_dollars=self.step_on_max_current_dollars
            self.max_current_dollars = self.current_dollars

