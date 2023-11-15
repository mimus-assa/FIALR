from plot.base_model import SubplotBase


class TextValuesSubplot(SubplotBase):
    def __init__(self, fig, position, agent, env, portfolio_manager):
        super().__init__(fig, position)
        self.agent = agent
        self.env = env
        self.portfolio_manager = portfolio_manager

    def update_limits(self):
        # No es necesario actualizar límites para texto
        pass

    def plot(self):
        # Clear the subplot
        self.ax.clear()

        # Add each piece of data as a new line of text
        self.ax.text(0, 1.0, f"Episode: {self.agent.episode}", transform=self.ax.transAxes)
        self.ax.text(0, 0.95, f"Step: {self.env.current_step}", transform=self.ax.transAxes)
        self.ax.text(0, 0.9, f"Current Dollars: {round(self.portfolio_manager.current_dollars,2)}", transform=self.ax.transAxes)
        self.ax.text(0, 0.85, f"Max Current Dollars: {round(self.portfolio_manager.max_current_dollars,2)}", transform=self.ax.transAxes)
        self.ax.text(0, 0.75, f"Stop Loss Price: {'None' if self.portfolio_manager.stop_loss_price is None else round(self.portfolio_manager.stop_loss_price,2)}", transform=self.ax.transAxes)
        self.ax.text(0, 0.7, f"Take Profit Price: {'None' if self.portfolio_manager.take_profit_price is None else round(self.portfolio_manager.take_profit_price,2)}", transform=self.ax.transAxes)
        self.ax.text(0, 0.65, f"Position Type: {self.portfolio_manager.position_type}", transform=self.ax.transAxes)
        self.ax.text(0, 0.6, f"Bonuses: {round(self.agent.reward_and_punishment.bonuses,4)}", transform=self.ax.transAxes)
        self.ax.text(0, 0.55, f"Penalty: {round(self.agent.reward_and_punishment.penalty,4)}", transform=self.ax.transAxes)
        self.ax.text(0, 0.5, f"Reward: {round(self.agent.trainer.reward,4)}", transform=self.ax.transAxes)
        self.ax.text(0, 0.45, f"Win Streak: {self.agent.reward_and_punishment.win_streak}", transform=self.ax.transAxes)
        self.ax.text(0, 0.4, f"Lose Streak: {self.agent.reward_and_punishment.lose_streak}", transform=self.ax.transAxes)

        # Remove the x and y axes
        self.ax.axis('off')

    def reset(self):
        super().reset()
        # No hay datos adicionales para resetear aquí