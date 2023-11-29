import numpy as np
import tqdm
from plot.plot_agent import PlotAgent
import random


class AgentTrainer:
    def __init__(self, agent):
        self.agent = agent
        self.environment=agent.environment
        self.config = agent.config
        self.portfolio_manager = agent.portfolio_manager
        self.agent.reset_agent()
        self.losses_0 = []
        self.plot_agent = PlotAgent(self.environment, self.portfolio_manager, self.agent)  # initialize the PlotAgent
        self.reward=0
        self.features_from_training=[]
        self.exploration_explotation = self.agent.exploration_explotation
        self.reward_and_punishment = self.agent.reward_and_punishment
        self.plot = True
    
    def calculate_features(self):
        num_features = 5
        features = [None] * num_features  # Pre-allocación de la lista
        current_dollars = self.portfolio_manager.current_dollars
        max_current_dollars = self.portfolio_manager.max_current_dollars
        in_position = int(self.portfolio_manager.in_position)
        max_current_dollars_stop_price = self.portfolio_manager.max_current_dollars * self.config.stop_price
        values = [current_dollars, max_current_dollars, max_current_dollars_stop_price]
        normalized_values = [self.agent.get_normalized_values(value) for value in values]
        features[0] = normalized_values[0]
        features[1] = normalized_values[1]
        features[2] = in_position
        features[3] = normalized_values[2]
        features[4] = self.agent.last_action
        return features

    def _train_step(self, current_state, episode):
        self.portfolio_manager.signal_c=0
        self.portfolio_manager.signal_o = 0
        self.agent.portfolio_manager.pnl_for_reward=0
        if self.environment.current_step == self.config.max_steps-1:
            action = self.agent.CLOSE
        else:
            action = self.exploration_explotation.choose_action(current_state, self.config.epsilon_start)
        if not self.portfolio_manager.in_position:
            if action in [self.agent.LONG, self.agent.SHORT]:
                self.portfolio_manager.open_position(action)
        elif self.portfolio_manager.in_position:
            if action == self.agent.CLOSE:
                self.portfolio_manager.close_position()
        features = self.calculate_features()
        self.features_from_training.append(features)
        self.environment.update_step_features(self.environment.current_step, features)
        next_state = self.agent.update_observation(self.environment.step())
        self.portfolio_manager.update_max_current_dollars()
        self.agent.last_action = action
        reward = self.reward_and_punishment.calculate_reward(self.agent.portfolio_manager.pnl_for_reward)
        self.reward += reward  # Acumula la recompensa
     #   print("step", self.agent.environment.current_step, "reward", reward , "total reward", self.reward)
        done = self.agent.is_done()
        self.agent.experience_replay.remember_experience(current_state, action, reward, next_state, done)
        if self.plot:
            self.plot_agent.plotter(self.environment.current_step)

        return next_state, reward, done



    def _update_metrics(self, reward, total_reward):
        self.agent.episode_rewards.append(reward)
        total_reward += self.reward 
        if len(self.agent.experience_replay.memory) > self.config.batch_size:
            self._update_losses()
            self._update_model()
        return total_reward


    def _update_losses(self):
        loss = self.agent.experience_replay.replay_experiences(self.agent.target_model, self.agent.model_manager)
        if loss is not None:
            self.losses_0.append(loss)



    def _update_model(self):
        if self.environment.current_step % self.config.target_update_frequency == 0:
            self.agent.model_manager.update_target_model(self.agent.target_model, tau=0.01)

    def _cleanup_train(self, episode):
        self._update_average_losses()
        self._clear_losses()

        self.agent.record_and_print_score(episode, self.config.episodes)
        self.agent.episode_rewards.clear()

        self.agent.experience_replay.clean_memory()

    def _update_average_losses(self):
        average_losses = [self._calculate_average_loss(loss_list) for loss_list in [self.losses_0]]
        self.agent.model_manager.average_losses = average_losses  
         

    def _calculate_average_loss(self, loss_list):
        return np.mean(loss_list) if len(loss_list) > 0 else None

    def _clear_losses(self):
        for loss_list in [self.losses_0]:
            loss_list.clear()

    def train(self, episode):
        if self.plot:
            self.plot_agent.reset()  # Reiniciar gráficos para el nuevo episodio
        current_state = self.agent.setup_for_training(episode)
        total_reward = 0
        self.reward = 0
        progress_bar = self._initialize_progress_bar()
        if self.plot:
            self.plot_agent.init_plot()

        done = False
        while not done:           
            current_state, reward, done = self._train_step(current_state, episode)
            total_reward = self._update_metrics(reward, total_reward)
            progress_bar.update()
            self.agent.exploration_explotation.update_epsilon()

        self._finalize_training(episode, progress_bar)


    def _finalize_training(self, episode, progress_bar):
        progress_bar.close()
        self._reset_portfolio_manager()
        self._cleanup_train(episode)
        self._check_and_reset_environment_step()
        if self.plot:
            self.plot_agent.save_plot()   

    def _reset_portfolio_manager(self):
        self.portfolio_manager.take_profit_price = None
        self.portfolio_manager.stop_loss_price = None


    def _initialize_progress_bar(self):
        steps_this_episode = self.config.max_steps - self.environment.current_step-1
        return tqdm.tqdm(total=steps_this_episode, desc="Training Progress")

    def _check_and_reset_environment_step(self):
        if self.environment.current_step >= self.config.max_steps-1:
            self.environment.current_step = self.config.starting_step


 