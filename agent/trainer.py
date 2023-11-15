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
        self.losses_1 = []
        self.losses_2 = []
        self.total_losses = []
        self.plot_agent = PlotAgent(self.environment, self.portfolio_manager, self.agent)  # initialize the PlotAgent
        self.reward=0
        self.features_from_training=[]
        

    
    def calculate_features(self):
        features = []
        current_dollars = self.portfolio_manager.current_dollars
        max_current_dollars = self.portfolio_manager.max_current_dollars
        in_position = int(self.portfolio_manager.in_position)
        max_current_dollars_stop_price = self.portfolio_manager.max_current_dollars * self.config.stop_price#la config del env no tiene mucho sentido puede ser movida a otroas configs y evitarnos esta
        ratio = self.portfolio_manager.ratio_levels[self.portfolio_manager.ratio] / 10
        stop_loss = self.portfolio_manager.stop_loss_levels[int(self.portfolio_manager.stop_loss)]
        bonuses = self.agent.reward_and_punishment.bonuses
        penalty = self.agent.reward_and_punishment.penalty

        features.extend([self.agent.get_normalized_values(value) for value in [current_dollars, max_current_dollars, max_current_dollars_stop_price, bonuses, penalty]])
        features.extend([in_position, ratio, stop_loss])

        action, _, _ = self.agent.last_action
        features.extend(np.eye(3)[action])
        return features


    def _train_step(self, current_state, episode):
        discrete_action, self.portfolio_manager.stop_loss, self.portfolio_manager.ratio = self.agent.exploration_explotation.choose_action(current_state, self.config.epsilon_start)
        self.agent.exploration_explotation.validate_action([discrete_action, self.portfolio_manager.stop_loss, self.portfolio_manager.ratio])
        action =  [discrete_action, self.portfolio_manager.stop_loss, self.portfolio_manager.ratio]
        
        # Store the portfolio value at the time of action
        portfolio_value_at_action = self.portfolio_manager.current_dollars

        features = self.calculate_features()
        self.features_from_training.append(features)
        self.environment.update_step_features(self.environment.current_step, features)

        next_state = self.agent.update_observation(self.environment.step())
        
        self.portfolio_manager.evaluate_and_close_position()
        if not self.portfolio_manager.in_position:
            current_close_price = self.environment.current_close
            self.agent.exploration_explotation.execute_action(action, current_close_price)

        self.portfolio_manager.update_max_portfolio_value()

        # Pass the stored portfolio value at action to calculate_reward
        reward = self.agent.reward_and_punishment.calculate_reward(portfolio_value_at_action)
        self.reward=reward
        done = self.agent.is_done()

        self.agent.experience_replay.remember_experience(current_state, action, reward, next_state, done)
        return next_state, reward, done


    def _update_metrics(self, reward, total_reward):
        self.agent.episode_rewards.append(reward)
        total_reward += reward
        if len(self.agent.experience_replay.memory) > self.config.batch_size:
            self._update_losses()
            self._update_model()
        return total_reward


    def _update_losses(self):
        loss = self.agent.experience_replay.replay_experiences(self.agent.target_model, self.agent.model_manager)
        if loss is not None:

            self.losses_0.append(loss[0])
            self.losses_1.append(loss[1])
            self.losses_2.append(loss[2])
            self.total_losses.append(loss[3])


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
        average_losses = [self._calculate_average_loss(loss_list) for loss_list in [self.total_losses, self.losses_0, self.losses_1, self.losses_2]]
        self.agent.model_manager.average_losses = average_losses  
         

    def _calculate_average_loss(self, loss_list):
        return np.mean(loss_list) if len(loss_list) > 0 else None

    def _clear_losses(self):
        for loss_list in [self.total_losses, self.losses_0, self.losses_1, self.losses_2]:
            loss_list.clear()

    def train(self, episode, plot=True):
        current_state = self.agent.setup_for_training(episode)
        total_reward = 0
        progress_bar = self._initialize_progress_bar()

        # Initialize the plot at the start of the episode
        if plot:
            self.plot_agent.init_plot()
        done = False
        # En tu bucle de entrenamiento, llama a plotter justo antes de cerrar la posición
        while not done:
            # Ejecuta un paso de entrenamiento y obtiene el nuevo estado y la recompensa
            current_state, reward, done = self._train_step(current_state, episode)
            
            # Evalúa y potencialmente cierra la posición
            self.portfolio_manager.evaluate_and_close_position()
            
            # Actualiza cualquier métrica necesaria
            total_reward = self._update_metrics(reward, total_reward)
            
            # Actualiza la barra de progreso
            progress_bar.update()
            
            # Plotea los datos del paso actual
            if plot:
                self.plot_agent.plotter()

        progress_bar.close()
        # Resto del método de entrenamiento

        # Reset the plot at the end of the episode
        if plot:
            #self.plot_agent.update_losses(episode)
            self.plot_agent.plot_updater.plot_data()  # add this line
        self._cleanup_train(episode)
        self._check_and_reset_environment_step()
        if plot:
            
            self.plot_agent.reset()


    def _initialize_progress_bar(self):
        steps_this_episode = self.config.max_steps - self.environment.current_step-1
        return tqdm.tqdm(total=steps_this_episode, desc="Training Progress")

    def _check_and_reset_environment_step(self):
        if self.environment.current_step >= self.config.max_steps - 1000:
            self.environment.current_step = self.config.starting_step

