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
        #print("current_state shape", current_state.shape)
        # Configuración inicial
        self.portfolio_manager.signal_c = 0
        self.portfolio_manager.signal_o = 0
        self.portfolio_manager.fee = 0
        self.portfolio_manager.pnl = 0

        current_reward = 0
        # Decidir la acción a tomar
        # Decidir la acción a tomar
        if self.environment.current_step == self.config.max_steps-1:
            action = self.agent.CLOSE
        elif self.environment.current_step in [i for i in range(0,self.config.window_size)]:
            action = self.agent.HOLD
        else:
            action = self.exploration_explotation.choose_action(current_state, self.config.epsilon_start)

        current_reward = self.reward_and_punishment.calculate_reward()
        self.reward += current_reward
        # Ejecutar la acción seleccionada y actualizar la posición
        if not self.portfolio_manager.in_position:
            if action in [self.agent.LONG, self.agent.SHORT]:
                self.portfolio_manager.open_position(action)
        elif self.portfolio_manager.in_position:
            if action == self.agent.CLOSE:
                self.portfolio_manager.close_position()

        # Registrar la acción actual
        self.agent.last_action = action

       

        features = self.calculate_features()
        self.features_from_training.append(features)
        self.environment.update_step_features(self.environment.current_step, features)
        self.portfolio_manager.update_max_current_dollars()
         
    
        if self.plot:
            self.plot_agent.plotter(self.environment.current_step)

        next_state = self.agent.update_observation(self.environment.step())
       # print("shape of the next state", next_state.shape)
        done = self.agent.is_done()
        self.agent.experience_replay.remember_experience(current_state, self.agent.last_action, current_reward, next_state, done)
        
        # Retornar el estado siguiente, la recompensa y la bandera de terminado
        return next_state, current_reward, done




    def _update_metrics(self, reward, total_reward):
        self.agent.episode_rewards.append(reward)
        total_reward += self.reward 

        # Actualizar las pérdidas y el modelo después de cada paso
        self._update_losses()
        self._update_model()

        return total_reward



    def _update_losses(self):
        loss = self.agent.experience_replay.replay_experiences(self.agent.batch_size ,self.agent.target_model, self.agent.model_manager, 0.99)
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

       # self.agent.experience_replay.clean_memory()

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
        self.total_reward = 0
        self.reward = 0
        progress_bar = self._initialize_progress_bar()
        if self.plot:
            self.plot_agent.init_plot()

        done = False
        while not done:           
            current_state, reward, done = self._train_step(current_state, episode)
            self.total_reward = self._update_metrics(reward, self.total_reward)
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


 