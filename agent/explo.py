# Import necessary libraries
import numpy as np
import random 

# Define the ExplorationExploitation class
class ExplorationExploitation:
    # Initialize the class with an agent as input
    def __init__(self, agent): 
        self.agent=agent
        self.environment = agent.environment
        self.portfolio_manager = agent.portfolio_manager
        # Define the number of possible actions based on the agent's environment
        self.action_size = agent.environment.action_space.n

        # Initialize the exploration probability (epsilon) from the agent's config
        self.epsilon = agent.config.epsilon_start
        # Define the end value for the epsilon from the agent's config
        self.epsilon_end = agent.config.epsilon_end
        # Define the rate at which epsilon will decay from the agent's config
        self.epsilon_decay = agent.config.epsilon_decay
        self.hold_action_threshold = 0.3
        self.adjust_epsilon_start()

    def adjust_epsilon_start(self):
        # Calcula la base del decaimiento exponencial
        # Queremos que en el último episodio, epsilon sea epsilon_start / 5
        b = (1/300) ** (1 / (self.agent.config.episodes - 1))

        # Calcula el factor de reducción exponencial
        reduction_factor = b ** self.agent.episode

        # Ajusta epsilon_start
        self.epsilon = self.agent.config.epsilon_start * reduction_factor


    # Define a method to decide whether to explore or exploit based on the current epsilon
    def should_explore(self, epsilon):
        return np.random.rand() <= epsilon

    # Define a method to choose a random action
    def choose_random_action(self):
        random_number = np.random.rand()
        # Utiliza el umbral pre-calculado para decidir la acción
        if random_number <= self.hold_action_threshold:
            action_level = 0  # "hold" action
        else:
            # De lo contrario, elige una acción aleatoria y los niveles de stop loss y ratio
            if self.portfolio_manager.in_position:
                action_level = 3
            else:
                action_level = random.randrange(1, self.action_size-1)
        return action_level

    # Define a method to choose the best action based on the predict function
    def choose_best_action(self, predict_function, state):
        # Get the predicted q_values from the predict function
        q_values = predict_function(state, verbose=0)
        # Unpack q_values into discrete action, stop loss, and ratio
        action_level = np.argmax(q_values[0])
        #print("choose_best_action", "step ",self.environment.current_step, "action ",action_level)
        return action_level


    def choose_action(self, state, epsilon):
        state_reshaped = np.reshape(state, (1, state.shape[1], state.shape[2]))
        
        if self.should_explore(epsilon):
            
            action = self.choose_random_action()
           # print("random action", action)
        else:
            action = self.choose_best_action(self.agent.model_manager.model.predict, state_reshaped)
           # print("best action", action)
        return action


    
    # Define a method to update epsilon
    def update_epsilon(self):
        #we do this only on the step 0 of the episode
        if self.agent.environment.current_step == 0:
            self.adjust_epsilon_start()
        # Update the epsilon, ensuring it never goes below the epsilon_end value
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
      #  print("epsilon: ", self.epsilon, "step: ", self.environment.current_step)


    def validate_action(self, action):
        # Validate action and reset signals
        self.portfolio_manager.signal_o = 0#talves el problema con el current dollar es que estamos llamando el vaalidate action siempre
        self.portfolio_manager.signal_c = 0
        #discrete_action, stop_loss, ratio = action

        #assert self.environment.action_space[0].contains(discrete_action), f"Invalid discrete action {discrete_action}"
        #assert self.environment.action_space[1].contains(stop_loss), f"Invalid stop loss {stop_loss}"
        #assert self.environment.action_space[2].contains(ratio), f"Invalid ratio {ratio}"
