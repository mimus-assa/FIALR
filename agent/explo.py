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
        self.action_size = agent.environment.action_space[0].n
        # Define a list of stop loss levels
        self.stop_loss_levels = agent.portfolio_manager.stop_loss_levels
        # Define a list of ratio levels
        self.ratio_levels = agent.portfolio_manager.ratio_levels
        # Initialize the exploration probability (epsilon) from the agent's config
        self.epsilon = agent.config.epsilon_start
        # Define the end value for the epsilon from the agent's config
        self.epsilon_end = agent.config.epsilon_end
        # Define the rate at which epsilon will decay from the agent's config
        self.epsilon_decay = agent.config.epsilon_decay

    # Define a method to decide whether to explore or exploit based on the current epsilon
    def should_explore(self, epsilon):
        return np.random.rand() <= epsilon

    # Define a method to choose a random action
    def choose_random_action(self,predict_function, state):
        random_number = np.random.rand()
        # If the random number is less than or equal to 0.33, choose the "hold" action
        if random_number <= 0.33:
            action = 0
            stop_loss = 0  # Assuming a stop_loss level of 0 is a "hold" action
            ratio = 0  # Assuming a ratio level of 0 is a "hold" action
        else:   
            # Otherwise, choose a random action and stop loss and ratio levels
            action = random.randrange(1, self.action_size)
            stop_loss = random.randrange(len(self.stop_loss_levels))
            ratio = random.randrange(len(self.ratio_levels))   
        return (action, stop_loss, ratio)

    # Define a method to choose the best action based on the predict function
    def choose_best_action(self, predict_function, state):
        # Get the predicted q_values from the predict function
        q_values = predict_function(state, verbose=0)
        # Unpack q_values into discrete action, stop loss, and ratio
        action, stop_loss, ratio = np.argmax(q_values[0]), np.argmax(q_values[1]), np.argmax(q_values[2])
        return (action, stop_loss, ratio)


    def execute_action(self, action, current_close):
        # Execute given action
        discrete_action, stop_loss, ratio = action
        if discrete_action in [1, 2]:
            position_types = {1: "long", 2: "short"}
            self.portfolio_manager.signal_o = f"{position_types[discrete_action]}"
            self.portfolio_manager.position_type = position_types[discrete_action]
            if self.portfolio_manager.position_type is not None:
                self.portfolio_manager.entry_price = current_close
                self.portfolio_manager.stop_loss = stop_loss
                self.portfolio_manager.ratio = ratio
                self.portfolio_manager.take_profit = self.portfolio_manager.risk_factor * self.portfolio_manager.ratio
                self.portfolio_manager.in_position = True

                # Actualiza los precios de stop loss y take profit
                self.portfolio_manager.get_stop_loss_price()
                self.portfolio_manager.get_take_profit_price()

        self.last_action = action

    def choose_action(self, state, epsilon):
        # Choose action based on current state and epsilon
        if self.agent.portfolio_manager.in_position:
            action = (self.agent.HOLD, 0 , 0)
        else:
            if self.should_explore(epsilon):
                state = np.reshape(state, (1, state.shape[0], state.shape[1]))
                action = self.choose_random_action(self.agent.model_manager.model.predict, state)
            else:
                state = np.reshape(state, (1, state.shape[0], state.shape[1]))
                action = self.choose_best_action(self.agent.model_manager.model.predict, state)
        return action
    
    # Define a method to update epsilon
    def update_epsilon(self):
        # Update the epsilon, ensuring it never goes below the epsilon_end value
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


    def validate_action(self, action):
        # Validate action and reset signals
        self.portfolio_manager.signal_o = 0#talves el problema con el current dollar es que estamos llamando el vaalidate action siempre
        self.portfolio_manager.signal_c = 0
        discrete_action, stop_loss, ratio = action

        assert self.environment.action_space[0].contains(discrete_action), f"Invalid discrete action {discrete_action}"
        assert self.environment.action_space[1].contains(stop_loss), f"Invalid stop loss {stop_loss}"
        assert self.environment.action_space[2].contains(ratio), f"Invalid ratio {ratio}"
