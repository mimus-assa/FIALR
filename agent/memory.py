# Import necessary libraries and modules
import numpy as np
from collections import deque
import tensorflow as tf
import os

# Define the ExperienceReplay class
class ExperienceReplay:
    # Initialize the class with an agent
    def __init__(self, agent):
        # Set memory size and batch size from agent
        self.memory_size = agent.memory_size
        self.batch_size = agent.batch_size
        # Initialize memory and next_index
        self.memory = [None] * self.memory_size
        self.next_index = 0

    # Method to remember experience
    def remember_experience(self, current_state, action, reward, next_state, done):
        # Save experience in memory
        self.memory[self.next_index] = (current_state, action, reward, next_state, done) # action is now a list
        # Update next_index
        self.next_index = (self.next_index + 1) % self.memory_size

    # Method to sample experiences
    def sample_experiences(self):
        # Remove None values from memory
        valid_memory = [m for m in self.memory if m is not None]
        # Return None if not enough valid memories
        if len(valid_memory) < self.batch_size:
            return None
        # Randomly select indices
        indices = np.random.choice(len(valid_memory), self.batch_size, replace=False)
        # Get batch from memories
        batch = [valid_memory[idx] for idx in indices]
        # Separate out states, actions, rewards, next_states, dones
        states = tf.constant([ex[0] for ex in batch], dtype=tf.float32)

        discrete_actions = tf.constant([ex[1] for ex in batch], dtype=tf.float32)


        rewards = [float(ex[2]) if isinstance(ex[2], np.ndarray) else ex[2] for ex in batch]
        rewards = tf.constant(rewards, dtype=tf.float32)
        next_states = tf.constant([ex[3] for ex in batch], dtype=tf.float32)
        dones = tf.constant([float(ex[4]) for ex in batch], dtype=tf.float32)

        return states, discrete_actions, rewards, next_states, dones

    # Method to clean memory
    def clean_memory(self):
        self.memory = [None] * self.memory_size
        self.next_index = 0

    # Method to compute targets
    def compute_targets(self, rewards, next_states, dones, target_model):
        # Get predictions for next states
        next_state_values = target_model.model.predict_on_batch(next_states)

        # Adjust next_state_values based on rewards and dones
        # Por ejemplo, podrías hacer algo como:
        # next_state_values = rewards + (1 - dones) * self.gamma * next_state_values.max(axis=1)
        # Pero necesitas definir 'self.gamma' (factor de descuento) en alguna parte de tu clase

        return next_state_values

    # Method to update deep model
    def update_deep_model(self, states, actions, targets, deep_model, target_action_probabilities):
        # Asegúrate de que 'targets' es un tensor de la forma correcta
        # No es necesario convertirlo en una lista de tensores, ya que tu modelo espera un solo target
        result = deep_model.model.train_on_batch(states, targets)

        return result


    # Method to replay experiences
    def replay_experiences(self, target_model, deep_model):
        # Sample experiences
        experiences = self.sample_experiences()
        
        if experiences is None:
            return None
        states, discrete_actions, rewards, next_states, dones = experiences
        actions = discrete_actions

        # Get probabilities of target actions
        target_action_probabilities = deep_model.model.predict_on_batch(states)

        target_action_probabilities = [tf.convert_to_tensor(tap) for tap in target_action_probabilities]

        # Compute targets
        targets = self.compute_targets(rewards, next_states, dones, target_model)
        # Update deep model
        loss = self.update_deep_model(states, actions, targets, deep_model, target_action_probabilities)

        return loss
