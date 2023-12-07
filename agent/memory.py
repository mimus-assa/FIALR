import numpy as np
import random
from collections import deque
import tensorflow as tf

class ExperienceReplay:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)

    def remember_experience(self, current_state, action, reward, next_state, done):
        self.memory.append((current_state, action, reward, next_state, done))

    def sample_experiences(self, batch_size):
        batch_size = min(len(self.memory), batch_size)
        experiences = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return np.array(states), np.array(actions), np.array(rewards), np.squeeze(np.array(next_states), axis=1), np.array(dones)

    def replay_experiences(self, batch_size, target_model, deep_model, gamma):
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.sample_experiences(batch_size)

        # Convertir a tensores TensorFlow
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Calcular Q-values objetivo para el próximo estado
        next_Q_values = target_model.model.predict(next_states, verbose=0)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        targets = rewards + (1 - dones) * gamma * max_next_Q_values


        with tf.GradientTape() as tape:
            # Calcular la pérdida
            states = tf.squeeze(states, axis=1)
            all_Q_values = deep_model.model(states)
            Q_values = tf.reduce_sum(all_Q_values * tf.one_hot(actions, all_Q_values.shape[1]), axis=1)
            loss = tf.reduce_mean(tf.square(targets - Q_values))

        # Calcular gradientes y aplicarlos
        gradients = tape.gradient(loss, deep_model.model.trainable_variables)
        deep_model.model.optimizer.apply_gradients(zip(gradients, deep_model.model.trainable_variables))

        return loss.numpy()


    def clean_memory(self):
        self.memory.clear()
