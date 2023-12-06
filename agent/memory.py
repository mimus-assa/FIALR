# Import necessary libraries and modules
import numpy as np
from collections import deque
import tensorflow as tf
import os

# Define the ExperienceReplay class
class ExperienceReplay:
    def __init__(self, agent):
        self.current_experience = None

    def remember_experience(self, current_state, action, reward, next_state, done):
        # Solo se guarda la experiencia más reciente
        self.current_experience = (current_state, action, reward, next_state, done)

    def sample_experiences(self):
        # Se devuelve la experiencia actual si existe
        return self.current_experience if self.current_experience is not None else None

    def replay_experiences(self, target_model, deep_model):
        experience = self.sample_experiences()

        if experience is None:
            return None

        # Descomposición de la experiencia
        state, action, reward, next_state, done = experience

        # Comprueba si algún elemento de la experiencia es None individualmente
        if state is None or action is None or reward is None or next_state is None or done is None:
            print("Experiencia incompleta o incorrecta.")
            return None

        # Convertir a tensores
        state = tf.constant(state, dtype=tf.float32)
        action = tf.constant(action, dtype=tf.float32)
        reward = tf.constant(reward, dtype=tf.float32)
        next_state = tf.constant(next_state, dtype=tf.float32)
        done = tf.constant(done, dtype=tf.float32)


        # Calcular el target (esto depende de cómo esté definido tu target_model)
        target = self.compute_targets(reward, next_state, done, target_model)

        # Actualizar el modelo
        loss = self.update_deep_model(state, action, target, deep_model)

        return loss


    # Method to clean memory
    def clean_memory(self):
        # Restablecer la experiencia actual a None
        self.current_experience = None

    # Method to compute targets
    def compute_targets(self, rewards, next_states, dones, target_model):
        # Asegúrate de que next_states tenga la forma correcta
        next_states = np.expand_dims(next_states, axis=0) if next_states.ndim == 2 else next_states

        # Obtén las predicciones para los siguientes estados
        next_state_values = target_model.model.predict_on_batch(next_states)

        # Ajusta next_state_values basado en las recompensas y dones
        # Por ejemplo, podrías hacer algo como:
        # next_state_values = rewards + (1 - dones) * self.gamma * next_state_values.max(axis=1)
        # Pero necesitas definir 'self.gamma' (factor de descuento) en alguna parte de tu clase

        return next_state_values


    # Method to update deep model
    def update_deep_model(self, states, actions, targets, deep_model):
        # Asegúrate de que states tenga la forma correcta
        states = np.expand_dims(states, axis=0) if states.ndim == 2 else states
        targets = np.expand_dims(targets, axis=0) if targets.ndim == 1 else targets
        result = deep_model.model.train_on_batch(states, targets)
        return result


