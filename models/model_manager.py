import os
import tensorflow as tf
from tensorflow.keras.models import Model
from models.transformer_model import Transformer, TransformerBlock
from configs import DeepModelConfig

class ModelManager:
    def __init__(self, window_size, action_size, number_of_features, model_path=None, config=None):
        self.window_size = window_size
        self.action_size = action_size
        self.number_of_features = number_of_features
        self.model_path = model_path
        self.config = config if config is not None else DeepModelConfig()
        self.batch_size = self.config.batch_size
        self.average_losses= [None, None, None, None]
        if model_path is not None and os.path.isfile(model_path):
            self.model = self.load_model()
        else:
            self.model = self._build_and_compile_model()

    def _build_and_compile_model(self):
        input_layer = tf.keras.layers.Input(shape=(self.config.window_size, self.number_of_features))
        transformer = Transformer(self.config)
        model = Model(inputs=input_layer, outputs=transformer(input_layer))

        initial_learning_rate = self.config.initial_learning_rate
        optimizer = tf.keras.optimizers.Nadam(learning_rate=initial_learning_rate, clipnorm=self.config.clipnorm)
        
        # Define losses for each output
        losses = {
            "transformer": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            "transformer_1": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            "transformer_2": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        }
        model.compile(loss=losses, optimizer=optimizer)
        return model

    def _update_weights(self, target_weights, online_weights, tau):
        for i in range(len(target_weights)):
            target_weights[i] = tau * online_weights[i] + (1 - tau) * target_weights[i]
        return target_weights

    def update_target_model(self, target_model, tau):
        transformer_layer = self._get_transformer_layer(self.model)
        target_transformer_layer = self._get_transformer_layer(target_model.model)

        for layer_name in ['transformer', 'transformer_1', 'transformer_2']:
            try:
                online_weights = transformer_layer.get_layer(layer_name).get_weights()
                target_weights = target_transformer_layer.get_layer(layer_name).get_weights()

                updated_weights = self._update_weights(target_weights, online_weights, tau)

                target_transformer_layer.get_layer(layer_name).set_weights(updated_weights)
            except ValueError as e:
                print(f"Layer {layer_name} does not exist.")

    def _get_transformer_layer(self, model):
        # Get the transformer layer's name from the model
        transformer_layer_name = [layer.name for layer in model.layers if 'transformer' in layer.name][0]
        transformer_layer = model.get_layer(transformer_layer_name)
        return transformer_layer

    def save_model(self):
        if self.model_path is not None:
            self.model.save(self.model_path)

    def load_model(self):
        if self.model_path is None or not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"No model file found at {self.model_path}.")
        return tf.keras.models.load_model(self.model_path, custom_objects={"Transformer": Transformer, "TransformerBlock": TransformerBlock})

    def clone_model_architecture(self):
        new_model = ModelManager(self.window_size, self.action_size, self.number_of_features, config=self.config)
        # Set weights for each output layer in the new model
        new_model.model.layers[-1].get_layer('transformer').set_weights(self.model.layers[-1].get_layer('transformer').get_weights())
        new_model.model.layers[-1].get_layer('transformer_1').set_weights(self.model.layers[-1].get_layer('transformer_1').get_weights())
        new_model.model.layers[-1].get_layer('transformer_2').set_weights(self.model.layers[-1].get_layer('transformer_2').get_weights())
        return new_model
