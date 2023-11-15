from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import MultiHeadAttention, Dropout, LayerNormalization
from tensorflow.keras.initializers import HeUniform ,GlorotUniform
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2


class TransformerConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class TransformerBlock(Model):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.config = config
        self.att = MultiHeadAttention(
            num_heads=config.attention_heads,
            key_dim=config.attention_key_dim,
            value_dim=config.attention_value_dim,
            dropout=config.attention_dropout,
            use_bias=True,
            kernel_initializer=GlorotUniform()
        )
        self.ffn = Sequential(
            [
                Dense(config.ffn_units, kernel_initializer=GlorotUniform()),
                LeakyReLU(),
                BatchNormalization(),
                Dense(config.input_dim, kernel_initializer=GlorotUniform()),
                LeakyReLU(),
                BatchNormalization()
            ]
        )

        self.dropout1 = Dropout(config.attention_dropout)
        self.dropout2 = Dropout(config.attention_dropout)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'config': self.config.__dict__  # convert config object to dictionary
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Create a TransformerConfig object from the 'config' dictionary
        config_obj = TransformerConfig(**config['config'])
        return cls(config_obj)


    
class Transformer(Model):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.transformer_block = TransformerBlock(config)
        self.flatten = Flatten()

        self.dense_actions = self.create_dense_layers([256, 64, 32], "actions")
        self.dense_stop_loss = self.create_dense_layers([32, 16, 8], "stop_loss")
        self.dense_ratio = self.create_dense_layers([32, 16, 8], "ratio")

        self.output_actions = Dense(3, activation="softmax", kernel_initializer=GlorotUniform(), name='transformer')
        self.output_stop_loss = Dense(4, activation="softmax", kernel_initializer=GlorotUniform(), name='transformer_1')
        self.output_ratio = Dense(4, activation="softmax", kernel_initializer=GlorotUniform(), name='transformer_2')

    def create_dense_layers(self, units, name):
        layers = []
        for unit in units:
            layers.extend([
                Dense(unit, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.005), name=f'dense_{unit}_{name}'),
                LeakyReLU(name=f'leaky_relu_{unit}_{name}'),
                BatchNormalization(name=f'batch_norm_{unit}_{name}'),
            ])
        return layers

    def call(self, inputs):
        x = self.transformer_block(inputs)
        x = self.flatten(x)

        x_actions = self.apply_dense_layers(x, self.dense_actions)
        actions = self.output_actions(x_actions)

        x_stop_loss = self.apply_dense_layers(x, self.dense_stop_loss)
        stop_loss = self.output_stop_loss(x_stop_loss)

        x_ratio = self.apply_dense_layers(x, self.dense_ratio)
        ratio = self.output_ratio(x_ratio)

        return [actions, stop_loss, ratio]

    def apply_dense_layers(self, x, layers):
        for layer in layers:
            x = layer(x)
        return x

    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'config': self.config.__dict__  # convert config object to dictionary
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Create a TransformerConfig object from the 'config' dictionary
        config_obj = TransformerConfig(**config['config'])
        return cls(config_obj)

