from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, MultiHeadAttention, Dropout, LayerNormalization, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import Sequential
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
        # Asegúrate de que la última capa FFN tenga la misma cantidad de unidades que el input_dim
        self.ffn = Sequential([
            Dense(config.ffn_units, kernel_initializer=GlorotUniform()),
            LeakyReLU(),
            BatchNormalization(),
            Dense(config.input_dim, kernel_initializer=GlorotUniform()),  # Coincide con las dimensiones de entrada
            LeakyReLU(),
            BatchNormalization()
        ])

        self.dropout1 = Dropout(config.attention_dropout)
        self.dropout2 = Dropout(config.attention_dropout)
        # Instancia LayerNormalization aquí
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)  # Usa la instancia norm1 aquí
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)  # Usa la instancia norm2 aquí

class Transformer(Model):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.transformer_block = TransformerBlock(config)
        self.flatten = Flatten()

        self.dense_actions = self.create_dense_layers([1024, 256, 128], "actions")
        self.output_actions = Dense(4, activation="softmax", kernel_initializer=GlorotUniform(), name='transformer')

    def create_dense_layers(self, units, name):
        layers = []
        for unit in units:
            layers.extend([
                Dense(unit, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.01), name=f'dense_{unit}_{name}'),
                LeakyReLU(name=f'leaky_relu_{unit}_{name}'),
                BatchNormalization(name=f'batch_norm_{unit}_{name}'),
            ])
        return layers

    def call(self, inputs):
        x = self.transformer_block(inputs)
        x = self.flatten(x)
        x_actions = self.apply_dense_layers(x, self.dense_actions)
        actions = self.output_actions(x_actions)
        return actions

    def apply_dense_layers(self, x, layers):
        for layer in layers:
            x = layer(x)
        return x
