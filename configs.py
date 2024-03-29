# Clase base para la configuración, define parámetros comunes para todas las configuraciones
class BaseConfig:
    def __init__(self):
        self.batch_size = 128  # Tamaño del lote para el entrenamiento
        self.window_size = 32  # Tamaño de la ventana para el muestreo de datos
        self.starting_step = 0  # Paso de inicio para el entrenamiento
        self.max_steps = 1000  # Máximo número de pasos para el entrenamiento
        #idea: podriamos hacer esto referente a una fecha(date) en vez de un numero de pasos

# Configuración específica para el modelo DeepModel
class DeepModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()  # Hereda la configuración de la clase base

        # Configuraciones específicas del Transformer
        self.attention_heads = 16
        self.attention_key_dim = 128
        self.attention_value_dim = 128
        self.attention_dropout = 0.3
        self.ffn_units = 64
        self.initial_learning_rate = 0.1
  
        self.clipnorm = 1.0
        self.input_dim=20
        
# Configuración específica para el ambiente de mercado de Bitcoin
class BtcMarketEnvConfig(BaseConfig):
    def __init__(self):
        super().__init__()  # Hereda la configuración de la clase base
         # Cantidad inicial de dólares para el trading

# Configuración específica para el agente
class AgentConfig(BaseConfig):
    def __init__(self):
        super().__init__()  # Hereda la configuración de la clase base
        #for the model 
        self.memory_size = 256 # Tamaño de la memoria del agente
        self.episodes = 100  # Número de episodios para el entrenamiento
        self.epsilon_start = 0.1  # Valor inicial de epsilon para la exploración
        self.epsilon_end = 0.0001  # Valor final de epsilon para la exploración
        self.epsilon_decay = 0.98  # Tasa de decaimiento para epsilon
        
        self.target_update_frequency = 250  # Frecuencia de actualización para el modelo objetivo

        #for the porfolio
        self.risk_factor = 0.008  # Factor de riesgo para la gestión de riesgos
        self.stop_price=0.5  # Precio de stop para la gestión de riesgos
        self.initial_dollars = 10000.0 

# Configuración para la preprocesamiento de datos
class PreprocessingConfig(BaseConfig):
    def __init__(self):
        super().__init__()  # Hereda la configuración de la clase base
        self.start_timestamp = 1565408700  # Timestamp de inicio para el muestreo de datos
        self.end_timestamp = 1672550802  # Timestamp de fin para el muestreo de datos
        self.val_ptc = 0.2  # Porcentaje de datos para validación
        self.test_ptc = 0.08  # Porcentaje de datos para pruebas
        self.cols = ['o', 'h', 'l', 'c',  'log_ret_oc', 'log_ret_lh', 'bb_upper',
                    'bb_middle', 'bb_lower', 'rsi', 'macd', 'macd_signal', 'macd_hist'] # Columnas a utilizar en los datos
        self.file = "data/training/5m_train_qlearning.csv"  # Archivo de datos a utilizar
