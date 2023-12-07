# Clase base para la configuración, define parámetros comunes para todas las configuraciones
class BaseConfig:
    def __init__(self):
        self.batch_size = 1  # Tamaño del lote para el entrenamiento
        self.window_size = 50  # Tamaño de la ventana para el muestreo de datos
        self.starting_step = 0  # Paso de inicio para el entrenamiento
        self.max_steps = 100+ self.window_size # Máximo número de pasos para el entrenamiento
        #idea: podriamos hacer esto referente a una fecha(date) en vez de un numero de pasos

# Configuración específica para el modelo DeepModel
class DeepModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()  # Hereda la configuración de la clase base

        # Configuraciones específicas del Transformer
        self.attention_heads = 4
        self.attention_key_dim = 256
        self.attention_value_dim = 256
        self.attention_dropout = 0.05
        self.ffn_units = 128

        self.initial_learning_rate = 0.01
  
        self.clipnorm = 1.0
        self.input_dim=14
        
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
        self.memory_size = 55 # Tamaño de la memoria del agente
        self.episodes = 1500  # Número de episodios para el entrenamiento
        self.epsilon_start = 0.05  # Valor inicial de epsilon para la exploración
        self.epsilon_end = 0.001  # Valor final de epsilon para la exploración
        self.epsilon_decay = 0.96  # Tasa de decaimiento para epsilon
        
        self.target_update_frequency = 5  # Frecuencia de actualización para el modelo objetivo

        #for the porfolio
        self.risk_factor = 0.008  # Factor de riesgo para la gestión de riesgos
        self.stop_price=0.5  # Precio de stop para la gestión de riesgos
        self.initial_dollars = 10000.0 

# Configuración para la preprocesamiento de datos
class PreprocessingConfig(BaseConfig):
    def __init__(self):
        super().__init__()  # Hereda la configuración de la clase base
        self.start_timestamp = 1577944800+5*60  # Timestamp de inicio para el muestreo de datos
        self.end_timestamp = 1672550802  # Timestamp de fin para el muestreo de datos
        self.val_ptc = 0.2  # Porcentaje de datos para validación
        self.test_ptc = 0.08  # Porcentaje de datos para pruebas
        self.cols = ['o', 'h', 'l', 'c',  'log_ret_oc', 'log_ret_lh', 'bb_upper',
                    'bb_middle', 'bb_lower', 'rsi', 'macd', 'macd_signal', 'macd_hist'] # Columnas a utilizar en los datos
        self.file = "data/training/5m_train_qlearning.csv"  # Archivo de datos a utilizar
