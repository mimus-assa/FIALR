import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tools.Preprocessing import Preprocessing
from envs.enviroment import *
from agent.agent import *
from configs import AgentConfig, PreprocessingConfig, DeepModelConfig
from models.model_manager import ModelManager

def main():
 #   print("Iniciando preprocesamiento...")
    preprocess_config = PreprocessingConfig()
    preprocess = Preprocessing(preprocess_config)
    train_0, train_prices = preprocess.process_data(preprocess_config.file)
  #  print("Preprocesamiento completado.")

   # print("Configurando el entorno y el agente...")
    config = DeepModelConfig()
    env_config = BtcMarketEnvConfig()
    agent_config = AgentConfig()
    env = BtcMarketEnv(train_0, train_prices, config=env_config)
    agent = DQNAgent(env, model_path='weights/Paguer_Network.h5', deep_model_config=config, config=agent_config)
   # print("Entorno y agente configurados.")

   # print("Iniciando entrenamiento del agente...")
    for i in range(1):
     #   print(f"Iniciando iteración de entrenamiento {i+1}...")
        agent.train_agent()
      #  print(f"Iteración de entrenamiento {i+1} completada.")

if __name__ == "__main__":
    main()
