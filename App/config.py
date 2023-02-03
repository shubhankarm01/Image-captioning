# pip install strictyaml

from pathlib import Path
import App

from strictyaml import YAML, load

from pydantic import BaseModel


root_path = Path(App.__file__).resolve().parent
config_file_path = root_path/"config.yml"
model_path = root_path/"Data/Trained_model"
feat_model_path = root_path/"Data/feat_model.h5"
embedded_mat_path = root_path/"Data/embeded_mat.npy"
word_index_path = root_path/"Data/word_index.json"


def find_config_file() -> Path:
    if config_file_path.is_file():
        return config_file_path

    raise Exception("Config file not found at {}".format(root_path))

def fetch_configuration(cfg_path: Path = None) -> YAML:
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, 'r') as config_file:
            config_parse = load(config_file.read())
            return config_parse.data

# class ModelConfig(BaseModel):
#     embeded_dim : int
#     units : int
#     len_sent : int

# class Config(BaseModel):
#     model_config: ModelConfig

# def validate_config(config_parse: YAML = None) -> Config:
#     if not config_parse:
#         config_parse = fetch_configuration()

#     config_ = Config(model_config = ModelConfig(** config_parse.data))

#     return config_

# config = validate_config()

config = fetch_configuration()