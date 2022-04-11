import yaml
import os

def load_config(config_path=os.path.join(os.path.dirname(__file__), 'config.yaml')):
    try:
        with open(config_path, "r") as stream:
            try:
                return(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

    except:
        class SetupFileIsMissing(Exception):pass
        raise SetupFileIsMissing(f'Please make a {config_path} file')