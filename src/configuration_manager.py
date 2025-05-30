import os
import yaml
from config_schema import PipelinesConfig, ExperimentsConfig
import constants as c

class ConfigurationManager:
    """
    Loads and manages configuration settings from three YAML files:
      - Pipelines configuration (pipelines.yaml)
      - Experiments configuration (experiments.yaml)
    
    This manager supports nested retrieval (using dot-notation) and
    applies environment variable overrides and expansion.
    """
    def __init__(self, 
                 ):

        self.project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # self.core_config_file = core_config_file or os.path.join(base_dir, "config", "config.yaml")
        self.data_path = os.path.join(self.project_path, c.DIR_DATA)
        self.examples_path = os.path.join(self.data_path, c.DIR_EXAMPLES)
        self.extracted_table_path = os.path.join(self.data_path, c.DIR_EXTRACTED_TABLES)
        self.prompt_templates_path = os.path.join(self.data_path, c.DIR_PROMPT_TEMPLATES)
        self.pipelines_configs_path = os.path.join(self.project_path, c.DIR_CONFIG, c.FILENAME_PIPELINE_CONFIG)
        self.experiments_configs_path = os.path.join(self.project_path, c.DIR_CONFIG, c.FILENAME_EXPERIMENTS_CONFIG)
        self.experiments_to_run_configs_path = os.path.join(self.project_path, c.DIR_CONFIG, c.FILENMES_EXPERIMENTS_TO_RUN)
        
        self.pipelines_configs = self._load_yaml(self.pipelines_configs_path)
        self.experiments_configs = self._load_yaml(self.experiments_configs_path)
        self.experiments_ids_to_run_configs = self._load_yaml(self.experiments_to_run_configs_path)

        self.experiments_path = os.path.join(self.project_path, c.DIR_EXPERIMENTS)
    def _load_yaml(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def load_and_validate_pipelines(self):
        pipelines_raw = self._load_yaml(self.pipelines_file)
        try:
            pipelines_config = PipelinesConfig(**pipelines_raw)
            return pipelines_config
        except Exception as e:
            raise ValueError(f"Invalid pipelines configuration: {e}")

    def load_and_validate_experiments(self):
        experiments_raw = self._load_yaml(self.experiments_file)
        try:
            experiments_config = ExperimentsConfig(**experiments_raw)
            return experiments_config
        except Exception as e:
            raise ValueError(f"Invalid experiments configuration: {e}")

    def _expand_env_vars(self, obj):
        """
        Recursively expand environment variables in strings within the given object.
        """
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        elif isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        else:
            return obj

    def get_project_path(self):
        return self.project_path
    
    def get_data_path(self):
        return self.data_path
    
    def get_examples_path(self):
        return self.examples_path
    
    def get_extracted_table_path(self):
        return self.extracted_table_path
    
    def get_prompt_templates_path(self):
        return self.prompt_templates_path

    def get_validated_responses_folder(self, experiment_id, current_task):
        """
        Retrieve the path for validated responses folder.
        """
        return os.path.join(self.get_current_experiment_path(experiment_id), current_task, c.DIR_VALIDATED_RESPONSES)

    def get_current_experiment_path(self, experiment_id: str):
        """
        Retrieve the current experiment path.
        """
        return os.path.join(self.project_path, c.DIR_EXPERIMENTS, experiment_id)

    def get_pipeline_config_by_id(self, id=None, default=None):
        """
        Retrieve a value from the pipelines configuration.
        If no key is provided, returns all pipelines configuration.
        """
        if id in self.pipelines_configs:
            return self.pipelines_configs[id]
        else:
            raise ValueError(f"Pipeline id: {id} not found.")

    def get_experiments_config(self, key=None, default=None):
        """
        Retrieve a value from the experiments configuration.
        If no key is provided, returns the entire experiments configuration.
        """
        experiments = self.experiments_configs.get("experiments", {})
        if key:
            return self._get_nested(experiments, key, default)
        return experiments

    def get_experiment_config_by_id(self, experiment_id):
        """
        Retrieve a specific experiment's configuration by its ID.
        The experiment_id can be either the YAML key or the "id" field in the experiment.
        """
        experiments = self.experiments_configs
        # First, try to get by direct key.
        if experiment_id in experiments:
            return experiments[experiment_id]
        else:
            raise ValueError(f"Experiment id: {experiment_id} not found.")

    def get_experiments_ids_to_run(self):
        """
        Retrieve a value from the experiments configuration.
        If no key is provided, returns the entire experiments configuration.
        """
        return self.experiments_ids_to_run_configs
    

    def _get_nested(self, data_dict, key, default=None):
        """
        Retrieve a nested value using a dot-separated key.
        """
        keys = key.split(".")
        value = data_dict
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def get_ground_truth_path(self, topic):
        if topic == c.TOPIC_TXT2SQL:
            return os.path.join(self.experiments_path, "gt_cs_txt2sql", "claims.json")
        elif topic == c.TOPIC_PANCREATIC_CANCER:
            return os.path.join(self.experiments_path, "gt_med_pancreatic_cancer", "claims.json")
        elif topic == c.TOPIC_ER:
            return os.path.join(self.experiments_path, "gt_cs_er", "claims.json")
        elif topic == c.TOPIC_HIV:
            return os.path.join(self.experiments_path, "gt_med_hiv", "claims.json")
    
    def get_experiments_path(self):
        return self.experiments_path

# Example usage:
if __name__ == "__main__":
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    config_manager = ConfigurationManager()
    # print("Project Path:", config_manager.get_core_config("project_path"))
    experiments_to_run = config_manager.get_experiments_ids_to_run()
    print("Experiments to run: ", experiments_to_run)
    print("Experiment config: ", config_manager.get_experiment_config_by_id(experiments_to_run[0]))
    exit()
    print("Pipelines:", config_manager.get_pipelines_config())
    print("Experiments:", config_manager.get_experiments_config())
    experiment = config_manager.get_experiment_by_id("cs_1_0_shot")
    print("Experiment cs_1_0_shot:", experiment)
    pipeline_id = config_manager.get_pipeline_by_id(experiment.get("pipeline_id"))
    print(pipeline_id)
    steps = pipeline_id.get("steps")
    print(steps)
