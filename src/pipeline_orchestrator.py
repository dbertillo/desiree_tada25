
from experiment_manager import ExperimentManager
from configuration_manager import ConfigurationManager
import constants as c
from termcolor import colored

class PipelineOrchestrator:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.experiments_ids_to_run = self.config_manager.get_experiments_ids_to_run()

    
    def run_experiment():
        return
        
        # Save metdata about the experiment.
        # For each task in "steps" do the following things:
            # 1. Create the folder
            # 2. Generate the prompts and save them in subfolder of the experiment called "prompts"
            # 3. Generate the responses and save them in subfolder of the experiment called "raw_responses"
            # 4. Save also the metadata of the responsed in a subfolder of the experiment called "metadata"
            # 5. Validate the responses and save them in a subfolder of the experiment called "validated_responses"

    def run_experiment_by_id(self, experiment_id):
        current_experiment = ExperimentManager(experiment_id, self.config_manager)
        last_validated_responses_folder = None
        for task in current_experiment.get_pipeline_steps():
            current_task = task.get(c.TASK)
            current_prompt_template_folder = task.get(c.MESSAGE_FOLDER)
            current_claims_format = task.get(c.CLAIMS_FORMAT)
            print(f"\n#### Running task {current_task} for {experiment_id} ####")
            execute_task = current_experiment.execute_task(current_task, current_prompt_template_folder, current_claims_format, last_validated_responses_folder)
            last_validated_responses_folder = self.config_manager.get_validated_responses_folder(experiment_id, current_task)

    def run_experiments(self, experiments_ids):
        for experiment_id in experiments_ids:
            self.run_experiment_by_id(experiment_id)


if __name__ == "__main__":
    config_manager = ConfigurationManager()
    experiments_ids_to_run = config_manager.get_experiments_ids_to_run()
    print("Experiments to run:", experiments_ids_to_run)
    po = PipelineOrchestrator()
    po.run_experiments(experiments_ids_to_run)
        
        

