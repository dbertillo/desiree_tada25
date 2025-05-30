import os
from configuration_manager import ConfigurationManager
from prompt_filler import PromptFiller
from model_invoker import ModelInvoker
import constants as c 
import json
from registries.registry import PROMPT_FORMATTERS
import openai
import registries.prompt_formatters_registry
from termcolor import colored
from claim_validator import ClaimValidator
from utils import pretty_print_candidate_claims, pretty_print_errors, pretty_print_raw_response, pretty_print_raw_response_specs
import re
from mapper import SpecificationsMatcher


class ExperimentManager:
    """
    Manages the configuration and paths for a specific experiment.
    This class is responsible for setting up the experiment environment,
    including directories and settings based on the experiment ID.
    It uses the ConfigurationManager to retrieve the necessary configurations.
    Attributes:
        experiment_id (str): The ID of the experiment.
        config_manager (ConfigurationManager): An instance of ConfigurationManager to retrieve configurations.
        project_path (str): The base path for the project.
        experiment_path (str): The path for the current experiment.
        settings (dict): The settings for the current experiment.
        description (str): A description of the experiment.
        domain (str): The domain of the experiment.
        topic (str): The topic of the experiment.
        pipeline_id (str): The ID of the pipeline used in the experiment.
        model (dict): A dictionary containing model settings.
        model_id (str): The ID of the model.
        model_type (str): The type of the model.
        model_size (str): The size of the model.
        model_version (str): The version of the model.
        shots (dict): A dictionary containing settings for shots.
        shots_mode (str): The mode for shots.
        shots_example (str): An example of shots.
        task_path (str): The path for the task.    
    """
    def __init__(self, experiment_id: str, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.experiment_id = experiment_id
        
        self.current_step = 0
        self.configure_settings()
        self.configure_paths()
        return

    def configure_paths(self):
        self.project_path = self.config_manager.get_project_path()
        self.data_path = self.config_manager.get_data_path()
        self.extracted_tables_path = self.config_manager.get_extracted_table_path()
        self.dataset_path = os.path.join(self.extracted_tables_path, self.dataset)
        
        
        self.prompt_templates_path = self.config_manager.get_prompt_templates_path()
        self.experiment_path = self.config_manager.get_current_experiment_path(self.experiment_id)
        self.experiment_settings = self.config_manager.get_experiment_config_by_id(self.experiment_id)
        return
    
    def configure_model_settings(self):
        self.model = self.settings.get(c.MODEL)
        self.model_id = self.model.get(c.MODEL_ID)
        self.model_type = self.model.get(c.MODEL_TYPE)
        self.model_size = self.model.get(c.MODEL_SIZE)
        self.model_version = self.model.get(c.MODEL_VERSION)
        self.model_id_second_step = self.settings.get(c.MODEL_ID_SECOND_STEP)
        self.model_type_second_step = self.settings.get(c.MODEL_TYPE_SECOND_STEP)
        self.model_id_third_ste = self.settings.get(c.MODEL_ID_THIRD_STEP)
        self.model_type_third_step = self.settings.get(c.MODEL_TYPE_THIRD_STEP)
        return

    def configure_shots_settings(self):
        self.shots = self.settings.get(c.SHOTS)
        self.shots_mode = self.shots.get(c.SHOTS_MODE)
        self.shots_examples = self.shots.get(c.SHOTS_EXAMPLES)
        return
    
    def configure_pipeline_settings(self):
        self.pipeline_id = self.settings.get(c.PIPELINE_ID)
        self.pipeline = self.config_manager.get_pipeline_config_by_id(self.pipeline_id)
        self.pipeline_steps = self.pipeline.get(c.PIPELINE_STEPS) #list of {'task': 'task_name', 'task_message_folder': 'message_folder_name'}
        self.pipeline_name = self.pipeline.get(c.PIPELINE_NAME)
        self.pipeline_id = self.pipeline.get(c.PIPELINE_ID)
        self.pipeline_description = self.pipeline.get(c.PIPELINE_DESCRIPTION)
        return
    
    def configure_LLMjudge_settings(self):
        self.ground_truth_path = self.config_manager.get_ground_truth_path(self.topic)
        self.experiments_path = self.config_manager.get_experiments_path()

        # Evaluate EXTRACTION EXPERIMENTS
        self.process_experiments = self.settings.get(c.PROCESS_EXPERIMENTS)
        self.experiments_to_evaluate_paths = []
        if self.process_experiments is not None:
            for experiment in self.process_experiments:
                self.experiments_to_evaluate_paths.append(os.path.join(self.experiments_path, experiment))
        self.current_experiment_to_evaluate_path = None
        self.ignore_experiments = self.settings.get(c.IGNORE_EXPERIMENTS)

        # Evaluate MATCHING EXPERIMENTS
        self.pairs_judge_experiment = self.settings.get(c.PAIRS_JUDGE_EXPERIMENT)
        if self.pairs_judge_experiment is not None:
            self.evaluated_specifications_experiments_paths = []
            self.evaluated_experiments = self.settings.get(c.EVALUATED_EXPERIMENTS)
            print("self.evaluated_experiments: ", self.evaluated_experiments)
            if self.evaluated_experiments is not None:
                for experiment in self.evaluated_experiments:
                    if 'cs' in experiment or 'med' in experiment:
                        current_evaluation_path = os.path.join(self.experiments_path, self.pairs_judge_experiment, experiment, 'evaluation_results.json')
                        if os.path.exists(current_evaluation_path):
                            self.evaluated_specifications_experiments_paths.append(current_evaluation_path)
                    # experiments_evaluated = os.listdir(os.path.join(self.experiments_path, self.pairs_judge_experiment, experiment))
                    # print("experiments_evaluated: ", experiments_evaluated)
                    #for evaluated_experiment in experiments_evaluated:
                    #    if 'cs' in evaluated_experiment or 'med' in evaluated_experiment:
                    #        self.evaluated_specifications_experiments_paths.append(os.path.join(self.experiments_path, experiment, self.pairs_judge_experiment, evaluated_experiment, 'evaluation_results.json'))
                    #        print(self.evaluated_specifications_experiments_paths)
                    #        exit()
        return
    
    def configure_settings(self):
        self.settings = self.config_manager.get_experiment_config_by_id(self.experiment_id)
        self.description = self.settings.get(c.DESCRIPTION)
        self.domain = self.settings.get(c.DOMAIN)
        self.topic = self.settings.get(c.TOPIC)

        self.dataset = self.set_dataset_path(self.settings.get(c.DATASET))
        self.configure_model_settings()
        self.configure_pipeline_settings()
        self.configure_shots_settings()
        self.configure_LLMjudge_settings()
        return
    
    def set_ground_truth_path(self):
        self.ground_truth_path = self.config_manager.get_ground_truth_path()
        return

    def set_dataset_path(self, dataset):
        """
        Set the dataset path based on the experiment settings.
        """
        if dataset.endswith(".json"):
            return self.settings.get(c.DATASET)
        else:
            # If the dataset is not a file, assume it's a folder and append ".json"
            return os.path.join(dataset + ".json")

    def get_pipeline_steps(self):
        """
        Retrieve the steps of the pipeline for the current experiment.
        """
        return self.pipeline_steps

    def get_message_folder_path(self, message_folder_name: str) -> str:
        """
        Build and return the full path for a message folder (template directory).
        Raises FileNotFoundError if the folder doesn't exist.
        """
        path = os.path.join(self.project_path, self.prompt_templates_path, message_folder_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Message folder {path} does not exist.")
        return path

    def get_dataset_path(self) -> str:
        """
        Build and return the full path for the dataset.
        """
        path = os.path.join(self.project_path, self.data_path, self.dataset)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset {path} does not exist.")
        return path

    def get_experiments_path(self) -> str:
        return self.experiments_path

    def setup_task_folders(self, task: str):
        self.task_path = os.path.join(self.experiment_path, task)
        print(self.task_path)
        if not os.path.exists(self.task_path):
            os.makedirs(self.task_path)
        self.prompts_dir = os.path.join(self.task_path, c.DIR_PROMPTS)
        if not os.path.exists(self.prompts_dir):
            os.makedirs(self.prompts_dir)
        self.filled_templates_dir = os.path.join(self.task_path, c.DIR_FILLED_TEMPLATES)
        if not os.path.exists(self.filled_templates_dir):
            os.makedirs(self.filled_templates_dir)
        self.raw_responses_dir = os.path.join(self.task_path, c.DIR_RAW_RESPONSES)
        if not os.path.exists(self.raw_responses_dir):
            os.makedirs(self.raw_responses_dir)
        self.metadata_dir = os.path.join(self.task_path, c.DIR_METADATA)
        if not os.path.exists(self.metadata_dir):
            os.makedirs(self.metadata_dir)
        self.validated_responses_dir = os.path.join(self.task_path, c.DIR_VALIDATED_RESPONSES)
        if not os.path.exists(self.validated_responses_dir):
            os.makedirs(self.validated_responses_dir)
        return  


    def save_filled_templates(self, filled_templates):
        for file_name, filled_template in filled_templates.items():
            file_path = os.path.join(self.filled_templates_dir, f"{file_name}.json")
            with open(file_path, 'w') as f:
                json.dump(filled_template, f, indent=4)
                f.close()
        return
    

    def save_raw_response(self, response, file_name):
        file_path = os.path.join(self.raw_responses_dir, f"{file_name}.txt")
        with open(file_path, 'w') as f:
            json.dump(response, f, indent=4)
            f.close()
        return

    def save_evaluation_raw_response_match(self, response, current_experiment_to_evaluate_path, table_id, index):
        current_evaluation = current_experiment_to_evaluate_path.split('/')[-1]
        current_evaluation_folder = os.path.join(self.experiment_path, current_evaluation)

        if not os.path.exists(current_evaluation_folder):
            os.makedirs(current_evaluation_folder)
        if not os.path.exists(os.path.join(current_evaluation_folder, table_id)):
            os.makedirs(os.path.join(current_evaluation_folder, table_id))

        file_name = f"{table_id}_{index}"

        if isinstance(response, str):
            file_path = os.path.join(current_evaluation_folder, table_id, f"{file_name}.txt")
            with open(file_path, 'w') as f:
                f.write(response)
                f.close()
        elif isinstance(response, dict):
            file_path = os.path.join(current_evaluation_folder, table_id, f"{file_name}.json")
            with open(file_path, 'w') as f:
                json.dump(response, f, indent=4)
                f.close()
        else:
            raise ValueError(f"Unsupported response type: {type(response)}")
        return

    def save_evaluation_raw_response(self, response, current_experiment_to_evaluate_path, file_name):
        current_evaluation = current_experiment_to_evaluate_path.split('/')[-1]
        current_evaluation_folder = os.path.join(self.experiment_path, current_evaluation)
        if not os.path.exists(current_evaluation_folder):
            os.makedirs(current_evaluation_folder)

        if isinstance(response, str):
            file_path = os.path.join(current_evaluation_folder, f"{file_name}.txt")
            with open(file_path, 'w') as f:
                f.write(response)
                # json.dump(response, f, indent=4)
                f.close()
        elif isinstance(response, dict):
            file_path = os.path.join(current_evaluation_folder, f"{file_name}.json")
            with open(file_path, 'w') as f:
                json.dump(response, f, indent=4)
                f.close()
        else:
            raise ValueError(f"Unsupported response type: {type(response)}")
        print(colored(f"\t\tEvaluation specs results saved to {file_path}", 'green'))
        return


    def save_validated_response(self, response, file_name):
        file_path = os.path.join(self.validated_responses_dir, f"{file_name}.txt")
        with open(file_path, 'w') as f:
            f.write(response)
            f.close()
        return


    def save_metadata(self, metadata, file_name):
        if not metadata:
            return
        file_path = os.path.join(self.metadata_dir, f"{file_name}.json")
        if isinstance(metadata, openai.types.chat.chat_completion.ChatCompletion):
            file_path = file_path.replace('.json', '.txt')
            with open(file_path, 'w') as f:
                f.write(str(metadata))  
                f.close()
            return
        if hasattr(metadata.get("body"), "read"):
                # Read from the stream and decode if necessary
                metadata["body"] = metadata["body"].read().decode('utf-8')
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            f.close()
        return


    def save_generated_prompt(self, generated_prompt, prompt_file_path):
        with open(prompt_file_path, 'w') as f:
            json.dump(generated_prompt, f, indent=4)
            f.close()
        return


    def old_generate_filled_templates(self, message_folder: str, validated_responses_dir: str = None):
        """
        Generate prompts for the given task.
        This method fills the templates with the corresponding data from the dataset.
        Then generate the prompts based on the inference model in usage, and save them.
        """
        message_folder_full_path = os.path.join(self.config_manager.get_prompt_templates_path(), message_folder)
        filled_templates = {}

        prompt_filler = PromptFiller(message_folder_full_path, self.dataset_path, self.domain, self.topic, self.shots_examples, self.ground_truth_path, self.current_experiment_to_evaluate_path)
        dataset_papers = prompt_filler.get_dataset_papers()
        
        # Check if filled templates already exist
        if os.path.exists(self.filled_templates_dir) and os.listdir(self.filled_templates_dir):
            for filename in os.listdir(self.filled_templates_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.filled_templates_dir, filename)
                    paper_table = os.path.splitext(filename)[0]
                    with open(file_path, 'r') as f:
                        filled_templates[paper_table] = json.load(f)
                        f.close()
        else:
            print(f"Generating new filled templates from {message_folder}")
            # prompt_filler = PromptFiller(message_folder_full_path, self.dataset_path, self.domain, self.topic, self.shots_examples, self.ground_truth_path, self.current_experiment_to_evaluate_path)
            if not validated_responses_dir:
                filled_templates = prompt_filler.fill_for_extraction_tasks()
            else:
                filled_templates = prompt_filler.fill_for_extraction_tasks(validated_responses_dir)
            self.save_filled_templates(filled_templates)

        print(colored(f"\tFilled templates saved or already present at {self.filled_templates_dir}", 'green'))
        return filled_templates

    def generate_filled_templates(self, message_folder: str, validated_responses_dir: str = None):
        """
        Generate prompts for the given task.
        This method fills the templates with the corresponding data from the dataset.
        Then generate the prompts based on the inference model in usage, and save them.
        """
        message_folder_full_path = os.path.join(self.config_manager.get_prompt_templates_path(), message_folder)
        filled_templates = {}

        prompt_filler = PromptFiller(
            message_folder_full_path, self.dataset_path, self.domain, self.topic,
            self.shots_examples, self.ground_truth_path, self.current_experiment_to_evaluate_path
        )
        dataset_papers = prompt_filler.get_dataset_papers()

        regenerate_needed = False
        existing_files = set()

        if os.path.exists(self.filled_templates_dir):
            existing_files = {os.path.splitext(f)[0] for f in os.listdir(self.filled_templates_dir) if f.endswith(".json")}
            # Check if any expected template is missing
            missing_files = set(dataset_papers) - existing_files
            if missing_files:
                regenerate_needed = True
        else:
            os.makedirs(self.filled_templates_dir, exist_ok=True)
            regenerate_needed = True

        if regenerate_needed:
            print(f"Generating new filled templates from {message_folder}")
            if not validated_responses_dir:
                filled_templates = prompt_filler.fill_for_extraction_tasks()
            else:
                filled_templates = prompt_filler.fill_for_extraction_tasks(validated_responses_dir)
            self.save_filled_templates(filled_templates)
        else:
            for paper_table in dataset_papers:
                file_path = os.path.join(self.filled_templates_dir, f"{paper_table}.json")
                with open(file_path, 'r') as f:
                    filled_templates[paper_table] = json.load(f)

        print(colored(f"\tFilled templates saved or already present at {self.filled_templates_dir}", 'green'))
        return filled_templates

    def generate_formatted_prompts(self, filled_templates):
        formmated_prompts = {}

        for paper_table, filled_template in filled_templates.items():
            # Check if the formatted prompt already exists
            prompt_file_path = os.path.join(self.prompts_dir, f"{paper_table}.json")

            if os.path.exists(prompt_file_path):
                # Load the existing formatted prompt
                with open(prompt_file_path, 'r') as f:
                    formmated_prompts[paper_table] = json.load(f)
            else:
                # Generate a new formatted prompt
                formmated_prompts[paper_table] = PROMPT_FORMATTERS[self.model_type](filled_template)
                self.save_generated_prompt(formmated_prompts[paper_table], prompt_file_path)
        
        print(colored(f"\tFormatted prompts saved or already present at {self.prompts_dir}", 'green'))
        return formmated_prompts
    

    def save_claims_json(self):
        validated_responses_files = os.listdir(self.validated_responses_dir)
        claims = {}

        claim_validator = ClaimValidator()

        for file_name in validated_responses_files:
            file_path = os.path.join(self.validated_responses_dir, file_name)
            with open(file_path, 'r') as f:
                claims[file_name.replace('.txt', '')] = claim_validator.build_claims(f.read())
                f.close()
        
        claims_file_path = os.path.join(self.experiment_path, c.FILENAME_CLAIMS)
        with open(claims_file_path, 'w') as f:
            json.dump(claims, f, indent=4)
            f.close()
        print(colored(f"\tClaims saved to {claims_file_path}", 'green'))


    def execute_extraction(self, task, message_folder, claims_format, previous_validated_response_folder=None):
    # Set up the necessary folders for the task.
        self.setup_task_folders(task)
        
        # Generate filled templates from the message folder.
        filled_templates = self.generate_filled_templates(message_folder, previous_validated_response_folder)
        # Generate formatted prompts from the filled_templates
        
        if self.current_step == 2:
            self.model_type = self.model_type_second_step
            self.model_id = self.model_id_second_step
        if self.current_step == 3:
            self.model_type = self.model_type_third_step
            self.model_id = self.model_id_third_ste
        
        formatted_prompts = self.generate_formatted_prompts(filled_templates)
        # Process each prompt one-by-one.
        for file_name, formatted_prompt in formatted_prompts.items():
            print(colored(f"\tProcessing {file_name}", 'blue'))
            # Check if validated response already exists for this file
            validated_file_path = os.path.join(self.validated_responses_dir, f"{file_name}.txt")
            if os.path.exists(validated_file_path):
                print(colored(f"\t\tValidated response already exists for {file_name} - Skipping extraction.", 'yellow'))
                continue

            raw_response_path = os.path.join(self.raw_responses_dir, f"{file_name}.txt")
            if os.path.exists(raw_response_path):
                print(colored(f"\t\tRaw response already exists for {file_name} - Skipping extraction.", 'yellow'))
                continue
            # if previous_validated_response_folder:
            #    model_invoker = ModelInvoker(self.model_type_second_step, self.model_id_second_step)
            #else:
            #    model_invoker = ModelInvoker(self.model_type, self.model_id)
            model_invoker = ModelInvoker(self.model_type, self.model_id)
            response, metadata = model_invoker.invoke(formatted_prompt)                
            pretty_print_raw_response(response)
            self.save_raw_response(response, file_name)
            self.save_metadata(metadata, file_name)

            # Validate the response using the provided claims format.
            claims_validator = ClaimValidator()
            valid, message, candidate_claims = claims_validator.validate_raw_response(response, claims_format)
            if valid:
                self.save_validated_response(candidate_claims, file_name)
                pretty_print_candidate_claims(candidate_claims)
            else:
                pretty_print_errors(message)
        return

    def llm_as_judge_evaluation(self, task, message_folder, current_experiment_to_evaluate_path):

        self.current_experiment_to_evaluate_path = current_experiment_to_evaluate_path
        print(colored(current_experiment_to_evaluate_path, 'blue'))
        filled_templates = self.generate_filled_templates(message_folder)
        # Generate formatted prompts from the filled_templates 
        formatted_prompts = self.generate_formatted_prompts(filled_templates)

        for file_name, formatted_prompt in formatted_prompts.items():
            print(colored(f"\tProcessing {file_name}", 'blue'))
            # Check if validated response already exists for this file
            validated_file_path = os.path.join(self.validated_responses_dir, f"{file_name}.txt")
            if os.path.exists(validated_file_path):
                print(colored(f"\t\Response already exists for {file_name} - Skipping extraction.", 'yellow'))
                continue
            model_invoker = ModelInvoker(self.model_type, self.model_id)
            response, metadata = model_invoker.invoke(formatted_prompt)                
            pretty_print_raw_response(response)
            self.save_evaluation_raw_response(response, current_experiment_to_evaluate_path, file_name)
            
            # self.save_metadata(metadata, file_name)

        return

    def llm_as_judge(self, task, message_folder):
        self.setup_task_folders(task)

        for current_experiment_to_evaluate_path in self.experiments_to_evaluate_paths:
            self.llm_as_judge_evaluation(task, message_folder, current_experiment_to_evaluate_path)
        exit()


    ################# -  JUDGE MATCHING PAIRS - #################


    def _load_claims(self, current_experiment_to_evaluate_path):
        extracted_claims_path = os.path.join(current_experiment_to_evaluate_path, c.FILENAME_CLAIMS)
        ground_truth_path = os.path.join(current_experiment_to_evaluate_path, c.FILENAME_CLAIMS)

        with open(extracted_claims_path, 'r') as f:
            extracted_claims = json.load(f)
            f.close()

        with open(self.ground_truth_path, 'r') as f:
            ground_truth_claims = json.load(f)
            f.close()

        return extracted_claims, ground_truth_claims
        
    def check_if_evaluation_pairs_already_present(self, current_experiment_to_evaluate_path, table_id):
        evaluation_results_path = current_experiment_to_evaluate_path.split('/')[-1]
        evaluation_results_path = os.path.join(self.experiment_path, evaluation_results_path, f"{table_id}_evaluation_results.json")

        if os.path.exists(evaluation_results_path):
            print(colored(f"\tEvaluation results already present for {table_id} - Skipping evaluation.", 'yellow'))
            return True
        return False

    def retrieve_evaluations_pairs_already_computed(self, current_experiment_to_evaluate_path, table_id):
        evaluation_results_path = current_experiment_to_evaluate_path.split('/')[-1]
        evaluation_results_path = os.path.join(self.experiment_path, evaluation_results_path, f"{table_id}_evaluation_results.json")

        with open(evaluation_results_path, 'r') as f:
            evaluation_results = json.load(f)
            f.close()
        return evaluation_results

    def safe_check_evaluation_based_on_measures_and_outcomes(self, extracted_claim, ground_truth_claim):
        m1 = str(extracted_claim.get('measures')).strip().lower()
        m2 = str(ground_truth_claim.get('measures')).strip().lower()
        o1 = str(extracted_claim.get('outcomes')).strip().lower()
        o2 = str(ground_truth_claim.get('outcomes')).strip().lower()

        if not m1 or not m2 or not o1 or not o2:
            return False
        
        if m1 == '[]' or m2 == '[]':
            return False

        if m1 and m2:
            if m1 == m2:
                if o1 and o2:
                    if o1 == o2:
                        return True
                    else:
                        return False
                else:
                    return False

    def llm_as_judge_pairs(self, task, message_folder):
        message_folder_full_path = os.path.join(self.config_manager.get_prompt_templates_path(), message_folder)

        self.setup_task_folders(task)
        # For each experiment to evaluate
        for current_experiment_to_evaluate_path in self.experiments_to_evaluate_paths:
            print("current_experiment_to_evaluate_path: ", current_experiment_to_evaluate_path)
            extracted_claims, ground_truth_claims = self._load_claims(current_experiment_to_evaluate_path)

            prompt_filler_judge = PromptFiller(message_folder_full_path, self.dataset_path, self.domain, self.topic, self.shots_examples, self.ground_truth_path, self.current_experiment_to_evaluate_path)
            evaluation_experiment = {}

            with open(self.dataset_path, 'r') as f:
                tables_in_datasets = json.load(f).keys()
                f.close()

            for table_id in list(extracted_claims.keys()):
                print("table_id: ", table_id)
                # if table_id == "PMC3544459_4" or table_id == "PMC3987090_4": #replaced with _5 
                #     continue
                if table_id not in tables_in_datasets:
                    continue
                if self.check_if_evaluation_pairs_already_present(current_experiment_to_evaluate_path, table_id):
                    evaluation_experiment[table_id] = self.retrieve_evaluations_pairs_already_computed(current_experiment_to_evaluate_path, table_id)
                    continue

                evaluation_table = {}
                evaluation_table['matches'] = []
                n_matches = 0
                actual_i = 0
                i = 0
                n_extracted_claims = len(extracted_claims[table_id])
                n_ground_truth_claims = len(ground_truth_claims[table_id])
                while i < len(extracted_claims[table_id]):
                    extracted_claim = extracted_claims[table_id][i]
                    actual_j = 0
                    j = 0
                    matched = False
                    while j < len(ground_truth_claims[table_id]):
                        ground_truth_claim = ground_truth_claims[table_id][j]
                        print(f"Extracted claim: {colored(extracted_claim, 'yellow')}")
                        print(f"Ground truth claim: {colored(ground_truth_claim, 'cyan')}")
                        if self.safe_check_evaluation_based_on_measures_and_outcomes(extracted_claim, ground_truth_claim):
                            classification = "yes"
                            evaluation_table['matches'].append({'Extracted_claim': extracted_claim, 'Ground_truth_claim' :ground_truth_claim, 'match': classification})
                            actual_j += 1
                            pretty_print_raw_response(classification)

                        else:
                            filled_template = prompt_filler_judge.fill_for_judging_claims_task(table_id, extracted_claim, ground_truth_claim)
                            formatted_prompt = PROMPT_FORMATTERS[self.model_type](filled_template)
                            
                            model_invoker = ModelInvoker(self.model_type, self.model_id)
                            response, metadata = model_invoker.invoke(formatted_prompt)
                            pretty_print_raw_response(response)
                            # self.save_evaluation_raw_response_match(response, current_experiment_to_evaluate_path, table_id, index=f"{actual_i}_{actual_j}")

                            match = re.search(r'<match>(.*?)</match>', response)
                            classification = match.group(1).strip() if match else None
                            actual_j += 1
                            evaluation_table['matches'].append({'Extracted_claim': extracted_claim, 'Ground_truth_claim' :ground_truth_claim, 'match': classification})

                        if classification == "yes":
                            # Remove both and break inner loop
                            extracted_claims[table_id].pop(i)
                            ground_truth_claims[table_id].pop(j)
                            matched = True
                            pretty_print_candidate_claims(classification)
                            n_matches += 1
                            break
                        else:
                            pretty_print_errors(classification)
                            j += 1
                    if not matched:
                        i += 1
                    actual_i += 1
                evaluation_table['total_extracted_claims'] = n_extracted_claims
                evaluation_table['total_ground_truth_claims'] = n_ground_truth_claims
                evaluation_table['number_of_matches'] = n_matches
                # evaluation_table.append({'total_extracted_claims': n_extracted_claims, 'total_ground_truth_claims': n_ground_truth_claims, 'number_of_matches': n_matches})
                evaluation_experiment[table_id] = evaluation_table
                self.save_evaluation_raw_response(evaluation_table, current_experiment_to_evaluate_path, file_name=f'{table_id}_evaluation_results')
            self.save_evaluation_raw_response(evaluation_experiment, current_experiment_to_evaluate_path, file_name='evaluation_results')
        

    def call_llm_judge_for_specifications(self, prompt_filler: PromptFiller, table_id, extracted_element, ground_truth_element):

        filled_template = prompt_filler.fill_for_judging_specs_task(table_id, extracted_element, ground_truth_element)
        formatted_prompt = PROMPT_FORMATTERS[self.model_type](filled_template)

        model_invoker = ModelInvoker(self.model_type, self.model_id)
        response, metadata = model_invoker.invoke(formatted_prompt)
        pretty_print_raw_response_specs(extracted_element, ground_truth_element, response)
        equivalent = re.search(r'<equivalent>(.*?)</equivalent>', response)
        classification = equivalent.group(1).strip() if equivalent else None

        if classification == "yes":
            pretty_print_candidate_claims(classification)
            return True
        else:
            pretty_print_errors(classification)
            return False
        
    def normalize_claim(self, claim):
        # Normalize the claim by removing leading and trailing whitespace
        normalized_claim = {}
        subject = claim.get('subject')
        measures = claim.get('measures')
        outcomes = claim.get('outcomes')

        if subject:
            normalized_claim['subject'] = {}
            for key, value in subject.items():
                normalized_claim['subject'][key.strip().lower()] = str(value).strip().lower()
        if measures:
            if isinstance(measures, str):
                normalized_claim['measures'] = measures.strip().lower()
            elif isinstance(measures, list):
                normalized_claim['measures'] = [measure.strip().lower() for measure in measures]

        if outcomes:
            if isinstance(outcomes, str):
                normalized_claim['outcomes'] = outcomes.strip().lower()
            elif isinstance(outcomes, list):
                normalized_claim['outcomes'] = [outcome.strip().lower() for outcome in outcomes]    
        
        return normalized_claim

    def _load_matched_claims(self):
        paired_claims = {}
        matches_string = 'matches'
        total_extracted_claims_string = 'total_extracted_claims'
        total_ground_truth_claims_string = 'total_ground_truth_claims'
        number_of_matches_string = 'number_of_matches'
        
        print("evaluated_specifications_experiments_paths: ", self.evaluated_specifications_experiments_paths)
        for experiment_evaluation_file_path in self.evaluated_specifications_experiments_paths:
            current_experiment = os.path.basename(os.path.dirname(experiment_evaluation_file_path))
            paired_claims[current_experiment] = {}
            # evaluation_path = os.path.join(experiment_evaluation_file_path, 'evaluation_results.json')
            with open(experiment_evaluation_file_path, 'r') as f:
                claims = json.load(f)
                f.close()

            for table_id, matching_results in claims.items():
                paired_claims[current_experiment][table_id] = {}
                paired_claims[current_experiment][table_id]['matches'] = []
                paired_claims[current_experiment][table_id]['extracted_matching_claims'] = []
                paired_claims[current_experiment][table_id]['ground_truth_matching_claims'] = []
                for attempt_match in matching_results[matches_string]:
                    if attempt_match['match'] == 'yes':
                        extracted_claim = attempt_match['Extracted_claim']
                        extracted_claim = self.normalize_claim(extracted_claim)
                        ground_truth_claim = attempt_match['Ground_truth_claim']
                        ground_truth_claim = self.normalize_claim(ground_truth_claim)
                        paired_claims[current_experiment][table_id]['matches'].append((extracted_claim, ground_truth_claim))
                        paired_claims[current_experiment][table_id]['extracted_matching_claims'].append(extracted_claim)
                        paired_claims[current_experiment][table_id]['ground_truth_matching_claims'].append(ground_truth_claim)

                paired_claims[current_experiment][table_id][total_extracted_claims_string] = matching_results[total_extracted_claims_string]
                paired_claims[current_experiment][table_id][total_ground_truth_claims_string] = matching_results[total_ground_truth_claims_string]
                paired_claims[current_experiment][table_id][number_of_matches_string] = matching_results[number_of_matches_string]


        return paired_claims

    def find_matching_elements_in_specs(self, matcher: SpecificationsMatcher, prompt_filler: PromptFiller, table_id, extracted_claim, ground_truth_claim, already_unmatched_keys, already_unmatched_values):
        extracted_claim_subject = extracted_claim['subject']
        ground_truth_claim_subject = ground_truth_claim['subject']
        
        for specification_key, specification_value in extracted_claim_subject.items():
            # For key
            if not matcher.is_already_mapped(specification_key) and specification_key not in already_unmatched_keys:
                matched = False
                for ground_truth_key, ground_truth_value in ground_truth_claim_subject.items():
                    if matcher.ground_truth_key_already_matched(ground_truth_key):
                        continue
                    if self.call_llm_judge_for_specifications(prompt_filler, table_id, specification_key, ground_truth_key):
                        matcher.add_llm_mapped_key(specification_key, ground_truth_key)
                        matched = True
                        break
                if not matched:
                    already_unmatched_keys.append(specification_key)

            # For value
            if not matcher.is_already_mapped(specification_value) and specification_value not in already_unmatched_values:
                matched = False
                for ground_truth_key, ground_truth_value in ground_truth_claim_subject.items():
                    if matcher.ground_truth_value_already_matched(ground_truth_value):
                        continue
                    if self.call_llm_judge_for_specifications(prompt_filler, table_id, specification_value, ground_truth_value):
                        matcher.add_llm_mapped_value(specification_value, ground_truth_value)
                        matched = True
                        break
                if not matched:
                    already_unmatched_values.append(specification_value)

    def llm_as_judge_specifications(self, task, message_folder):
        message_folder_full_path = os.path.join(self.config_manager.get_prompt_templates_path(), message_folder)
        self.setup_task_folders(task)

        # Evaluating the matching specifications.
        # 1. Read matched pairs!
        # 2. Check, for each matched pair, how many specifications are matched.
        # 3. Matching specifications should be handled by the same "SpecificationMatcher" class, 
        # which tracks the mapping (to reduce to the minimum the number of calls to the LLM).
        # In fact, it first try to match specifications based on keys and values based on strings equality
        # When it fails to find the equal match, we need to ask to the LLM to determine if they are a match or not.
        # Eventually, all the strings are going to be handled.

        paired_claims = self._load_matched_claims() # matching pairs extracted_claim and ground_truth_claim
        
        for experiment, evaluation in paired_claims.items():
            experiment_results = {}

            prompt_filler = PromptFiller(message_folder_full_path, self.dataset_path, self.domain, self.topic, self.shots_examples, self.ground_truth_path, self.current_experiment_to_evaluate_path)

            with open(self.dataset_path, 'r') as f:
                tables_in_datasets = json.load(f).keys()
                f.close()
            
            for table_id, matched_claims in evaluation.items():
                if table_id not in tables_in_datasets:
                    continue
            
                current_evaluation_results_json_path = os.path.join(self.experiment_path, experiment, f'{table_id}_evaluation_results.json')
                if os.path.exists(current_evaluation_results_json_path):
                    print(colored(f"\tEvaluation results already exist for {table_id} - Skipping evaluation.", 'yellow'))
                    with open(current_evaluation_results_json_path, 'r') as f:
                        experiment_results[table_id] = json.load(f)
                        f.close()
                    continue
                extracted_claims, ground_truth_claims = matched_claims['extracted_matching_claims'], matched_claims['ground_truth_matching_claims']
                matcher = SpecificationsMatcher(extracted_claims, ground_truth_claims)
                matched_claims = matched_claims['matches']

                already_unmatched_keys = []
                already_unmatched_values = []

                for extracted_claim, ground_truth_claim in matched_claims:
                    self.find_matching_elements_in_specs(matcher, prompt_filler, table_id, extracted_claim, ground_truth_claim, already_unmatched_keys, already_unmatched_values)

                precision, recall = matcher.evaluate_extracted_specifications()
                
                result = {
                    'matched_specifications': matcher.get_matched_specifications(),
                    'unmatched_specifications': matcher.get_unmatched_specifications(),
                    'unmatched_extracted_keys': matcher.get_unmatched_extracted_keys(),
                    'unmatched_extracted_values': matcher.get_unmatched_extracted_values(),
                    'unmatched_ground_truth_keys': matcher.get_unmatched_ground_truth_keys(),
                    'unmatched_ground_truth_values': matcher.get_unmatched_ground_truth_values(),
                    'mapped_keys': matcher.get_mapped_keys(),
                    'mapped_values': matcher.get_mapped_values(),
                    'precision': precision,
                    'recall': recall
                }
                file_name = f"{table_id}_evaluation_results"
                self.save_evaluation_raw_response(result, os.path.join(self.experiment_path, experiment), file_name=f'{table_id}_evaluation_results')
                experiment_results[table_id] = result
            self.save_evaluation_raw_response(experiment_results, os.path.join(self.experiment_path, experiment), file_name='evaluation_results')

    ################# - /JUDGE MATCHING PAIRS - #################

    def execute_task(self, task, message_folder, claims_format, previous_validated_response_folder):
        self.current_step = 1

        if 'step_2' in task or 'step2' in task:
            self.current_step = 2
        if 'step_3' in task or 'step3' in task:
            self.current_step = 3
    
        if task == 'direct_extraction':
            self.execute_extraction(task, message_folder, claims_format)
            self.save_claims_json()
            return 
        
        if task == 'boostrap_extraction_step_1':
            return self.execute_extraction(task, message_folder, claims_format)
        
        if task == 'boostrap_extraction_step_2':
            self.execute_extraction(task, message_folder, claims_format, previous_validated_response_folder)
            self.save_claims_json()
            return

        if task == 'algcand_step1':
            self.execute_extraction(task, message_folder, claims_format)
            return
        
        if task == 'alg_nocand_step1':
            self.execute_extraction(task, message_folder, claims_format)
            return
    
        if task == 'algcand_step2':
            self.execute_extraction(task, message_folder, claims_format, previous_validated_response_folder)
        
        if task == 'algcand_step3':
            self.execute_extraction(task, message_folder, claims_format, previous_validated_response_folder)
            self.save_claims_json()
            return
    
        # if task == 'llm_as_judge':
        #    self.llm_as_judge(task, message_folder)
        
        if task == 'llm_as_judge_pairs':
            self.llm_as_judge_pairs(task, message_folder)

        if task == 'llm_as_judge_specs':
            self.llm_as_judge_specifications(task, message_folder)