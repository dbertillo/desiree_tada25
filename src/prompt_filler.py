import constants as c
from configuration_manager import ConfigurationManager
import os
import re
import json
from termcolor import colored

class PromptFiller:
    '''
    PromptFiller class is responsible for filling the message templates with the content from data.
    It takes a message folder, a dataset path, and functions to handle placeholders, extraction, and replacement.
    The main method is `fill`, which processes the dataset and fills the templates, and returns a dictionary with roles and content (with placeholders replaced).
    '''
    def __init__(
            self,
            message_folder_path, 
            dataset_path, 
            domain, 
            topic,
            examples,
            ground_truth_path=None,
            evaluate_experiment_path = None
            ):
        
        self.message_folder_path = message_folder_path
        self.dataset_path = dataset_path
        print("self.dataset_path", self.dataset_path)
        self.examples = examples

        if self.validate_domain(domain):
            self.domain = domain
        if self.validate_topic(topic):
            self.topic = topic

        self.ground_truth_path = ground_truth_path
        
        self.evaluate_experiment_path = evaluate_experiment_path

        self.extracted_claim = None
        self.ground_truth_claim = None
        self.extracted_element = None
        self.ground_truth_element = None

    def validate_domain(self, domain):
        if domain not in [c.DOMAIN_CS, c.DOMAIN_MED]:
            raise ValueError(f"Invalid domain: {domain}. Expected one of {c.DOMAIN_CS}, {c.DOMAIN_MED}.")
        return True

    def validate_topic(self, topic):
        if topic not in [c.TOPIC_TXT2SQL, c.TOPIC_PANCREATIC_CANCER, c.TOPIC_ER, c.TOPIC_HIV]:
            raise ValueError(f"Invalid topic: {topic}. Expected one of {c.TOPIC_TXT2SQL}, {c.TOPIC_PANCREATIC_CANCER}, {c.TOPIC_ER}, {c.TOPIC_HIV}.")
        return True
    
    def get_dataset_papers(self):
        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)
        return dataset.keys()


    def _load_dataset(self):
        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)
        return dataset
    
    def _load_templates(self):
        # Load all message templates from the folder
        # and return them as a list of dictionaries.
        # Each dictionary contains 'role' and 'content' keys.
        template_files = sorted(os.listdir(self.message_folder_path))
        templates = []
        for file in template_files:
            with open(os.path.join(self.message_folder_path, file), 'r') as f:
                content = f.read()
                role = os.path.splitext(file)[0].split("_")[1].replace('.txt', '')
                templates.append({"role": role, "content": content})
        return templates

    def _extract_placeholders(self, text):
        return re.findall(r"#placeholder\{[^}]*\}", text)

    def _replace_placeholder(self, text: str, placeholder: str, value: str):
        if isinstance(value, list):
            value = ' '.join(value)
        if isinstance(value, dict):
            value = str(value)
        return text.replace(placeholder, value)

    def _get_examples(self):
        # Load the examples from the examples folder
        # and return them as a list of dictionaries.
        config_manager = ConfigurationManager()
        examples_path = os.path.join(config_manager.get_examples_path())

        examples_content = []

        for example in self.examples:
            if '.txt' not in example:
                example = example + '.txt'
            example_path = os.path.join(examples_path, example)
            with open(example_path, 'r') as f:
                content = f.read()
                examples_content.append(content)
        return examples_content


    def _load_claims(self, preivous_validate_response_folder, table_id):
        if preivous_validate_response_folder is None:
            claims = []
        else:
            claims_path = os.path.join(preivous_validate_response_folder, table_id + '.txt')
            if not os.path.exists(claims_path):
                return ''
            with open(claims_path, 'r') as f:
                claims = f.read()
        return claims

    def _load_ground_truth(self, ground_truth_path, table_id):
        if ground_truth_path is None:
            ground_truth_claims = []
        else:
            with open(ground_truth_path, 'r') as f:
                ground_truth_claims = json.load(f)
        return str(ground_truth_claims[table_id])
    
    def _load_claims_from_experiment(self, evaluate_experiment_path, table_id):
        if evaluate_experiment_path is None:
            claims = []
        else:
            claims_path = os.path.join(evaluate_experiment_path, "claims.json")
            with open(claims_path, 'r') as f:
                claims = json.load(f)
        claims = claims.get(table_id, [])
        return str(claims)

    def _get_value_for_placeholder(self, placeholder: str, table_id: str, table: dict, previous_validate_response_folder=None):

        claims = None
        ground_truth_claims = None
        if previous_validate_response_folder is not None and self.evaluate_experiment_path is not None:
            raise ValueError("You cannot provide both previous_validate_response_folder and evaluate_experiment_path at the same time.")

        if previous_validate_response_folder is not None:
            claims = self._load_claims(previous_validate_response_folder, table_id)

        if self.evaluate_experiment_path is not None:
            claims = self._load_claims_from_experiment(self.evaluate_experiment_path, table_id)

        if self.ground_truth_path is not None:
            ground_truth_claims = self._load_ground_truth(self.ground_truth_path, table_id)
    
        MAP_PLACEHOLDER_TO_EXTRACTEDTABLES = {
            c.PLACEHOLDER_CAPTION: table.get(c.DATASOURCE_CAPTION),
            c.PLACEHOLDER_CITATIONS: table.get(c.DATASOURCE_CITATIONS, "None"),
            c.PLACEHOLDER_TABLE: table.get(c.DATASOURCE_HTML_TABLE),
            c.PLACEHOLDER_TABLE_HEAD: table.get(c.DATASOURCE_TABLE_HEAD),
            c.PLACEHOLDER_DOMAIN: self.domain,
            c.PLACEHOLDER_TOPIC: self.topic,
            c.PLACEHOLDER_FOOTNOTES: table.get(c.DATASOURCE_FOOTNOTES, "None"),
            c.PLACEHOLDER_EXAMPLES: self._get_examples(),
            c.PLACEHOLDER_CLAIMS: claims,
            c.PLACEHOLDER_GROUND_TRUTH: ground_truth_claims,
            c.PLACEHOLDER_CANDIDATE_PAIRS: str(table.get(c.DATASOURCE_CANDIDATES_PAIRS, "None")),
            c.PLACEHOLDER_EXTRACTED_CLAIM: self.extracted_claim,
            c.PLACEHOLDER_GROUND_TRUTH_CLAIM: self.ground_truth_claim,
            c.PLACEHOLDER_EXTRACTED_ELEMENT: self.extracted_element,
            c.PLACEHOLDER_GROUND_TRUTH_ELEMENT: self.ground_truth_element
        }

        return MAP_PLACEHOLDER_TO_EXTRACTEDTABLES[placeholder]


    def fill_for_extraction_tasks(self, preivous_validate_response_folder=None):
        dataset = self._load_dataset()
        filled_templates = {}

        for table_id, table in dataset.items():
            # if preivous_validate_response_folder or self.ground_truth_path is not None:
            #    table_id
            #    table_id = table_id.replace(table_id.split('_')[1], str(int(table_id.split('_')[1]) + 1))
            current_message_templates = self._load_templates()
            for message in current_message_templates:
                placeholders = self._extract_placeholders(message['content'])
                if placeholders:
                    for placeholder in placeholders:
                        value = self._get_value_for_placeholder(placeholder, table_id, table, preivous_validate_response_folder)
                        message['content'] = self._replace_placeholder(message['content'], placeholder, value)
                else:
                    message['content'] = message['content']
            # if not preivous_validate_response_folder:
            #    table_id = table_id.replace(table_id.split('_')[1], str(int(table_id.split('_')[1]) + 1))
            filled_templates[table_id] = current_message_templates
             
        return filled_templates        

    def fill_for_judging_claims_task(self, table_id, extracted_claim, ground_truth_claim):
        dataset = self._load_dataset()
        table = dataset.get(table_id)

        # UPDATE EXTRACTED CLAIM AND GROUND TRUTH CLAIM
        self.extracted_claim = extracted_claim
        self.ground_truth_claim = ground_truth_claim

        current_message_templates = self._load_templates()
        for message in current_message_templates:
                placeholders = self._extract_placeholders(message['content'])
                if placeholders:
                    for placeholder in placeholders:
                        value = self._get_value_for_placeholder(placeholder, table_id, table, None)
                        message['content'] = self._replace_placeholder(message['content'], placeholder, value)
                else:
                    message['content'] = message['content']
        
        return current_message_templates
    

    def fill_for_judging_specs_task(self, table_id, extracted_element, ground_truth_element):
        dataset = self._load_dataset()
        table = dataset.get(table_id)

        self.extracted_element = extracted_element
        self.ground_truth_element = ground_truth_element

        current_message_templates = self._load_templates()
        for message in current_message_templates:
                placeholders = self._extract_placeholders(message['content'])
                if placeholders:
                    for placeholder in placeholders:
                        value = self._get_value_for_placeholder(placeholder, table_id, table, None)
                        message['content'] = self._replace_placeholder(message['content'], placeholder, value)
                else:
                    message['content'] = message['content']
        
        return current_message_templates


if __name__ == "__main__":
    # Example usage
    message_folder_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/data/prompt_templates/direct_extraction_v0/"
    dataset_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/data/extracted_tables/gt_cs_txt2sql.json"

    prompt_filler = PromptFiller(message_folder_path, dataset_path, c.DOMAIN_CS, c.TOPIC_TXT2SQL)
    filled_templates = prompt_filler.fill_for_extraction_tasks()
    print(filled_templates)
    print(len(filled_templates))



