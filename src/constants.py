# src/constants.py

######### PROJECT SETTINGS #########
# Directory keys
DIR_DATA = "data"
DIR_EXTRACTED_TABLES = "extracted_tables"
DIR_EXPERIMENTS = "experiments"
DIR_PROMPT_TEMPLATES = "prompt_templates"
DIR_EXAMPLES = "examples"
DIR_ARTICLES = "articles"
DIR_CONFIG = "config"

# Filename keys
FILENAME_PIPELINE_CONFIG = "pipelines.yaml"
FILENAME_EXPERIMENTS_CONFIG = "experiments.yaml"
FILENMES_EXPERIMENTS_TO_RUN = "experiments_to_run.yaml"
######### \PROJECT SETTINGS #########

#--------------------------------------------

######### EXPERIMENT SETTINGS #########
# Directories keys
# DIR_TASK = "task"
DIR_PROMPTS = "prompts"
DIR_FILLED_TEMPLATES = "filled_templates"
DIR_RAW_RESPONSES = "raw_responses"
DIR_METADATA = "metadata"
DIR_VALIDATED_RESPONSES = "validated_responses"

# Filename keys
FILENAME_CLAIMS = "claims.json"
FILENAME_LOG = "experiment.log"
######### \EXPERIMENT SETTINGS #########

#Keywords used in experiments.yaml
DESCRIPTION = "description"
DOMAIN = "domain"
TOPIC = "topic"
DATASET = "dataset"
PIPELINE_ID = "pipeline_id"
MODEL = "model"
MODEL_ID = "model_id"
MODEL_TYPE = "model_type"
MODEL_SIZE = "model_size"
MODEL_VERSION = "model_version"
SHOTS = "shots"
SHOTS_MODE = "shots_mode"
SHOTS_EXAMPLES = "shots_examples"
MODEL_ID_SECOND_STEP = "model_id_second_step"
MODEL_TYPE_SECOND_STEP = "model_type_second_step"
MODEL_ID_THIRD_STEP = "model_id_third_step"
MODEL_TYPE_THIRD_STEP = "model_type_third_step"

#Keywords used in pipelines.yaml
PIPELINE_STEPS = "steps"
PIPELINE_NAME = "name"
PIPELINE_DESCRIPTION = "description"

# Keywords used in steps.yaml
TASK = "task"
MESSAGE_FOLDER = "message_folder"
CLAIMS_FORMAT = "claims_format"

CLAIMS_FORMAT_PAIRS = "pairs"
CLAIMS_FORMAT_TRIPLETS = "triplets"

# Keywords for judge pipelines
PROCESS_EXPERIMENTS = "process_experiments"
IGNORE_EXPERIMENTS = "ignore_experiments"
EVALUATED_EXPERIMENTS = "evaluated_experiments"
PAIRS_JUDGE_EXPERIMENT = "pairs_judge_experiment"
# --------------------------------------------

######### PLACEHOLDERS #########

# Domains keys
DOMAIN_CS = "computer science"
DOMAIN_MED = "medicine"

# Topics keys
TOPIC_TXT2SQL = "txt2sql"
TOPIC_PANCREATIC_CANCER = "pancreatic_cancer"
TOPIC_ER = "entity resolution"
TOPIC_HIV = "hiv"

# Placeholders keys
PLACEHOLDER_DOMAIN = "#placeholder{domain}"
PLACEHOLDER_CITATIONS = "#placeholder{citations}"
PLACEHOLDER_CAPTION = "#placeholder{caption}"
PLACEHOLDER_TABLE = "#placeholder{table}"
PLACEHOLDER_TOPIC = "#placeholder{topic}"
PLACEHOLDER_CLAIMS = "#placeholder{claims}"
PLACEHOLDER_GROUND_TRUTH = "#placeholder{ground_truth}"
PLACEHOLDER_TABLE_HEAD = "#placeholder{table_head}"
PLACEHOLDER_FOOTNOTES = "#placeholder{footnotes}"
PLACEHOLDER_EXAMPLES = "#placeholder{examples}"
PLACEHOLDER_CANDIDATE_PAIRS = "#placeholder{candidates_pairs}"
PLACEHOLDER_EXTRACTED_CLAIM = "#placeholder{extracted_claim}"
PLACEHOLDER_GROUND_TRUTH_CLAIM = "#placeholder{ground_truth_claim}"
PLACEHOLDER_EXTRACTED_ELEMENT = "#placeholder{extracted_element}"
PLACEHOLDER_GROUND_TRUTH_ELEMENT = "#placeholder{ground_truth_element}"

# --------------------------------------------

######### DATASOURCE KEYWORDS #########

DATASOURCE_CAPTION = "caption"
DATASOURCE_CITATIONS = "citations"
DATASOURCE_TABLE_HEAD = "table_head"
DATASOURCE_FOOTNOTES = "footnotes"
DATASOURCE_HTML_TABLE = "html_table"
DATASOURCE_LATEX_TABLE = "latex_table"
DATASOURCE_OG_TABLE = "og_table"
DATASOURCE_CANDIDATES_PAIRS = "candidates_pairs"


# --------------------------------------------


######### CLAIMS KEYWORDS #########

CLAIMS = "claims"
MEASURES = "measures"
OUTCOMES = "outcomes"
MEASURE = "measure"
OUTCOME = "outcome"
SPECIFICATIONS = "specifications"
SUBJECT = "subject"
