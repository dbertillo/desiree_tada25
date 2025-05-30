import os
import json
import pandas as pd
from termcolor import colored

def select_llm(experiment):
    if 'chatgpt4+llama8' in experiment:
        return 'gpt4-o+llama8'
    if 'llama70+llama8' in experiment:
        return 'llama70+llama8'
    if 'claude3' in experiment:
        return 'claude3'
    elif 'chatgpt4' in experiment or 'gpt4o' in experiment or 'gpt4-o' in experiment:
        return 'gpt4-o'
    elif 'llama8' in experiment:
        return 'llama8'
    elif 'llama70' in experiment:
        return 'llama70'
    else:
        raise ValueError(f"LLM not found for experiment: {experiment}")
    
def select_pipeline(experiment):
    if 'bootstrap_1_shot' in experiment:
        return 'boostrap (1-shot)'
    elif '1_1_easy_shot' in experiment:
        return 'direct extracion (1-shot)'
    elif '1_0_shot' in experiment or '1_0_':
        return 'direct extracion (0-shot)'


def values_to_compute_total_precision_recall(matched_extracted_specifications, unmtached_extracted_specifications, ground_truth_claims):
    matched_extracted_specification_len = len(matched_extracted_specifications)
    unmatched_extracted_specification_len = len(unmtached_extracted_specifications)

    total_ground_truth_specifications = 0
    for claim in ground_truth_claims:
        subject = claim['subject']
        total_ground_truth_specifications += len(subject)

    return matched_extracted_specification_len, unmatched_extracted_specification_len, total_ground_truth_specifications

def count_elements(mapped_element, table_content, in_table, in_caption, in_paragraph, in_footnote, generated, in_table_gt, in_caption_gt, in_paragraph_gt, in_footnote_gt, human_annotation):
    for e_spec, g_spec in mapped_element.items():
        # print(e_spec + " : " + g_spec)
        # Check for table
        e_spec = e_spec.strip().lower()
        if check_presence(e_spec, table_content['html_table']):
            in_table += 1
        elif check_presence(e_spec, table_content['caption']):
            in_caption += 1
        elif check_presence(e_spec, table_content.get('paragraph')):
            in_paragraph += 1
        elif check_presence(e_spec, table_content.get('footnote')):
            in_footnote += 1
        else:
            generated += 1

        g_spec = g_spec.strip().lower()
        if check_presence(g_spec, table_content['html_table']):
            in_table_gt += 1
        # Check for caption
        elif check_presence(g_spec, table_content['caption']):
            in_caption_gt += 1
        # Check for paragraph
        elif check_presence(g_spec, table_content.get('paragraph')):
            in_paragraph_gt += 1
        # Check for footnote
        elif check_presence(g_spec, table_content.get('footnote')):
            in_footnote_gt += 1
        else:
            # print(colored("Unknown: ", "red") + e_spec + " : " + g_spec)
            human_annotation += 1
    return in_table, in_caption, in_paragraph, in_footnote, generated, in_table_gt, in_caption_gt, in_paragraph_gt, in_footnote_gt, human_annotation


def find_origin_gt_elements(unmatched_elements_ground_truth, table_content):
    in_table_repetition = 0
    in_caption_repetition = 0
    in_paragraph_repetition = 0
    in_footnote_repetition = 0
    human_repetition = 0

    in_table_unique = 0
    in_caption_unique = 0
    in_paragraph_unique = 0
    in_footnote_unique = 0
    human_unique = 0

    for element in unmatched_elements_ground_truth:
        # Check for table
        element = element.strip().lower()
        if check_presence(element, table_content['html_table']):
            in_table_repetition += 1
        elif check_presence(element, table_content['caption']):
            in_caption_repetition += 1
        elif check_presence(element, table_content.get('paragraph')):
            in_paragraph_repetition += 1
        elif check_presence(element, table_content.get('footnotes')):
            in_footnote_repetition += 1
        else:
            human_repetition += 1
    
    set_unmatched_elements_ground_truth = set(unmatched_elements_ground_truth)
    for element in set_unmatched_elements_ground_truth:
        # Check for table
        element = element.strip().lower()
        if check_presence(element, table_content['html_table']):
            in_table_unique += 1
        elif check_presence(element, table_content['caption']):
            in_caption_unique += 1
        elif check_presence(element, table_content.get('paragraph')):
            in_paragraph_unique += 1
        elif check_presence(element, table_content.get('footnotes')):
            in_footnote_unique += 1
        else:
            human_unique += 1
    return in_table_repetition, in_caption_repetition, in_paragraph_repetition, in_footnote_repetition, human_repetition, in_table_unique, in_caption_unique, in_paragraph_unique, in_footnote_unique, human_unique

def check_presence(s, source):
    if source is None:
        return False
    if isinstance(source, list):
        source = "".join(source)
    source = source.strip().lower()
    if s in source:
        return True
    return False

def find_unmatched_ground_truth_elements(mapped_keys, mapped_values, ground_truth_claims):
    mapped_keys_inverse = {v: k for k, v in mapped_values.items()}
    mapped_value_inverse = {v: k for k, v in mapped_keys.items()}
    # Merge the two inverse dictionaries into one
    mapped_inverse = {}
    mapped_inverse.update(mapped_keys_inverse)
    mapped_inverse.update(mapped_value_inverse)
    unmatched_elements_gt = []
    all_elements_gt = []
    for claim in ground_truth_claims:
        subject = claim['subject']
        for key, value in subject.items():
            all_elements_gt.append(key)
            all_elements_gt.append(value)
            if key not in mapped_inverse:
                unmatched_elements_gt.append(key)
            if value not in mapped_inverse:
                unmatched_elements_gt.append(value)
    return unmatched_elements_gt, len(set(all_elements_gt))

def retrieve_matched_gt_claims(matched_claims):
    gt_claims = []
    for match in matched_claims:
        ground_truth_claim = match['Ground_truth_claim']
        gt_claims.append(ground_truth_claim)
    return gt_claims


    

def analyze_specs(data_source_path, judge_pairs_path, judge_specs_path, ground_truth_path, txt2sql_experiments_to_check):
    print(colored("Missed:: Rispondiamo alla domanda: Da dove provengono gli elementi della ground truth che non sono stati mappati?", 'red'))
    with open(data_source_path, "r") as f:
        data_source = json.load(f)
    experiments = os.listdir(judge_specs_path)
    print("experiments: ", experiments)

    spec_analysis_by_table = []
    spec_analysis_by_experiment = {}
    for experiment in experiments:
        if experiment not in txt2sql_experiments_to_check:
            continue
        if 'llm_as_judge_specs' == experiment:
            continue

        matched_specifications_len = 0
        unmatched_specifications_len = 0
        ground_truth_specifications_len = 0
        total_missed_in_table_unique = 0
        total_missed_in_caption_unique = 0
        total_missed_in_paragraph_unique = 0
        total_missed_in_footnote_unique = 0
        total_n_unique_elements = 0
        with open(ground_truth_path, "r") as f:
            gt_claim = json.load(f)

        print(colored(experiment, 'green'))
        experiment_path_evaluation = os.path.join(judge_specs_path, experiment, 'evaluation_results.json')
        evaluation_path = os.path.join(judge_pairs_path, experiment, 'evaluation_results.json')
        with open(evaluation_path, "r") as f:
            matched_claims = json.load(f)
        print("experiment_path: ", experiment_path_evaluation)
        if os.path.isfile(experiment_path_evaluation):
            with open(experiment_path_evaluation, "r") as f:
                data = json.load(f)
                for table_id, evaluation_data in data.items():
                    if table_id  not in data_source.keys():
                        continue
                    # print(colored(f"\t{table_id}", 'cyan'))
                    matched_gt_claims = retrieve_matched_gt_claims(matched_claims[table_id]['matches'])
                    current_gt_claim = gt_claim[table_id]
                    in_table = 0
                    in_caption = 0
                    in_paragraph = 0
                    in_footnote = 0
                    generated = 0
                    in_table_gt = 0
                    in_caption_gt = 0
                    in_paragraph_gt = 0
                    in_footnote_gt = 0
                    human_annotation = 0
                    in_table, in_caption, in_paragraph, in_footnote, generated, in_table_gt, in_caption_gt, in_paragraph_gt, in_footnote_gt, human_annotation = count_elements(evaluation_data['mapped_keys'], data_source[table_id], in_table, in_caption, in_paragraph, in_footnote, generated, in_table_gt, in_caption_gt, in_paragraph_gt, in_footnote_gt, human_annotation)
                    # Second call with unpacked addition
                    in_table, in_caption, in_paragraph, in_footnote, generated, in_table_gt, in_caption_gt, in_paragraph_gt, in_footnote_gt, human_annotation = count_elements(evaluation_data['mapped_values'], data_source[table_id], in_table, in_caption, in_paragraph, in_footnote, generated, in_table_gt, in_caption_gt, in_paragraph_gt, in_footnote_gt, human_annotation)
                    unmatched_elements_ground_truth, n_unique_elements = find_unmatched_ground_truth_elements(evaluation_data['mapped_keys'], evaluation_data['mapped_values'], matched_gt_claims)
                    # missed_in_table, missed_in_caption, missed_in_paragraph, missed_in_footnote, missed_human = find_origin_gt_elements(unmatched_elements_ground_truth, cs_txt2sql_data_source[table_id])
                    missed_in_table_repetition, missed_in_caption_repetition, missed_in_paragraph_repetition, missed_in_footnote_repetition, missed_human_repetition, missed_in_table_unique, missed_in_caption_unique, missed_in_paragraph_unique, missed_in_footnote_unique, missed_human_unique = find_origin_gt_elements(unmatched_elements_ground_truth, data_source[table_id])
                    
                    specs_analysis_experiment_table = {
                        'experiment_id': experiment,
                        'table_id': table_id,
                        'number of matched elements': len(evaluation_data['mapped_keys']) + len(evaluation_data['mapped_values']),
                        'in_table': in_table,
                        'in_caption': in_caption,
                        'in_paragraph': in_paragraph,
                        'in_footnote': in_footnote,
                        'generated': generated,
                        'in_table_gt': in_table_gt,
                        'in_caption_gt': in_caption_gt,
                        'in_paragraph_gt': in_paragraph_gt,
                        'in_footnote_gt': in_footnote_gt,
                        'human_annotation': human_annotation,
                        'in_table / in_table gt': in_table / in_table_gt if in_table_gt > 0 else 0,
                        'in_caption / in_caption gt': in_caption / in_caption_gt if in_caption_gt > 0 else 0,
                        'in_paragraph / in_paragraph gt': in_paragraph / in_paragraph_gt if in_paragraph_gt > 0 else 0,
                        'in_footnote / in_footnote gt': in_footnote / in_footnote_gt if in_footnote_gt > 0 else 0,
                        'generated / human_annotation': generated / human_annotation if human_annotation > 0 else 0,
                        # 'missed_in_table_repetition': missed_in_table_repetition,
                        # 'missed_in_caption_repetition': missed_in_caption_repetition,
                        # 'missed_in_paragraph_repetition': missed_in_paragraph_repetition,
                        # 'missed_in_footnote_repetition': missed_in_footnote_repetition,
                        # 'missed_human_annotation_repetition': missed_human_repetition,
                        # 'pct_missed_in_table_repetition': missed_in_table_repetition / n_unique_elements if n_unique_elements > 0 else 0,
                        # 'pct_missed_in_caption_repetition': missed_in_caption_repetition / n_unique_elements if n_unique_elements > 0 else 0,
                        # 'pct_missed_in_paragraph_repetition': missed_in_paragraph_repetition / n_unique_elements if n_unique_elements > 0 else 0,
                        # 'pct_missed_in_footnote_repetition': missed_in_footnote_repetition / n_unique_elements if n_unique_elements > 0 else 0,
                        # 'pct_missed_human_annotation_repetition': missed_human_repetition / n_unique_elements if n_unique_elements > 0 else 0,
                        'missed_in_table_unique': missed_in_table_unique,
                        'missed_in_caption_unique': missed_in_caption_unique,
                        'missed_in_paragraph_unique': missed_in_paragraph_unique,
                        'missed_in_footnote_unique': missed_in_footnote_unique,
                        'missed_human_annotation_unique': missed_human_unique,
                        'pct_missed_in_table_unique': missed_in_table_unique / n_unique_elements if n_unique_elements > 0 else 0,
                        'pct_missed_in_caption_unique': missed_in_caption_unique / n_unique_elements if n_unique_elements > 0 else 0,
                        'pct_missed_in_paragraph_unique': missed_in_paragraph_unique / n_unique_elements if n_unique_elements > 0 else 0,
                        'pct_missed_in_footnote_unique': missed_in_footnote_unique / n_unique_elements if n_unique_elements > 0 else 0,
                        'pct_missed_human_annotation_unique': missed_human_unique / n_unique_elements if n_unique_elements > 0 else 0,
                        'precision': evaluation_data['precision'],
                        'recall': evaluation_data['recall'],
                    }
                    # print(colored("\t\tspecs_data: ", 'blue'), specs_analysis_experiment_table)
                    spec_analysis_by_table.append(specs_analysis_experiment_table)
        
                    matched_extracted_specification_len, unmatched_extracted_specification_len, total_ground_truth_specifications = values_to_compute_total_precision_recall(evaluation_data['matched_specifications'], evaluation_data['unmatched_specifications'], current_gt_claim)
                    matched_specifications_len += matched_extracted_specification_len
                    unmatched_specifications_len += unmatched_extracted_specification_len
                    ground_truth_specifications_len += total_ground_truth_specifications
                    total_missed_in_table_unique += missed_in_table_unique 
                    total_missed_in_caption_unique += missed_in_caption_unique 
                    total_missed_in_paragraph_unique += missed_in_paragraph_unique 
                    total_missed_in_footnote_unique += missed_in_footnote_unique + missed_human_unique
                    total_n_unique_elements += n_unique_elements
        spec_analysis_by_experiment[experiment] = {
            'pipeline': select_pipeline(experiment),
            'llm': select_llm(experiment),
            'precision': matched_specifications_len / (matched_specifications_len + unmatched_specifications_len) if (matched_specifications_len + unmatched_specifications_len) > 0 else 0,
            'noise': 1 - (matched_specifications_len / (matched_specifications_len + unmatched_specifications_len)) if (matched_specifications_len + unmatched_specifications_len) > 0 else 0,
            'recall': matched_specifications_len / ground_truth_specifications_len if ground_truth_specifications_len > 0 else 0,
            'f1_measure': 2 * (matched_specifications_len / (matched_specifications_len + unmatched_specifications_len)) * (matched_specifications_len / ground_truth_specifications_len) / ((matched_specifications_len / (matched_specifications_len + unmatched_specifications_len)) + (matched_specifications_len / ground_truth_specifications_len)) if (matched_specifications_len + unmatched_specifications_len) > 0 and ground_truth_specifications_len > 0 else 0,
            'missed_in_table_unique': total_missed_in_table_unique / total_n_unique_elements if total_n_unique_elements > 0 else 0,
            'missed_in_caption_unique': total_missed_in_caption_unique / total_n_unique_elements if total_n_unique_elements > 0 else 0,
            'missed_in_footnote_unique': total_missed_in_footnote_unique / total_n_unique_elements if total_n_unique_elements > 0 else 0,
            'missed_in_paragraph_unique': 1 - ( (total_missed_in_table_unique / total_n_unique_elements) + (total_missed_in_caption_unique / total_n_unique_elements) + (total_missed_in_footnote_unique / total_n_unique_elements) ) if total_n_unique_elements > 0 else 0,

        }
    return spec_analysis_by_experiment, spec_analysis_by_table
                    

def txt2sql():

    txt2sql_data_source_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/data/extracted_tables/gt_cs_txt2sql.json"
    txt2sql_ground_truth_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/gt_cs_txt2sql/claims.json"
    txt2sql_judge_pairs_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/cs_judge_pairs_0_shot_claude3"
    txt2sql_judge_specs_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/cs_judge_specs_claude3_best"
    txt2sql_experiments_to_check = [
        # "cs_1_0_shot_chatgpt4",
        "cs_1_1_easy_shot_chatgpt4"
        # "cs_1_0_shot_llama8",
        # "cs_1_1_easy_shot_llama8",
        # "cs_1_0_shot_claude3",
        # "cs_1_1_easy_shot_claude3",
        # "cs_1_0_shot_llama70",
        # "cs_1_1_easy_shot_llama70",
        # "cs_bootstrap_1_shot_chatgpt4+llama8"
    ]
    if not os.path.exists(txt2sql_data_source_path):
        print(colored(f"Data source file {txt2sql_data_source_path} does not exist.", 'red'))
        exit(1)
    
    if not os.path.exists(txt2sql_judge_pairs_path):
        print(colored(f"Judge pairs path {txt2sql_judge_pairs_path} does not exist.", 'red'))
        exit(1)
    
    if not os.path.exists(txt2sql_judge_specs_path):
        print(colored(f"Judge specs path {txt2sql_judge_specs_path} does not exist.", 'red'))
        exit(1)
    txt2sql_output_file_name = "best_cs_txt2sql_specs_analysis_by_experiment.xlsx"

    spec_analysis_by_experiment, spec_analysis_by_table = analyze_specs_2(txt2sql_data_source_path, txt2sql_judge_pairs_path, txt2sql_judge_specs_path, txt2sql_ground_truth_path, txt2sql_experiments_to_check, txt2sql_output_file_name)
    return spec_analysis_by_experiment, spec_analysis_by_table
    # pd.DataFrame(spec_analysis_by_experiment).T.to_excel(os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025", "cs_txt2sql_specs_analysis_by_experiment.xlsx"), index=True)
    # pd.DataFrame(spec_analysis_by_table).to_excel(os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025", "cs_txt2sql_specs_analysis_by_table.xlsx"), index=False)

def analyze_origin(ground_truth_matched_claims, data_source_table, ground_truth_claims):
    def clean_string(s):
        if isinstance(s, int):
            s = str(s)
        return s.strip().lower().replace('\n', ' ').replace(' ', ''). replace('\t', '').replace('%', '').replace('(', '').replace(')', '').replace(',', '').replace('.', '')
    
    origin_structured = 0
    origin_unstrucutred = 0
    checked_specification = []
    ground_truth_unmatched_claims = [claim for claim in ground_truth_claims if claim not in ground_truth_matched_claims]
    
    for claim in ground_truth_unmatched_claims:
        html_table = clean_string(data_source_table.get('html_table', ''))
        for key, value in claim['subject'].items():
            if (key, value) in checked_specification:
                continue
            key = clean_string(key)
            value = clean_string(value)
            if key in html_table or value in html_table:
                origin_structured += 1
            else:
                # print(colored(f"Unstructured: {key} : {value}", 'red'))
                origin_unstrucutred += 1
            checked_specification.append((key, value))

    return origin_structured, origin_unstrucutred, len(checked_specification)


def analyze_specs_2(data_source_path, judge_pairs_path, judge_specs_path, ground_truth_path, experiments_to_check, output_file_name):

    # LOAD DATA SOURCE
    with open(data_source_path, "r") as f:
        data_source = json.load(f)
    
    # EXPERIMENTS
    # Filter experiments to keep only those that match with txt2sql_experiments_to_check
    print(judge_pairs_path)
    experiments = [exp for exp in os.listdir(judge_pairs_path) if exp in experiments_to_check]
    results_by_experiment = {}
    results_by_experiment_table = []
    for experiment in experiments:
        print("Experiment: ", experiment)

        extracted_claims_path = os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/", experiment, 'claims.json')
        with open(extracted_claims_path, "r") as f:
            extracted_claims = json.load(f)
        
        evaluation_specs_path = os.path.join(judge_specs_path, experiment, 'evaluation_results.json')
        with open(evaluation_specs_path, "r") as f:
            evaluation_specs_data = json.load(f)
        
        evaluation_pairs_path = os.path.join(judge_pairs_path, experiment, 'evaluation_results.json')
        with open(evaluation_pairs_path, "r") as f:
            evaluation_pairs_data = json.load(f)
        
        with open(ground_truth_path, "r") as f:
            ground_truth_claims = json.load(f)

        total_matched_specifications_number = 0
        total_unmatched_specifications_number = 0
        total_ground_truth_specifications_number = 0
        avg_specifications_per_extracted_claim = 0
        avg_specifications_per_gt_matched_claim = 0
        total_specifications_in_extracted_claims = 0
        total_actual_matching = 0
        total_ground_truth_specifications_in_gt_matched_claims = 0
        total_extracted_claims = 0

        total_origin_structured = 0
        total_origin_unstrucutred = 0
        total_checked_specifications_number = 0
        ground_truth_matched_claims = []
        for table_id, specs_data in evaluation_specs_data.items():
            # print(table_id)
            matched_specifications = len(specs_data['matched_specifications'])
            unmatched_specifications = len(specs_data['unmatched_specifications'])
            # print(colored(f"\tMatched claims: {len(extracted_claims[table_id])}, Ground truth claims: {len(ground_truth_claims[table_id])}", 'cyan'))
            evaluations_pairs_table = evaluation_pairs_data[table_id]
            n_ground_truth_specifications_that_matched = 0
            actual_matchings = 0
            for match in evaluations_pairs_table['matches']:
                if match['match'] == 'yes':
                    actual_matchings += 1
                    ground_truth_match = match['Ground_truth_claim']
                    ground_truth_matched_claims.append(ground_truth_match)
                    n_ground_truth_specifications_that_matched += len(ground_truth_match['subject'])
            # print(colored(f"\tOrigin Structured: {origin_structured}, Origin Unstructured: {origin_unstrucutred}", 'yellow'))
            # print(colored(f"\tChecked Specifications: {len(checked_specification)}", 'yellow'))
            current_precision = matched_specifications / (matched_specifications + unmatched_specifications) if (matched_specifications + unmatched_specifications) > 0 else 0
            current_recall = matched_specifications / n_ground_truth_specifications_that_matched if n_ground_truth_specifications_that_matched > 0 else 0
            current_f1_measure = 2 * (current_precision * current_recall) / (current_precision + current_recall) if (current_precision + current_recall) > 0 else 0
            
            total_matched_specifications_number += matched_specifications
            total_unmatched_specifications_number += unmatched_specifications
            total_ground_truth_specifications_number += n_ground_truth_specifications_that_matched
            total_specifications_in_extracted_claims += matched_specifications + unmatched_specifications
            total_actual_matching += actual_matchings
            total_ground_truth_specifications_in_gt_matched_claims += n_ground_truth_specifications_that_matched
            total_extracted_claims += len(extracted_claims[table_id]) if table_id in extracted_claims else 0
            avg_specifications_per_extracted_claim += (matched_specifications + unmatched_specifications) / len(extracted_claims[table_id]) if len(extracted_claims[table_id]) > 0 else 0
            avg_specifications_per_gt_matched_claim += n_ground_truth_specifications_that_matched / actual_matchings if actual_matchings > 0 else 0

            origin_structured, origin_unstrucutred, checked_specification = analyze_origin(ground_truth_matched_claims, data_source.get(table_id), ground_truth_claims[table_id])
            total_origin_structured += origin_structured
            total_origin_unstrucutred += origin_unstrucutred
            total_checked_specifications_number += checked_specification

            results_by_experiment_table.append({
                'experiment_id': experiment,
                'table_id': table_id,
                'pipeline': select_pipeline(experiment),
                'llm': select_llm(experiment),
                'matched_specifications': matched_specifications,
                'unmatched_specifications': unmatched_specifications,
                'total_ground_truth_specifications_matched': n_ground_truth_specifications_that_matched,
                'precision': current_precision,
                'recall': current_recall,
                'f1_measure': current_f1_measure
            })

        experiment_precision = total_matched_specifications_number / (total_matched_specifications_number + total_unmatched_specifications_number) if (total_matched_specifications_number + total_unmatched_specifications_number) > 0 else 0
        experiment_recall = total_matched_specifications_number / total_ground_truth_specifications_number if total_ground_truth_specifications_number > 0 else 0
        experiment_f1_measure = 2 * (experiment_precision * experiment_recall) / (experiment_precision + experiment_recall) if (experiment_precision + experiment_recall) > 0 else 0
        results_by_experiment[experiment] = {
            'pipeline': select_pipeline(experiment),
            'llm': select_llm(experiment),
            'precision': experiment_precision,
            'recall': experiment_recall,
            'f1_measure': experiment_f1_measure,
            'avg_specifications_per_extracted_claim': total_specifications_in_extracted_claims / total_extracted_claims if total_extracted_claims > 0 else 0,
            'avg_specifications_per_gt_matched_claim':  total_ground_truth_specifications_in_gt_matched_claims / total_actual_matching if total_actual_matching > 0 else 0,
            # 'origin_structured': total_origin_structured,
            # 'origin_unstrucutred': total_origin_unstrucutred,
            # 'checked_specifications_number': total_checked_specifications_number,
            'pct_origin_structured': total_origin_structured / (total_origin_structured + total_origin_unstrucutred) if (total_origin_structured + total_origin_unstrucutred) > 0 else 0,
            'pct_origin_unstructured': total_origin_unstrucutred / (total_origin_structured + total_origin_unstrucutred) if (origin_structured + total_origin_unstrucutred) > 0 else 0
        }
    pd.DataFrame(results_by_experiment).T.to_excel(os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025", output_file_name), index=True)
    pd.DataFrame(results_by_experiment_table).to_excel(os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025", f"all_{output_file_name}"), index=False)

    return results_by_experiment, results_by_experiment_table

def er():

    er_data_source_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/data/extracted_tables/gt_cs_er.json"
    er_ground_truth_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/gt_cs_er/claims.json"
    er_judge_pairs_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/er_cs_judge_pairs_claude3"
    er_judge_specs_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/er_cs_judge_specs_claude3"
    er_experiments_to_check = [
            'er_cs_1_1_easy_shot_gpt4o',
            # 'er_cs_1_0_shot_gpt4o',
            # 'er_cs_1_1_easy_shot_llama70',
            # 'er_cs_1_0_shot_llama70',
            # 'er_cs_1_1_easy_shot_llama8',
            # 'er_cs_1_0_shot_llama8',
            # 'er_cs_1_0_shot_claude3',
            # 'er_cs_1_1_easy_shot_claude3',
            # 'er_cs_judge_pairs_claude3',
            # 'er_cs_bootstrap_1_shot_gpt4o+llama8'
    ]
    if not os.path.exists(er_data_source_path):
        print(colored(f"Data source file {er_data_source_path} does not exist.", 'red'))
        exit(1)
    
    if not os.path.exists(er_judge_pairs_path):
        print(colored(f"Judge pairs path {er_judge_pairs_path} does not exist.", 'red'))
        exit(1)
    
    if not os.path.exists(er_judge_specs_path):
        print(colored(f"Judge specs path {er_judge_specs_path} does not exist.", 'red'))
        exit(1)
    er_output_file_name = "cs_er_specs_analysis_by_experiment.xlsx"
    spec_analysis_by_experiment, spec_analysis_by_table = analyze_specs_2(er_data_source_path, er_judge_pairs_path, er_judge_specs_path, er_ground_truth_path, er_experiments_to_check, er_output_file_name)
    return spec_analysis_by_experiment, spec_analysis_by_table
    # pd.DataFrame(spec_analysis_by_experiment).T.to_excel(os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025", "cs_er_specs_analysis_by_experiment.xlsx"), index=True)
    # pd.DataFrame(spec_analysis_by_table).to_excel(os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025", "cs_txt2sql_specs_analysis_by_table.xlsx"), index=False)

def hiv():
    hiv_data_source_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/data/extracted_tables/gt_med_hiv.json"
    hiv_ground_truth_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/gt_med_hiv/claims.json"
    hiv_judge_pairs_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/hiv_med_judge_pairs_claude3"
    hiv_judge_specs_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/hiv_med_judge_specs_claude3"
    hiv_experiments_to_check = [
        # "hiv_med_1_0_shot_claude3",
        # "hiv_med_1_1_easy_shot_claude3",
        # "hiv_med_1_0_shot_gpt4o",
        "hiv_med_1_1_easy_shot_gpt4o"
        # "hiv_med_1_0_shot_llama70",
        # "hiv_med_1_1_easy_shot_llama70",
        # "hiv_med_1_0_shot_llama8",
        # "hiv_med_1_1_easy_shot_llama8",
        # "hiv_med_bootstrap_1_shot_chatgpt4+llama8"
    ]
    hiv_output_file_name = "med_hiv_specs_analysis_by_experiment.xlsx"

    spec_analysis_by_experiment, spec_analysis_by_table = analyze_specs_2(hiv_data_source_path, hiv_judge_pairs_path, hiv_judge_specs_path, hiv_ground_truth_path, hiv_experiments_to_check, hiv_output_file_name)
    return spec_analysis_by_experiment, spec_analysis_by_table
def pancreatic_cancer():

    txt2sql_data_source_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/data/extracted_tables/gt_med_pancreatic_cancer.json"
    txt2sql_ground_truth_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/gt_med_pancreatic_cancer/claims.json"
    txt2sql_judge_pairs_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/med_judge_pairs_0_shot_claude3"
    txt2sql_judge_specs_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/med_judge_specs_claude3_best"
    txt2sql_experiments_to_check = [
            # "med_1_0_claude3",
            # "med_1_1_easy_shot_claude3",
            # "med_1_0_chatgpt4",
            # "med_1_1_easy_shot_chatgpt4"
            # "med_1_0_llama70",
            "med_1_1_easy_shot_llama70"
            # "med_1_0_llama8",
            # "med_1_1_easy_shot_llama8",
            # "med_boostrap_0_shot_llama70+llama8",
    ]
    if not os.path.exists(txt2sql_data_source_path):
        print(colored(f"Data source file {txt2sql_data_source_path} does not exist.", 'red'))
        exit(1)
    
    if not os.path.exists(txt2sql_judge_pairs_path):
        print(colored(f"Judge pairs path {txt2sql_judge_pairs_path} does not exist.", 'red'))
        exit(1)
    
    if not os.path.exists(txt2sql_judge_specs_path):
        print(colored(f"Judge specs path {txt2sql_judge_specs_path} does not exist.", 'red'))
        exit(1)
    pc_output_file_name = "med_pc_specs_analysis_by_experiment.xlsx"
    spec_analysis_by_experiment, spec_analysis_by_table = analyze_specs_2(txt2sql_data_source_path, txt2sql_judge_pairs_path, txt2sql_judge_specs_path, txt2sql_ground_truth_path, txt2sql_experiments_to_check, pc_output_file_name)
    return spec_analysis_by_experiment, spec_analysis_by_table
    # pd.DataFrame(spec_analysis_by_experiment).T.to_excel(os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025", "med_pc_specs_analysis_by_experiment.xlsx"), index=True)
    # pd.DataFrame(spec_analysis_by_table).to_excel(os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025", "cs_txt2sql_specs_analysis_by_table.xlsx"), index=False)



if __name__ == "__main__":
    # Set pandas display options to show full dataframes
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    result_by_experiment, result_by_experiment_table = txt2sql()
    print(pd.DataFrame(result_by_experiment).T)

    result_by_experiment, result_by_experiment_table = er()
    print(pd.DataFrame(result_by_experiment).T)

    result_by_experiment, result_by_experiment_table = pancreatic_cancer()
    print(pd.DataFrame(result_by_experiment).T)

    result_by_experiment, result_by_experiment_table = hiv()
    print(pd.DataFrame(result_by_experiment).T)
