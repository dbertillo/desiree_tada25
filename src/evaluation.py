import os
import json
import pandas as pd
from termcolor import colored
import seaborn as sns
import matplotlib.pyplot as plt


def corr_matrix_by_pipeline(results_path):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    results_path = os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025", results_path)
    df_results = pd.read_excel(results_path)

    features = ['relational_table', 'nested_table', 'cross_table', 
                'nested_index_table', 'nested_header_table', 
                '#rows', '#columns', '#cells', 'dim_measures_outcomes_vectors']
    
    metrics = ['precision', 'recall', 'f1_measure']

    topic = results_path.split('/')[-1].split('_')[2]
    pipelines = df_results['pipeline'].unique()

    for pipeline in pipelines:
        df_pipeline = df_results[df_results['pipeline'] == pipeline]
        
        # Skip if not enough data
        if len(df_pipeline) < 3:
            print(f"Skipping pipeline '{pipeline}' due to insufficient data")
            continue

        corr_matrix = df_pipeline[features + metrics].corr()
        metric_feature_corr = corr_matrix.loc[features, metrics]
        threshold = 0.3
        mask = metric_feature_corr.abs() < threshold

        plt.figure(figsize=(8, 6))
        sns.heatmap(metric_feature_corr, annot=True, mask=mask, cmap="RdYlGn", center=0)
        plt.title(f"[{topic} - {pipeline}] Correlation of Metrics with Table Features")
        plt.ylabel("Table/Claims Features")
        plt.xlabel("Evaluation Metrics")
        plt.tight_layout()

        output_dir = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025/plots"
        filename = f"{topic}_{pipeline}_corr_matrix.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

def corr_matrix_llm(results, features, topic, corr_dimension):
    results = results.groupby(corr_dimension)[["precision", "recall", "f1_measure"] + features].mean()
    encoded = pd.get_dummies(results[corr_dimension], prefix=corr_dimension)

    # Combine encoded models with the feature columns
    correlation_df = pd.concat([encoded, results[features]], axis=1)

    # Compute correlation matrix
    correlation_matrix = correlation_df.corr()

    # Extract only the correlations between models and features
    model_feature_corr = correlation_matrix.loc[encoded.columns, features]
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(model_feature_corr, annot=True, cmap="RdYlGn", center=0)
    plt.title(f"[{topic}] - Correlation Between {corr_dimension} Models and Table Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025/plots", f"{topic}_{corr_dimension}_corr_matrix.png"))

    # plt.show()


def corr_matrix(results_path):
    # table_id	relation_table	nested_table	cross_table	nested_index_table	nested_header_table	characters_count	data_table	result_table	rows (no index)	columns (index included)	notes
    results_path = os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025", results_path)
    df_results = pd.read_excel(results_path)

    features = ['relational_table', 'nested_table', 'cross_table', 'nested_index_table', 'nested_header_table', '#rows', '#columns', '#cells', 'dim_measures_outcomes_vectors']

    metrics = ['precision', 'recall', 'f1_measure']
    corr_matrix = df_results[features + metrics].corr()
    metric_feature_corr = corr_matrix.loc[features, metrics]
    threshold = 0.3
    mask = metric_feature_corr.abs() < threshold

    topic = results_path.split('/')[-1].split('_')[2]
    print(topic)
    # Plot as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(metric_feature_corr, annot=True, mask=mask, cmap="RdYlGn", center=0)
    plt.title(f"[{topic}] - Correlation of Precision, Recall, and F1-Measure with Table Features")
    plt.ylabel("Table/Claims Features")
    plt.xlabel("Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025/plots", f"{topic}_corr_matrix.png"))

    # corr_dimensions = ['llm', 'pipeline']
    # for corr_dimension in corr_dimensions:
    #     corr_matrix_llm(df_results, features, topic, corr_dimension)
    # # plt.show()





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


def normalize_stats_table(stats_table_path):

    def count_rows_and_columns(html_table):
        rows = html_table.count('<tr>')
        if rows > 0:
            columns = html_table.count('<td>') / (2 * rows)
        else:
            rows = html_table.count('<div>')
            columns = html_table.count('<span>') / (2 * rows)
        return rows, columns
    

    df_features_tables = pd.read_excel(stats_table_path)
    features = ['table_id' ,'relational_table', 'nested_table', 'cross_table', 'nested_index_table', 'nested_header_table', 'rows (no index)', 'columns (index included)']

    for feature in features:
        if feature == 'rows (no index)' or feature == 'columns (index included)' or feature == 'table_id':
            continue
        df_features_tables[feature] = df_features_tables[feature].map({"yes": 1, "no": 0, True :1, False: 0, 'True': 1, 'False': 0})
    # Drop columns that are not in the features list
    columns_to_keep = [col for col in df_features_tables.columns if col in features]
    df_features_tables = df_features_tables[columns_to_keep]

    # Rename columns
    df_features_tables = df_features_tables.rename(columns={
        'table_id': 'table',
        'rows (no index)': 'rows number',
        'columns (index included)': 'columns number'
    })

    return df_features_tables

def get_average_dimensions_vectors_measures_outcomes(ground_truth_claims):
    sum_measures = 0
    for claim in ground_truth_claims:
        measures = str(claim['measures'])
        sum_measures += len(measures.split(','))
    return sum_measures / len(ground_truth_claims)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_metrics_by_dim_vectors(all_results_path):
    """
    This function reads the results Excel file, filters out entries related to LLaMA8,
    groups by `dim_measures_outcomes_vectors`, and computes average precision, recall, and F1-measure.
    It also plots these metrics to help visualize performance across different vector dimensions.
    """
    # Load the data
    all_results_path = os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025", all_results_path)

    df = pd.read_excel(all_results_path)

    # Exclude rows where LLM is 'llama8'
    # df_filtered = df[df['llm'] != 'llama8']
    df_filtered = df

    # Group by the number of dimensions (measures+outcomes) and compute average metrics
    grouped = df_filtered.groupby('dim_measures_outcomes_vectors')[['precision', 'recall', 'f1_measure']].mean().reset_index()

    # Sort by dimensions for better visualization
    grouped = grouped.sort_values(by='dim_measures_outcomes_vectors')

    print(grouped)

    # Plotting the metrics
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=grouped, x='dim_measures_outcomes_vectors', y='precision', label='Precision')
    sns.lineplot(data=grouped, x='dim_measures_outcomes_vectors', y='recall', label='Recall')
    sns.lineplot(data=grouped, x='dim_measures_outcomes_vectors', y='f1_measure', label='F1-Measure')
    plt.xlabel('Dimension of Measures/Outcomes Vectors')
    plt.ylabel('Metric Value')
    plt.title('Performance Metrics vs. Vector Dimensions (excluding llama8)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return grouped

def retrieve_evas_claims_metrics(
        experiments_to_check, 
        save, 
        experiment_path, 
        output_file_name, 
        dataset_tables_path,
        stats_tables_path,
        ground_truth_claims_path
        ):
    judje_evaluation_experiment_path = experiment_path
    total_extracted_claims_string = 'total_extracted_claims'
    total_ground_truth_claims_string = 'total_ground_truth_claims'
    number_of_matches_string = 'number_of_matches'
    
    with open(dataset_tables_path, 'r') as f:
        # print(f"Loading {tables_path}...")
        # print("Does it exists? ", os.path.exists(tables_path))
        data_tables = json.load(f)
        tables_keys = data_tables.keys()

    with open(ground_truth_claims_path, 'r') as f:
        # print(f"Loading {ground_truth_claims_path}...")
        # print("Does it exists? ", os.path.exists(ground_truth_claims_path))
        ground_truth_claims = json.load(f)
        

    df_stats_tables = normalize_stats_table(stats_tables_path)
    experiments_evas = os.listdir(judje_evaluation_experiment_path)
    recap_stats = {}
    recap_stats_by_experimentid_table = []
    for experiment in experiments_evas:
        if 'cs' not in experiment and 'med' not in experiment:
            continue
        if experiment not in experiments_to_check:
            continue
        experiment_path = os.path.join(judje_evaluation_experiment_path, experiment)
        print(colored(f"Experiment: {experiment}", 'blue'))
        if os.path.isdir(experiment_path):
            evaluation_data = {}
            for file in os.listdir(experiment_path):
                if file == 'evaluation_results.json':
                    with open(os.path.join(experiment_path, file), 'r') as f:
                        # print(f"Loading {file}...")
                        # print("Does it exists? ", os.path.exists(os.path.join(experiment_path, file)))
                        matches = json.load(f)
                        if (len(matches.keys()) == 0):
                            continue
                    for table, content in matches.items():
                        if table not in tables_keys:
                            continue
                        # print("\ttotal extracted claims: ", content[total_extracted_claims])
                        # print("\ttotal ground truth claims: ", content[total_ground_truth_claims])
                        # print("\tnumber of matches: ", content[number_of_matches])
                        precision = int(content[number_of_matches_string]) / int(content[total_extracted_claims_string])
                        recall = int(content[number_of_matches_string]) / int(content[total_ground_truth_claims_string])
                        # print("\t\t\tprecision: ", colored(precision, 'cyan'))
                        # print("\t\t\trecall: ", colored(recall, 'magenta'))

                        # print(colored(f"ground_truth_claims[{table}]", 'green'), ground_truth_claims[table])
                        # print(colored(f"table: {table}", 'green'))
                        # print(colored(df_stats_tables[df_stats_tables['table']==table], 'green'))
                        evaluation_data[table] = ({
                                'experiment': experiment,
                                'table': table,
                                'llm': select_llm(experiment),
                                'pipeline': select_pipeline(experiment),
                                'total_extracted_claims': content[total_extracted_claims_string],
                                'total_ground_truth_claims': content[total_ground_truth_claims_string],
                                'number_of_matches': content[number_of_matches_string],
                                'precision': precision,
                                'recall': recall,
                                'f1_measure': 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0,
                                'relational_table': df_stats_tables[df_stats_tables['table'] == table]['relational_table'].values[0],
                                'nested_table': df_stats_tables[df_stats_tables['table'] == table]['nested_table'].values[0],
                                'cross_table': df_stats_tables[df_stats_tables['table'] == table]['cross_table'].values[0],
                                'nested_index_table': df_stats_tables[df_stats_tables['table'] == table]['nested_index_table'].values[0],
                                'nested_header_table': df_stats_tables[df_stats_tables['table'] == table]['nested_header_table'].values[0],
                                '#rows': df_stats_tables[df_stats_tables['table'] == table]['rows number'].values[0],
                                '#columns': df_stats_tables[df_stats_tables['table'] == table]['columns number'].values[0],
                                '#cells': df_stats_tables[df_stats_tables['table'] == table]['rows number'].values[0] * df_stats_tables[df_stats_tables['table'] == table]['columns number'].values[0],
                                # 'dim_measures_outcomes_vectors': len(str(ground_truth_claims[table][0]['measures']).split(','))
                                'dim_measures_outcomes_vectors': get_average_dimensions_vectors_measures_outcomes(ground_truth_claims[table])
                                })
                        recap_stats_by_experimentid_table.append(evaluation_data[table])
                    df = pd.DataFrame.from_dict(evaluation_data, orient='index')
                    df = df[['table', 'total_extracted_claims', 'total_ground_truth_claims', 'number_of_matches', 'precision', 'recall', 'f1_measure']]
                    df = df.set_index('table')
                    # df = df.sort_values(by=['precision', 'recall'], ascending=False)
                    # print(df)
                    # print("Mean precision: ", df['precision'].mean())
                    # print("Mean recall: ", df['recall'].mean())
                    # print("Mean F1 measure: ", 2 * (df['precision'].mean() * df['recall'].mean()) / (df['precision'].mean() + df['recall'].mean()))

                    total_numebr_of_matches = df['number_of_matches'].sum()
                    total_extracted_claims = df['total_extracted_claims'].sum()
                    total_ground_truth_claims = df['total_ground_truth_claims'].sum()
                    total_precision = total_numebr_of_matches / total_extracted_claims
                    total_recall = total_numebr_of_matches / total_ground_truth_claims
                    total_f1_measure = 2 * (total_precision * total_recall) / (total_precision + total_recall)

                    recap_stats[experiment] = { 
                                                # 'number_of_tables': len(df),
                                                'Pipeline': select_pipeline(experiment),
                                                'LLM': select_llm(experiment),
                                                'Precision': total_precision,
                                                'Recall': total_recall,
                                                'F1-Measure': total_f1_measure,
                                                }
                    # df.to_csv(f'{experiment}_evaluation_results.csv')
                    # df.to_excel(f'{experiment}_evaluation_results.xlsx')
    if save:
        df_recap = pd.DataFrame.from_dict(recap_stats, orient='index')
        # Sort by total_f1_measure in descending order
        df_recap_sorted = df_recap.sort_values(by='Pipeline', ascending=False)
        # Print the sorted DataFrame
        df_recap_sorted.to_excel(os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025", output_file_name))
        print(df_recap_sorted)
        print(df_recap_sorted.set_index('Pipeline').to_latex())
        df_every_result = pd.DataFrame.from_dict(recap_stats_by_experimentid_table)
        df_every_result.to_excel(os.path.join("/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025", f"all_{output_file_name}"))
    return recap_stats, recap_stats_by_experimentid_table

def compute_txt2sql():
    cs_txt2sql_results_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025/cs_txt2sql_total_recap_claims_metrics.xlsx"
    cs_txt2sql_experiments_to_check = [
        "cs_1_0_shot_chatgpt4",
        "cs_1_1_easy_shot_chatgpt4",
        "cs_1_0_shot_llama8",
        "cs_1_1_easy_shot_llama8",
        "cs_1_0_shot_claude3",
        "cs_1_1_easy_shot_claude3",
        "cs_1_0_shot_llama70",
        "cs_1_1_easy_shot_llama70",
        "cs_bootstrap_1_shot_chatgpt4+llama8"
    ]
    cs_txt2sql_experiment_path="/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/cs_judge_pairs_0_shot_claude3"
    cs_txt2sql_output_file_name="cs_txt2sql_total_recap_claims_metrics.xlsx"
    cs_txt2sql_extracted_tables_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/data/extracted_tables/gt_cs_txt2sql.json"
    cs_txt2sql_ground_truth_claims_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/gt_cs_txt2sql/claims.json"
    cs_txt2sql_stats_tables_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/data/extracted_tables/cs_txt2sql_stats_tables.xlsx"
    all_cs_txt2sql_results_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/notebooks/tada2025/all_cs_txt2sql_total_recap_claims_metrics.xlsx"
    # corr_matrix(all_cs_txt2sql_results_path, cs_txt2sql_table_stats_path)
    cs_txt2sql_recap_stats, cs_txt2sql_recap_stats_by_experimentid_table = retrieve_evas_claims_metrics(
        experiments_to_check=cs_txt2sql_experiments_to_check, 
        save=True, 
        experiment_path=cs_txt2sql_experiment_path, 
        output_file_name=cs_txt2sql_output_file_name,
        dataset_tables_path=cs_txt2sql_extracted_tables_path,
        stats_tables_path=cs_txt2sql_stats_tables_path,
        ground_truth_claims_path=cs_txt2sql_ground_truth_claims_path
        )
    corr_matrix(f"all_{cs_txt2sql_output_file_name}")
    corr_matrix_by_pipeline(f"all_{cs_txt2sql_output_file_name}")
    analyze_metrics_by_dim_vectors(f"all_{cs_txt2sql_output_file_name}")

def compute_pancreatic_cancer():

    med_pancreatic_cancer_table_stats_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/data/extracted_tables/med_pc_stats_tables.xlsx"

    med_pancreatic_cancer_experiments_to_check = [
            "med_1_0_claude3",
            "med_1_1_easy_shot_claude3",
            "med_1_0_chatgpt4",
            "med_1_1_easy_shot_chatgpt4",
            "med_1_0_llama70",
            "med_1_1_easy_shot_llama70",
            "med_1_0_llama8",
            "med_1_1_easy_shot_llama8",
            "med_boostrap_0_shot_llama70+llama8",
        ]

    med_pancreatic_cancer_experiment_path="/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/med_judge_pairs_0_shot_claude3"
    med_pancreatic_cancer_output_file_name="med_pancreatic_cancer_total_recap_claims_metrics.xlsx"

    med_pancreatic_cancer_extracted_tables_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/data/extracted_tables/gt_med_pancreatic_cancer.json"

    med_pancreatic_cancer_ground_truth_claims_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/gt_med_pancreatic_cancer/claims.json"


    med_pancreatic_cancer_recap_stats, med_pancreatic_cancer_recap_stats_by_experimentid_table = retrieve_evas_claims_metrics(
        experiments_to_check=med_pancreatic_cancer_experiments_to_check, 
        save=True, 
        experiment_path=med_pancreatic_cancer_experiment_path, 
        output_file_name=med_pancreatic_cancer_output_file_name,
        dataset_tables_path=med_pancreatic_cancer_extracted_tables_path,
        stats_tables_path=med_pancreatic_cancer_table_stats_path,
        ground_truth_claims_path=med_pancreatic_cancer_ground_truth_claims_path
        )
    corr_matrix(f"all_{med_pancreatic_cancer_output_file_name}")
    # corr_matrix_by_pipeline(f"all_{med_pancreatic_cancer_output_file_name}")
    analyze_metrics_by_dim_vectors(f"all_{med_pancreatic_cancer_output_file_name}")

def compute_cs_er():
    cs_er_table_stats_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/data/extracted_tables/cs_er_stats_tables.xlsx"
    cs_er_experiments_to_check = [
            'er_cs_1_1_easy_shot_gpt4o',
            'er_cs_1_0_shot_gpt4o',
            'er_cs_1_1_easy_shot_llama70',
            'er_cs_1_0_shot_llama70',
            'er_cs_1_1_easy_shot_llama8',
            'er_cs_1_0_shot_llama8',
            'er_cs_1_0_shot_claude3',
            'er_cs_1_1_easy_shot_claude3',
            'er_cs_judge_pairs_claude3',
            'er_cs_bootstrap_1_shot_gpt4o+llama8'
    ]
    cs_er_judge_claims_experiment_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/er_cs_judge_pairs_claude3"
    cs_er_output_file_name = "cs_er_total_recap_claims_metrics.xlsx"
    cs_er_extracted_tables_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/data/extracted_tables/gt_cs_er.json"
    cs_er_ground_truth_claims_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/gt_cs_er/claims.json"

    cs_er_recap_stats, cs_er_recap_stats_by_experimentid_table = retrieve_evas_claims_metrics(
        experiments_to_check=cs_er_experiments_to_check, 
        save=True, 
        experiment_path=cs_er_judge_claims_experiment_path, 
        output_file_name=cs_er_output_file_name,
        dataset_tables_path=cs_er_extracted_tables_path,
        stats_tables_path=cs_er_table_stats_path,
        ground_truth_claims_path=cs_er_ground_truth_claims_path
    )

    corr_matrix(f"all_{cs_er_output_file_name}")
    corr_matrix_by_pipeline(f"all_{cs_er_output_file_name}")


def hiv():
    med_hiv_table_stats_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/data/extracted_tables/med_hiv_stats_tables.xlsx"
    hiv_experiments_to_check = [
                                "hiv_med_1_1_easy_shot_gpt4o", 
                                "hiv_med_1_1_easy_shot_claude3", 
                                "hiv_med_1_1_easy_shot_llama70", 
                                "hiv_med_1_1_easy_shot_llama8", 
                                "hiv_med_1_0_shot_claude3", 
                                "hiv_med_1_0_shot_gpt4o",
                                "hiv_med_1_0_shot_llama70",
                                "hiv_med_1_0_shot_llama8",
                                "hiv_med_boostrap_1_shot_chatgpt4+llama8",
                                ]
    med_hiv_experiment_path="/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/hiv_med_judge_pairs_claude3"
    med_hiv_output_file_name="med_hiv_total_recap_claims_metrics.xlsx"
    med_hiv_extracted_tables_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/data/extracted_tables/gt_med_hiv.json"
    med_hiv_ground_truth_claims_path = "/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/gt_med_hiv/claims.json"

    med_hiv_recap_stats, med_hiv_recap_stats_by_experimentid_table = retrieve_evas_claims_metrics(
        experiments_to_check=hiv_experiments_to_check, 
        save=True, 
        experiment_path=med_hiv_experiment_path, 
        output_file_name=med_hiv_output_file_name,
        dataset_tables_path=med_hiv_extracted_tables_path,
        stats_tables_path=med_hiv_table_stats_path,
        ground_truth_claims_path=med_hiv_ground_truth_claims_path
    )

def main():
    # compute_txt2sql()
    # compute_pancreatic_cancer()
    # compute_cs_er()
    hiv()
    return

    
    

    cs_er_experiments_to_check = [
        # "er_cs_1_1_easy_shot_claude3",
        "er_cs_1_1_easy_shot_gpt4o",
        "er_cs_1_1_easy_shot_llama70",
        "er_cs_1_1_easy_shot_llama8",
        "er_cs_boostrap_0_shot_chatgpt4+llama8",
    ]
    cs_er_cancer_experiment_path="/Users/danielebertillo/Desktop/Scrivania - danielAir/expresso/experiments/er_cs_judge_pairs_claude3"
    er_cs_cancer_output_file_name="er_cs_total_recap_claims_metrics.xlsx"
    print(colored("Entity resolution (Computer Science) Experiments to check: ", 'green'), cs_er_experiments_to_check)
    cs_er_cancer_recap_stats, er_cs_cancer_recap_stats_by_experimentid_table = retrieve_evas_claims_metrics(experiments_to_check=cs_er_experiments_to_check, save=True, experiment_path=cs_er_cancer_experiment_path, output_file_name=er_cs_cancer_output_file_name)

from collections import Counter
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # data = [2.0, 1.5, 2.5, 4.0, 1.0, 2.0, 3.0, 3.0, 1.666666667, 1.75, 2.0, 2.0, 2.0, 1.75, 2.0, 2.0, 3.0, 1.0, 1.5, 3.571428571]
    # value_counts = Counter(data)

    # # Sort the values for consistent plotting
    # sorted_items = sorted(value_counts.items())
    # values, counts = zip(*sorted_items)

    # # Plot the histogram (bar plot)
    # plt.figure(figsize=(10, 6))
    # plt.bar(values, counts, width=0.1)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Frequency of Each Unique Value')
    # plt.grid(axis='y')
    # plt.tight_layout()
    # plt.show()    
    main()
