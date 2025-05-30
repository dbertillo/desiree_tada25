from termcolor import colored


def pretty_print(delimiter, s, color):
    """
    Pretty print a string with a delimiter and color.
    """
    print(colored(delimiter, color))
    print(colored(s, color))
    print(colored(delimiter, color))

def pretty_print_raw_response(raw_response):
    """
    Pretty print the raw response from the LLM.
    """
    color = 'yellow'
    delimiter = "\t\t----??????????????????????----"
    pretty_print(delimiter, raw_response, color)

def pretty_print_raw_response_specs(extracted_element, ground_truth_element, raw_response):
    """
    Pretty print the raw response from the LLM.
    """
    color = 'yellow'
    delimiter = "\t\t----??????????????????????----"
    response = f"EXTRACTED: {extracted_element}\nGROUND TRUTH: {ground_truth_element}\nRAW RESPONSE: {raw_response}"
    pretty_print(delimiter, response, color)

def pretty_print_candidate_claims(candidate_claims):
    """
    Pretty print the candidate claims with a delimiter and color.
    """
    delimiter = "\t\t----vvvvvvvvvvvvvvvvvv----"
    delimiter = "-"
    color = 'green'
    pretty_print(delimiter, candidate_claims, color)


def pretty_print_errors(errors):
    """
    Pretty print the errors with a delimiter and color.
    """
    delimiter = "\t\t----xxxxxxxxxxxxxxxxx----"
    color = 'red'
    pretty_print(delimiter, errors, color)