import os
import json

class SpecificationsMatcher():
    def __init__(self, extracted_claims, ground_truth_claims):
        self.mapped_keys = {} # from extracted to ground truth
        self.mapped_values = {} # from extracted to ground truth

        self.unmatched_extracted_keys = []
        self.unmatched_extracted_values = []
        self.unmatched_ground_truth_keys = []
        self.unmatched_ground_truth_values = []

        self.matched_specifications = [] # unmatched extracted specifications
        self.unmatched_specifications = [] #unmatched extracted specifications

        self.extracted_claims_subject = [item['subject'] for item in extracted_claims]
        self.ground_truth_claims_subject = [item['subject'] for item in ground_truth_claims]

        self.map_keys_on_equal_strings(self.extracted_claims_subject, self.ground_truth_claims_subject)
        self.maps_values_on_equal_strings(self.extracted_claims_subject, self.ground_truth_claims_subject)


    def map_keys_on_equal_strings(self, extracted_claims_subject: list, ground_truth_claims_subject: list):

        unmatched_extracted_keys = set()
        unmatched_ground_truth_keys = set()
        
        for extracted_claim_subject in extracted_claims_subject:
            unmatched_extracted_keys.update(k for k, _ in extracted_claim_subject.items())
        for ground_truth_claim_subject in ground_truth_claims_subject:
            unmatched_ground_truth_keys.update(k for k, _ in ground_truth_claim_subject.items())

        # Create normalized key maps
        extracted_key_map = {k.strip().lower(): k for k in unmatched_extracted_keys}
        ground_truth_key_map = {k.strip().lower(): k for k in unmatched_ground_truth_keys}

        # Intersect normalized keys
        common_keys = set(extracted_key_map.keys()) & set(ground_truth_key_map.keys())

        for norm_key in common_keys:
            original_extracted_key = extracted_key_map[norm_key]
            original_ground_truth_key = ground_truth_key_map[norm_key]
            self.mapped_keys[original_extracted_key] = original_ground_truth_key

            unmatched_extracted_keys.remove(original_extracted_key)
            unmatched_ground_truth_keys.remove(original_ground_truth_key)

        self.unmatched_extracted_keys = list(unmatched_extracted_keys)
        self.unmatched_ground_truth_keys = list(unmatched_ground_truth_keys)

    def old_map_keys_on_equal_strings(self, extracted_claims_subject: list, ground_truth_claims_subject: list):

        unmatched_extracted_keys = set()
        unmatched_ground_truth_keys = set()
        for extracted_claim_subject in extracted_claims_subject:
            unmatched_extracted_keys.update(k for k, _ in extracted_claim_subject.items())
        for ground_truth_claim_subject in ground_truth_claims_subject:
            unmatched_ground_truth_keys.update(k for k, _ in ground_truth_claim_subject.items())

        unmatched_extracted_keys = list(unmatched_extracted_keys)
        unmatched_ground_truth_keys = list(unmatched_ground_truth_keys)

        for unmatched_key in unmatched_extracted_keys[:]:
            if unmatched_key in unmatched_ground_truth_keys:
                self.mapped_keys[unmatched_key] = unmatched_key
                unmatched_extracted_keys.remove(unmatched_key)
                unmatched_ground_truth_keys.remove(unmatched_key)

        self.unmatched_extracted_keys = unmatched_extracted_keys
        self.unmatched_ground_truth_keys = unmatched_ground_truth_keys


    def maps_values_on_equal_strings(self, extracted_claims_subject, ground_truth_claims_subject):

        unmatched_extracted_values = set()
        unmatched_ground_truth_values = set()
        
        for extracted_claim_subject in extracted_claims_subject:
            unmatched_extracted_values.update(v for _, v in extracted_claim_subject.items())
        for ground_truth_claim_subject in ground_truth_claims_subject:
            unmatched_ground_truth_values.update(v for _, v in ground_truth_claim_subject.items())

        # Create normalized value maps
        extracted_value_map = {str(v).strip().lower(): v for v in unmatched_extracted_values}
        ground_truth_value_map = {str(v).strip().lower(): v for v in unmatched_ground_truth_values}

        # Intersect normalized values
        common_values = set(extracted_value_map.keys()) & set(ground_truth_value_map.keys())

        for norm_val in common_values:
            original_extracted_val = extracted_value_map[norm_val]
            original_ground_truth_val = ground_truth_value_map[norm_val]
            self.mapped_values[original_extracted_val] = original_ground_truth_val

            unmatched_extracted_values.remove(original_extracted_val)
            unmatched_ground_truth_values.remove(original_ground_truth_val)

        self.unmatched_extracted_values = list(unmatched_extracted_values)
        self.unmatched_ground_truth_values = list(unmatched_ground_truth_values)
    
    def old_maps_values_on_equal_strings(self, extracted_claims_subject, ground_truth_claims_subject):

        unmatched_extracted_values = set()
        unmatched_ground_truth_values = set()
        
        for extracted_claim_subject in extracted_claims_subject:
            unmatched_extracted_values.update(v for _, v in extracted_claim_subject.items())
        for ground_truth_claim_subject in ground_truth_claims_subject:
            unmatched_ground_truth_values.update(v for _, v in ground_truth_claim_subject.items())

        unmatched_extracted_values = list(unmatched_extracted_values)
        unmatched_ground_truth_values = list(unmatched_ground_truth_values)

        for unmatched_value in unmatched_extracted_values[:]:
            if unmatched_value in unmatched_ground_truth_values:
                self.mapped_values[unmatched_value] = unmatched_value
                unmatched_extracted_values.remove(unmatched_value)
                unmatched_ground_truth_values.remove(unmatched_value)
            
        self.unmatched_extracted_values = unmatched_extracted_values
        self.unmatched_ground_truth_values = unmatched_ground_truth_values
            
    def add_llm_mapped_key(self, extracted_key, ground_truth_key):
        if extracted_key not in self.mapped_keys:
            self.mapped_keys[extracted_key] = ground_truth_key
            if extracted_key in self.unmatched_extracted_keys:
                self.unmatched_extracted_keys.remove(extracted_key)
            if ground_truth_key in self.unmatched_ground_truth_keys:
                self.unmatched_ground_truth_keys.remove(ground_truth_key)
        else:
            raise Warning(f"Key {extracted_key} already mapped to {self.mapped_keys[extracted_key]}")

    def add_llm_mapped_value(self, extracted_value, ground_truth_value):
        if extracted_value not in self.mapped_values:
            self.mapped_values[extracted_value] = ground_truth_value
            if extracted_value in self.unmatched_extracted_values:
                self.unmatched_extracted_values.remove(extracted_value)
            if ground_truth_value in self.unmatched_ground_truth_values:
                self.unmatched_ground_truth_values.remove(ground_truth_value)
        else:
            raise Warning(f"Value {extracted_value} already mapped to {self.mapped_values[extracted_value]}")
        
    def is_already_mapped(self, extracted_element):
        if extracted_element in self.mapped_keys or extracted_element in self.mapped_values:
            return True
        return False
    
    def ground_truth_key_already_matched(self, ground_truth_key):
        if ground_truth_key in self.mapped_keys.values():
            return True
        return False

    def ground_truth_value_already_matched(self, ground_truth_value):
        if ground_truth_value in self.mapped_values.values():
            return True
        return False
    
    def get_mapped_keys(self):
        return self.mapped_keys
    
    def get_mapped_values(self):
        return self.mapped_values
    
    def get_unmatched_extracted_keys(self):
        return self.unmatched_extracted_keys
    
    def get_unmatched_extracted_values(self):
        return self.unmatched_extracted_values
    
    def get_matched_specifications(self):
        return self.matched_specifications
    
    def get_unmatched_specifications(self):
        return self.unmatched_specifications

    def get_unmatched_ground_truth_keys(self):
        return self.unmatched_ground_truth_keys
    
    def get_unmatched_ground_truth_values(self):
        return self.unmatched_ground_truth_values

    def compute_precision_recall(self, n_extracted_specfications: int, n_ground_truth_specfications: int):
        precision = len(self.matched_specifications) / n_extracted_specfications if n_extracted_specfications > 0 else 0
        recall = len(self.matched_specifications) / n_ground_truth_specfications if n_ground_truth_specfications > 0 else 0
        return precision, recall

    def evaluate_extracted_specifications(self):
        
        for subject in self.extracted_claims_subject:
            for extracted_key, extracted_value in subject.items():
                ground_truth_key = self.mapped_keys.get(extracted_key)
                ground_truth_value = self.mapped_values.get(extracted_value)
    
                if ground_truth_key and ground_truth_value:
                    self.matched_specifications.append(((extracted_key, extracted_value)))
                else:
                    self.unmatched_specifications.append(((extracted_key, extracted_value)))
        precision, recall = self.compute_precision_recall(sum(len(d) for d in self.extracted_claims_subject), sum(len(d) for d in self.ground_truth_claims_subject))  

        return precision, recall