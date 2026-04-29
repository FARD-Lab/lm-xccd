import os
import re
import json
import pathlib

from sklearn.metrics import precision_score, recall_score, f1_score


current_location = pathlib.Path(__file__).parent.resolve()


class Analyser:
    """
    A class to analyze and extract results from test and report files.
    Attributes:
        test_data (dict): A dictionary containing the test data.
        report_file (str): The path to the report file.
        results (dict): A dictionary containing the extracted results.
    """

    def __init__(self, test_file, report_file) -> None:
        """
        Initializes the Analyser class with the test and report files.

        Args:
            test_file (str): The path to the test data file.
            report_file (str): The path to the report file.
        """
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.response_rate = None
        self.report_file = report_file
        self.test_data = self.read_data(test_file)
        self.results = self._extract_results()
        
    
    def compute_missing_samples(self, type):
        missing_ids = []
        correct_ids = []
        for sample in self.ground_truth:
            sample_key = list(sample.keys())[0]
            if sample_key in self.predicted_results:
                if self.predicted_results[sample_key] != sample[sample_key]:
                    missing_ids.append(sample_key)
                else:
                    correct_ids.append(sample_key)
                    
        with open(os.path.join(current_location, 'offline_results', f'{type}_missing_index.txt'), "w") as file:
            for id in missing_ids:
                file.write(f"{id}\n")
                
        with open(os.path.join(current_location, 'offline_results', f'{type}_correct_index.txt'), "w") as file:
            for id in correct_ids:
                file.write(f"{id}\n")
                
        
    def read_data(self, data_file):
        """
        Reads data from the specified data file and returns it as a dictionary.

        Args:
            data_file (str): The path to the data file.

        Returns:
            dict: The data as a dictionary.
        """
        data = []
        with open(data_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
                
        return data
    
    def _extract_results(self):
        """
        Extracts results from the report file and returns them as a dictionary.

        Returns:
            dict: The extracted results as a dictionary.
        """
        pure_results = self.read_data(self.report_file)
        ### Experiment Specific:
        pure_results = pure_results[:1000]
        result = {}
        for data in pure_results:
            try:
                data_id = data['idx']
            except:
                assert 1 == 1
                
            clone_result = self.extract_llm_result(data['text'])
            if clone_result == -1:
                print(f"Warning: The text for data_id {data_id} does not contain a clear 'yes' or 'no' conclusion.")
                continue
            result[data_id] = clone_result
            
        return result
    
    def extract_llm_result(self, text):
        text = text.split("|assistant|")[1].lower()  # take the last part after the assistant tag
        keys = ["functionality_", "conclusion", ]
        
        idx = text.rfind(keys[0])
        idx2 = text.rfind(keys[1])          # find *last* occurrence
        if idx == -1 and idx2 == -1:
                return -1
        
        if idx or idx2:
            text = text.split(keys[0]) if idx else text.split(keys[1]) 
            if len(text) < 2:
                return -1  # take the part after the last occurrence of the key
                
            tail = text[1][:100]
            non_clone = bool(re.search(r"\bno\b", tail, flags=re.IGNORECASE))
            clone = bool(re.search(r"\byes\b", tail, flags=re.IGNORECASE))
            if clone and not non_clone:
                return 1
        
            elif non_clone and not clone:
                return 0
            else:
                print(f"Warning: The text '{text}' does not contain a clear 'yes' or 'no' conclusion.")
                return -1
                # raise ValueError("The text does not contain a clear 'yes' or 'no' conclusion.")
            
        

    
        # tail = text[idx+5:].lstrip(" :\t\n\r")
        # tail = tail[:10]
        # non_clone = bool(re.search(r"\bno\b", tail, flags=re.IGNORECASE))
        # clone = bool(re.search(r"\byes\b", tail, flags=re.IGNORECASE))


        # if clone and not non_clone:
        #     return 1
        
        # elif non_clone and not clone:
        #     return 0
        # else:
        #     print(f"Warning: The text '{text}' does not contain a clear 'yes' or 'no' conclusion.")
        #     raise ValueError("The text does not contain a clear 'yes' or 'no' conclusion.")

    
    def _extract_ground_truth_labels(self):
        samples_label = [{sample['index']: sample['label']} for sample in self.test_data]
        return samples_label
    
    def compute_metrics(self, ouput_dir, description = None, save_to_file=False):
        description = description.replace("/", "_")
        print(f"the file name is {description}")
        ground_truth_labels = []
        predictions_labels = []
        unprocessed_samples = 0
        for element in self.ground_truth:
            key = list(element.keys())[0]

            if key in self.predicted_results:
                ground_truth_labels.append(element[key])
                predictions_labels.append(self.predicted_results[key])
            else:
                unprocessed_samples = unprocessed_samples + 1
                     
        missed_samples = []             
        self.precision = precision_score(ground_truth_labels, predictions_labels)
        self.recall = recall_score(ground_truth_labels, predictions_labels)
        self.f1_score = f1_score(ground_truth_labels, predictions_labels) 
        self.response_rate = len(predictions_labels)/ 1000 #Fix to calculate dynamically
        if save_to_file:
            self.write_results_to_file(ouput_dir, description)
            
    def write_results_to_file(self, output_dir, description):
        file_description_text = f"{description}\n\n"
        file_description_text = file_description_text + f"F1 score: {self.f1_score}\n"
        file_description_text = file_description_text + f"Precision: {self.precision}\n"
        file_description_text = file_description_text + f"Recall: {self.recall}\n"
        file_description_text = file_description_text + f"Response Rate: {self.response_rate}\n"
        file_name = '_'.join(description.split(" "))+'.txt'

        print(f"the result metric locaiton: {os.path.join(output_dir, file_name)}")
        with open(os.path.join(output_dir, file_name), "w") as file:
            file.write(file_description_text)
        
        
    @property
    def predicted_results(self):
        """
        Returns the extracted results as a dictionary.

        Returns:
            dict: The extracted results as a dictionary.
        """
        return self.results
    
    @property
    def ground_truth(self):
        """
        Returns the extracted results as a dictionary.

        Returns:
            dict: The extracted results as a dictionary.
        """
        return self._extract_ground_truth_labels()
    
    

if __name__ == '__main__':
    # analyser = Analyser(
    #     os.path.join(current_location, 'python_java_clones.jsonl'),
    #     os.path.join(current_location, 'results', 'python_java_results_deepseek.jsonl')
    # )
    # analyser.compute_metrics("Python-Java Clones DeepSeek Reasoning results", True)
    # analyser.compute_missing_samples(type='python_java_deepseek')

    # analyser = Analyser(
    #     os.path.join(current_location, 'python_java_clones.jsonl'),
    #     os.path.join(current_location, 'offline_results', 'python-java-mini-v2-main-prompt.jsonl')
    # )
    # analyser.compute_metrics("Python-Java Clones phi3 Reasoning results v2", True)
    # analyser.compute_missing_samples(type='python_java_deepseek')

    # analyser = Analyser(
    #     os.path.join(current_location, 'python_java_clones.jsonl'),
    #     os.path.join(current_location, 'offline_results', 'python-java-mini-v2-main-prompt.jsonl')
    # )
    # analyser.compute_metrics("Same Distribution Test All Clones phi3 Reasoning results new", True)
    # analyser.compute_missing_samples(type='same_dist_deepseek')
    # analyser = Analyser(
    #     os.path.join(current_location, 'extended-experiments/train-test-data/different_distribution_test_clones.jsonl'),
    #     os.path.join(current_location, 'offline_results', 'test-different-dist-main-prompt.jsonl')
    # )
    # analyser.compute_metrics("Different Distribution Test All Clones Phi3 pure results force conclusion ", True)
    # analyser.compute_missing_samples(type='diff_dist_phi3')

    # analyser = Analyser(
    #     os.path.join(current_location, 'extended-experiments/train-test-data/same_distribution_test_clones.jsonl'),
    #     os.path.join(current_location, 'offline_results', 'test-same-dist-main-prompt.jsonl')
    # )
    # analyser.compute_metrics("Same Distribution Test All Clones Phi3 pure results force conclusion ", True)
    # analyser.compute_missing_samples(type='same_dist_phi3')

    # analyser = Analyser(
    #     os.path.join(current_location, 'extended-experiments/train-test-data/same_distribution_test_clones.jsonl'),
    #     os.path.join(current_location, 'offline_results', 'test-same-dist-main-prompt.jsonl')
    # )
    # analyser.compute_metrics("Same Distribution Test All Clones Phi3 pure results ", True)
    # analyser.compute_missing_samples(type='same_dist_phi3')
    

    # analyser = Analyser(
    #     os.path.join(current_location, 'extended-experiments/train-test-data/same_distribution_test_clones.jsonl'),
    #     os.path.join(current_location, 'offline_results', '.jsonl')
    # )
    # analyser.compute_metrics("Same Distribution Test All Clones Phi3 pure results ", True)
    # analyser.compute_missing_samples(type='same_dist_phi3')
    
    analyser = Analyser(
        os.path.join(current_location, 'extended-experiments/train-test-data/different_distribution_test_clones.jsonl'),
        os.path.join(current_location, 'offline_results', 'test-different-dist-main-prompt-force-conclusion.jsonl')
    )
    analyser.compute_metrics("Different Distribution Test All Clones Phi3 pure results - Force conclusion ", True)
    analyser.compute_missing_samples(type='different_dist_phi3')
