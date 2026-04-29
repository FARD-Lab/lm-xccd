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
        self.report_file = report_file
        self.test_data = self.read_data(test_file)
        self.results = self._extract_results()

    def compute_missing_samples(self, type, output_dir=None):
        """
        Writes missing/correct sample indices to output_dir.
        Defaults to current_location/offline_results to match the second script.
        """
        if output_dir is None:
            output_dir = os.path.join(current_location, "offline_results")
        os.makedirs(output_dir, exist_ok=True)

        missing_ids = []
        correct_ids = []
        for sample in self.ground_truth:
            sample_key = list(sample.keys())[0]
            if sample_key in self.predicted_results:
                if self.predicted_results[sample_key] != sample[sample_key]:
                    missing_ids.append(sample_key)
                else:
                    correct_ids.append(sample_key)

        with open(os.path.join(output_dir, f"{type}_missing_index.txt"), "w") as file:
            for id in missing_ids:
                file.write(f"{id}\n")

        with open(os.path.join(output_dir, f"{type}_correct_index.txt"), "w") as file:
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
        with open(data_file, "r") as f:
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
                data_id = data["idx"]
            except:
                assert 1 == 1

            clone_result = self.extract_llm_result(data["text"], data["final_conclusion"])
            if clone_result == -1:
                print(
                    f"Warning: The text for data_id {data_id} does not contain a clear 'yes' or 'no' conclusion."
                )
                continue
            result[data_id] = clone_result

        return result

    def extract_llm_result(self, text, conclusion):
        text = text.split("|assistant|")[1].lower()  # take the last part after the assistant tag
        keys = ["functionality_", "conclusion"]

        idx = text.rfind(keys[0])
        idx2 = text.rfind(keys[1])  # find *last* occurrence
        if idx == -1 and idx2 == -1:
            if conclusion.lower() == "yes":
                return 1
            if conclusion.lower() == "no":
                return 0

        if idx or idx2:
            text = text.split(keys[0]) if idx else text.split(keys[1])
            if len(text) < 2:
                if conclusion.lower() == "yes":
                    return 1
                else:
                    return 0

            tail = text[1][:100]
            non_clone = bool(re.search(r"\bno\b", tail, flags=re.IGNORECASE))
            clone = bool(re.search(r"\byes\b", tail, flags=re.IGNORECASE))
            if clone and not non_clone:
                return 1
            elif non_clone and not clone:
                return 0
            else:
                if conclusion.lower() == "yes":
                    return 1
                else:
                    return 0
                print(f"Warning: The text '{text}' does not contain a clear 'yes' or 'no' conclusion.")
                return -1

    def _extract_ground_truth_labels(self):
        samples_label = [{sample["index"]: sample["label"]} for sample in self.test_data]
        return samples_label

    def compute_metrics(self, output_dir, description=None, save_to_file=False):
        """
        Matches the second script's output handling:
        - takes output_dir explicitly
        - sanitizes description to be a safe filename
        - writes metrics to output_dir
        """
        if description is None:
            description = "results"
        description = description.replace("/", "_")
        print(f"the file name is {description}")

        os.makedirs(output_dir, exist_ok=True)

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

        self.precision = precision_score(ground_truth_labels, predictions_labels)
        self.recall = recall_score(ground_truth_labels, predictions_labels)
        self.f1_score = f1_score(ground_truth_labels, predictions_labels)

        if save_to_file:
            self.write_results_to_file(output_dir, description)

    def write_results_to_file(self, output_dir, description):
        file_description_text = f"{description}\n\n"
        file_description_text = file_description_text + f"F1 score: {self.f1_score}\n"
        file_description_text = file_description_text + f"Precision: {self.precision}\n"
        file_description_text = file_description_text + f"Recall: {self.recall}\n"
        file_description_text = file_description_text + f"Response Rate: 1.0"
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


if __name__ == "__main__":
    analyser = Analyser(
        os.path.join(current_location, "extended-experiments/train-test-data/same_distribution_test_clones.jsonl"),
        os.path.join(
            current_location,
            "offline_results",
            "classifier-head-results",
            "test-same-distribution-phi3-contrastivehead.jsonl",
        ),
    )

    # output location calculated like the second script
    output_dir = os.path.join(current_location, "offline_results")

    analyser.compute_metrics(
        ouput_dir=output_dir,
        description="Same Distribution Test phi3 contrastive head ",
        save_to_file=True,
    )
    analyser.compute_missing_samples(type="same_dist_phi3_contrastive_head", output_dir=output_dir)