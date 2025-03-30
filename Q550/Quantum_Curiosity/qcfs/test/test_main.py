import unittest
import os
import shutil
import pandas as pd
import numpy as np

from unittest.mock import MagicMock

from qcfs.main import (
    setup_experiment_folder,
    initialize_experiment_data,
    available_actions,
    prepare_training_data,
    update_X_train_X_test,
    calculate_policy,
    evaluate_policy,
    exploration_explotation,
    update_columns,
    N_features,
)

class TestMainFunctions(unittest.TestCase):

    def test_setup_experiment_folder(self):
        """Test that the experiment folder is created and parameters.txt is written."""
        experiment_name = "test_experiment"
        setup_experiment_folder(experiment_name)

        # Check if the folder is created
        self.assertTrue(os.path.exists(f'../../Experiments/{experiment_name}'))

        # Check if parameters.txt exists
        self.assertTrue(os.path.exists(f'../../Experiments/{experiment_name}/parameters.txt'))

        # Clean up after the test
        shutil.rmtree(f'../../Experiments/{experiment_name}')

    def test_initialize_experiment_data(self):
        """Test that the dataset is loaded and preprocessed correctly."""
        # Create the directory if it doesn't exist
        os.makedirs("../../Datasets/adult", exist_ok=True)

        # Create a mock dataset with N_features + 1 columns (features + label)
        num_features = N_features  # Use the imported N_features
        features = {f"feature{i}": [i*j for j in range(1, 6)] for i in range(1, num_features + 1)}
        features["label"] = [0, 1, 0, 1, 0]  # Add label column
        mock_data = pd.DataFrame(features)
        mock_data.to_csv("../../Datasets/adult/adult_int.csv", index=False)

        # Call the function
        data = initialize_experiment_data()

        # Check if the data is loaded and preprocessed correctly
        self.assertIsInstance(data, pd.DataFrame)
        expected_columns = num_features + 1  # N features + 1 label column
        self.assertEqual(data.shape[1], expected_columns, 
                        f"Expected {expected_columns} columns (N_features={num_features} + 1 label), got {data.shape[1]}")

        # Clean up after the test
        os.remove("../../Datasets/adult/adult_int.csv")
        os.rmdir("../../Datasets/adult")

    def test_available_actions(self):
        """Test that available actions are correctly calculated."""
        number_of_columns = 7
        selected_columns = [0, 2, 3, 5, 6]  # Columns to exclude
        initial_state = 5
        current_state = 3
        threshold = 0.5
        exploration = 0  # Exploitation mode

        # Mock the quantum feature selector
        class MockQuantumFeatureSelector:
            def measure(self, actions, return_probabilities=False):
                if return_probabilities:
                    # Mock probabilities for actions
                    return [0.6 if action == 1 else 0.4 for action in actions]
                return actions[0]  # Always return the first action

        qfs = MockQuantumFeatureSelector()

        # Call the function
        actions = available_actions(number_of_columns, selected_columns, initial_state, current_state, threshold, exploration, qfs)

        # Check if the correct actions are returned
        expected_actions = [1]  # Only action 1 has a probability > threshold
        self.assertListEqual(sorted(actions), sorted(expected_actions))

        # Test exploration mode
        exploration = 1  # Exploration mode
        actions = available_actions(number_of_columns, selected_columns, initial_state, current_state, threshold, exploration, qfs)

        # In exploration mode, all valid actions should be returned
        expected_actions = [1, 4]
        self.assertListEqual(sorted(actions), sorted(expected_actions))

    def test_prepare_training_data(self):
        """Test that training and testing data are prepared correctly."""
        data = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [6, 7, 8, 9, 10],
            "label": [0, 1, 0, 1, 0]
        })
        selected_columns = [0, 1]
        X_train, X_test, y_train, y_test = prepare_training_data(data, selected_columns)

        # Check the split sizes
        self.assertEqual(len(X_train), 4)
        self.assertEqual(len(X_test), 1)
        self.assertEqual(len(y_train), 4)
        self.assertEqual(len(y_test), 1)

    def test_update_X_train_X_test(self):
        """Test that training and testing datasets are updated correctly."""
        data = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [6, 7, 8, 9, 10],
            "label": [0, 1, 0, 1, 0]
        })
        columns = [0, 1]
        X_train, X_test, y_train, y_test = update_X_train_X_test(columns, data)

        # Check the split sizes
        self.assertEqual(len(X_train), 4)
        self.assertEqual(len(X_test), 1)
        self.assertEqual(len(y_train), 4)
        self.assertEqual(len(y_test), 1)

    def test_calculate_policy(self):
        """Test that the policy is calculated correctly."""
        def mock_measure(actions, return_probabilities=False):
            if return_probabilities:
                return [1 / len(actions)] * len(actions)  # Mock uniform probabilities
            return actions[0]  # Always return the first available action

        qfs = MagicMock()
        qfs.measure = MagicMock(side_effect=mock_measure)  # Mock the measure method
        number_of_columns = 3
        initial_state = 0

        policy_columns = calculate_policy(number_of_columns, initial_state, qfs)

        # Sort the policy columns before comparison
        self.assertListEqual(sorted(policy_columns), [0, 1, 2])

    def test_evaluate_policy(self):
        """Test that the policy is evaluated correctly."""
        class MockLearner:
            def fit(self, X, y, num_it, learning_rate):
                pass

            def predict(self, X):
                return np.array([1] * len(X))  # Return predictions matching the input length

        learner = MockLearner()
        data = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],
            "label": [1, 1, 1, 1]  # Ensure labels match predictions
        })
        policy_columns = [0, 1]
        train_acc, test_acc = evaluate_policy(learner, policy_columns, data, episode_size=2)

        # Check if the accuracies are correct
        self.assertEqual(train_acc, 1.0)
        self.assertEqual(test_acc, 1.0)

    def test_exploration_explotation(self):
        """Test the exploration vs. exploitation decision."""
        np.random.seed(42)  # Set seed for reproducibility
        epsilon = 0.5

        # Call the function multiple times
        results = [exploration_explotation(epsilon) for _ in range(10)]

        # Check if the results are either 0 or 1
        self.assertTrue(all(r in [0, 1] for r in results))

    def test_update_columns(self):
        """Test that columns are updated correctly."""
        columns = [0, 1]
        action = 2
        updated_columns = update_columns(action, columns)

        # Check if the action is added to the columns
        self.assertListEqual(updated_columns, [0, 1, 2])

if __name__ == '__main__':
    unittest.main()