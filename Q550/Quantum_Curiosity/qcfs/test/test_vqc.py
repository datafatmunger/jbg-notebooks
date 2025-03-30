import unittest
import numpy as np
from qcfs.learners.vqc import QuantumLearner

class TestQuantumLearner(unittest.TestCase):
    def test_quantum_learner(self):
        """Test the QuantumLearner class with basic training and prediction."""
        np.random.seed(42)

        # Sample training data (binary classification, {-1, 1} labels)
        X_train = np.array([
            [0.1, 0.2], 
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8]
        ])
        y_train = np.array([-1, 1, -1, 1])

        # Initialize learner
        num_features = X_train.shape[1]
        ql = QuantumLearner(num_features)

        # Train the model
        ql.fit(X_train, y_train, num_it=50, learning_rate=0.1)

        # Make predictions
        predictions = ql.predict(X_train)

        # Validate predictions
        self.assertEqual(predictions.shape, (X_train.shape[0],), "Prediction output shape mismatch.")
        self.assertTrue(all(pred in [-1, 1] for pred in predictions), "Predictions are not binary labels.")

        # Test with new data
        X_test = np.array([[0.2, 0.3], [0.6, 0.7]])
        predictions = ql.predict(X_test)
        self.assertEqual(predictions.shape, (X_test.shape[0],), "Test prediction output shape mismatch.")
        self.assertTrue(all(pred in [-1, 1] for pred in predictions), "Test predictions are not binary labels.")

    def test_quantum_learner_multiple_episodes(self):
        """Test the QuantumLearner class with multiple episodes to observe improvement."""
        np.random.seed(42)

        # Sample training data
        X_train = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [0.2, 0.3, 0.4]
        ])
        y_train = np.array([-1, 1, -1, 1])

        # Initialize learner
        num_features = X_train.shape[1]
        learning_rate = 0.1
        epsilon = 0.3  # Exploration rate
        num_episodes = 3
        ql = QuantumLearner(num_features)

        for episode in range(num_episodes):
            print(f"\n🔄 Episode {episode + 1} start")

            # Simulate feature selection process
            selected_features = []
            available_features = list(range(X_train.shape[1]))
            last_error = 0.5  # Initialize error

            while available_features:
                # Exploration vs. exploitation
                explore = np.random.rand() < epsilon
                if explore:
                    action = np.random.choice(available_features)  # Randomly select a feature
                    print(f"🧭 Exploration: Selected feature {action}")
                else:
                    action = available_features[0]  # Exploit (select the first available feature)
                    print(f"📈 Exploitation: Selected feature {action}")

                # Update selected features and remove from available features
                selected_features.append(action)
                available_features.remove(action)

                # Simulate training with selected features
                X_train_selected = X_train[:, selected_features]

                # Pad the input to match the expected number of features
                num_features = len(selected_features)
                X_train_padded = np.zeros((X_train_selected.shape[0], num_features))
                X_train_padded[:, :len(selected_features)] = X_train_selected

                # Train the learner
                ql.fit(X_train_padded, y_train, num_it=50, learning_rate=learning_rate)
                predictions = ql.predict(X_train_padded)
                print(f"Predictions: {predictions}")

                # Validate predictions
                self.assertEqual(predictions.shape, (X_train_padded.shape[0],), "Prediction output shape mismatch.")
                self.assertTrue(all(pred in [-1, 1] for pred in predictions), "Predictions are not binary labels.")

                # Calculate error and reward
                error = 1 - np.mean(predictions == y_train)
                reward = last_error - error
                print(f"Error: {error}, Reward: {reward}")

                # Update for next iteration
                last_error = error

            # Validate final state
            self.assertGreater(len(selected_features), 0, "No features were selected.")
            print(f"🎯 Episode {episode + 1} complete. Selected features: {selected_features}")

if __name__ == '__main__':
    unittest.main()