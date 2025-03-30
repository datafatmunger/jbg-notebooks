import unittest
import numpy as np
from qcfs.fs.qfs import QuantumFeatureSelection

class TestQuantumFeatureSelection(unittest.TestCase):
    def test_unitary_update(self):
        """Test the unitary_update method of QuantumFeatureSelection."""
        num_features = 4
        learning_rate = 0.1
        qfs = QuantumFeatureSelection(num_features, learning_rate)

        # Test measurement before update
        print("Testing initial measurement...")
        available_features = [0, 1, 2, 3]
        probs_before = qfs.measure(available_features, return_probabilities=True)
        self.assertEqual(len(probs_before), len(available_features), "Incorrect number of probabilities")
        self.assertTrue(np.isclose(np.sum(probs_before), 1.0), "Probabilities do not sum to 1")
        print("✅ Initial measurement test passed.")

        # Test unitary update with positive reward
        print("Testing unitary update with positive reward...")
        action = 1  # Example action
        reward = 1  # Positive reward
        qfs.unitary_update(action, reward)
        
        # Test measurement after update
        probs_after = qfs.measure(available_features, return_probabilities=True)
        self.assertEqual(len(probs_after), len(available_features), "Incorrect number of probabilities after update")
        self.assertTrue(np.isclose(np.sum(probs_after), 1.0), "Probabilities do not sum to 1 after update")
        print("✅ Unitary update test (positive reward) passed.")

        # Test unitary update with negative reward
        print("Testing unitary update with negative reward...")
        reward = -1  # Negative reward
        qfs.unitary_update(action, reward)
        
        # Test measurement after negative update
        probs_final = qfs.measure(available_features, return_probabilities=True)
        self.assertEqual(len(probs_final), len(available_features), "Incorrect number of probabilities after negative update")
        self.assertTrue(np.isclose(np.sum(probs_final), 1.0), "Probabilities do not sum to 1 after negative update")
        print("✅ Unitary update test (negative reward) passed.")

        print("\n🎉 All unitary_update tests passed successfully!")

    def test_quantum_feature_selection(self):
        """Test the QuantumFeatureSelection class with quantum operations."""
        num_features = 5
        learning_rate = 0.1
        qfs = QuantumFeatureSelection(num_features, learning_rate)

        # Test initialization and basic properties
        print("Testing initialization...")
        self.assertEqual(qfs.num_features, num_features, "Incorrect number of features")
        self.assertEqual(len(qfs.decision_register), qfs.num_qubits, "Incorrect decision register size")
        print("✅ Initialization test passed.")

        # Test measurement
        print("Testing measurement...")
        available_features = list(range(num_features))
        print("Available features:", available_features)
        
        # Test multiple measurements return valid features
        for _ in range(10):
            action = qfs.get_feature(available_features)
            self.assertIn(action, available_features, f"Invalid action selected: {action}")
        print("✅ Measurement test passed.")

        # Test probability distribution
        print("Testing probability distribution...")
        probs = qfs.measure(available_features, return_probabilities=True)
        self.assertEqual(len(probs), len(available_features), "Incorrect number of probabilities")
        self.assertTrue(np.isclose(np.sum(probs), 1.0), "Probabilities do not sum to 1")
        print("✅ Probability test passed.")

        # Test unitary update effect
        print("Testing unitary update effect...")
        action = 1  # Example action
        qfs.unitary_update(action, reward=1)
        new_probs = qfs.measure(available_features, return_probabilities=True)
        self.assertEqual(len(new_probs), len(available_features), "Incorrect number of probabilities after update")
        self.assertTrue(np.isclose(np.sum(new_probs), 1.0), "Probabilities do not sum to 1 after update")
        print("✅ Update effect test passed.")

        print("\n🎉 All tests passed successfully!")

if __name__ == '__main__':
    unittest.main()