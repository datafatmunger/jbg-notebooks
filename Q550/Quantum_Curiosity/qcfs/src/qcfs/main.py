import os
import shutil
import time
import pandas as pd

from pennylane import numpy as np

from qcfs.fs.qfs import QuantumFeatureSelection
from qcfs.learners.vqc import QuantumLearner
import tracemalloc  # Memory tracking

from qcfs.enums import QuantumBackend

# Global Parameters
N_features = 5
N_data = 1
experiment = 'test'
number_of_experiment = 1
location = '../Datasets/adult'
outputlocation = 'Datasets'
file = 'adult'
filename = file + '_int.csv'
learning_rate = 0.005
discount_factor = 0.01
epsilon = 0.1
episode_size = 10
internal_trashold = 0
external_trashold = 0

backend = QuantumBackend.DEFAULT

def setup_experiment_folder(experiment):
    """Set up the experiment folder."""
    if not os.path.exists(f'Experiments/{experiment}'):
        os.makedirs(f'Experiments/{experiment}')
    else:
        shutil.rmtree(f'Experiments/{experiment}')
        os.makedirs(f'Experiments/{experiment}')

    with open(f'Experiments/{experiment}/parameters.txt', "w") as text_file:
        text_file.write(f'experiment: {experiment}\n')
        text_file.write(f'number of experiments: {number_of_experiment}\n')
        text_file.write(f'file: {file}\n')
        text_file.write(f'episode size: {episode_size}\n')
        text_file.write(f'internal trashold: {internal_trashold}\n')
        text_file.write(f'external trashold: {external_trashold}\n')

def initialize_experiment_data():
    """Load and preprocess the dataset."""
    data = pd.read_csv(f'{location}/{filename}', index_col=0)
    size = int(N_data * len(data.index))
    data = data.sample(n=size)
    data = data.iloc[:, -N_features - 1:]
    return data

def available_actions(number_of_columns, columns, initial_state, current_state, threshold, exploration, qfs):
    """
    Quantum-inspired function to return all available actions in the given state.

    Args:
        number_of_columns (int): Total number of columns/features.
        columns (list): List of already selected columns.
        initial_state (int): The initial state.
        current_state (int): The current state.
        threshold (float): Threshold for filtering actions.
        exploration (int): Exploration mode (0 for exploitation, 1 for exploration).
        qfs (QuantumFeatureSelector): Quantum feature selector object.

    Returns:
        list: List of available actions.
    """
    # Define all possible columns
    all_columns = list(range(number_of_columns))

    # Convert any tensor values in columns to integers
    columns = [int(col) if hasattr(col, 'item') else int(col) for col in columns]
    initial_state = int(initial_state) if hasattr(initial_state, 'item') else int(initial_state)
    current_state = int(current_state) if hasattr(current_state, 'item') else int(current_state)

    # Exclude already selected columns and current state (if different from initial state)
    exclude = columns.copy()
    if current_state != initial_state:
        exclude.append(current_state)
    available_act = list(set(all_columns) - set(exclude))
    print(f"Available actions after excluding selected columns and states: {available_act}")

    # If no actions are available, return empty list
    if not available_act:
        return []

    # Quantum-inspired filtering
    if exploration == 0:  # Exploitation mode
        probabilities = qfs.measure(available_act, return_probabilities=True)
        print(f"Probabilities of actions: {probabilities}")
        filtered_act = [action for action, prob in zip(available_act, probabilities) if prob > threshold]
        # If filtering removes all actions, return original list in exploitation mode
        if not filtered_act and available_act:
            print("No actions above threshold, using all available actions")
            return available_act
        available_act = filtered_act
        print(f"Available actions after applying threshold: {available_act}")

    return available_act

def run_episode(learner, qfs, data, number_of_columns, initial_state, episodes_number, df):
    """Run a single episode of the experiment."""
    for i in range(int(episodes_number)):
        print(f"\n🔄 Starting Episode {i + 1}")
        episode_available_act = list(range(number_of_columns))
        episode_columns = []
        episode_last_error = 0.5
        episode_current_state = initial_state

        while episode_available_act:
            print(f"\n🟢 Available features: {episode_available_act}")

            # Determine exploration vs. exploitation
            exploration = exploration_explotation(epsilon)

            # Update the available actions list
            episode_available_act = available_actions(
                number_of_columns, episode_columns, initial_state, episode_current_state, internal_trashold, exploration, qfs
            )
            print(f"Final available actions: {episode_available_act}")

            # Ensure there are available features
            if not episode_available_act:
                print("❌ No available features left. Terminating episode.")
                break

            # Select next feature using quantum measurement
            try:
                # Single quantum measurement to select feature
                episode_action = qfs.get_feature(episode_available_act)
                print(f"🎯 Selected feature: {episode_action}")
            except ValueError as e:
                print(f"❌ Error during quantum measurement: {e}")
                break

            # Update the selected feature list
            episode_columns = update_columns(episode_action, episode_columns)
            print(f"📌 Updated selected features: {episode_columns}")

            # Prepare training dataset with selected features
            X_train_episode, X_test_episode, y_train_episode, y_test_episode = update_X_train_X_test(episode_columns, data)

            # Train the learner
            learner.fit(X_train_episode, y_train_episode, num_it=50, learning_rate=learning_rate)
            predictions = learner.predict(X_test_episode)

            # Calculate error and reward
            episode_error = 1 - np.mean(predictions == y_test_episode)
            episode_reward = episode_last_error - episode_error
            print(f"📉 Model error: {episode_error}")
            print(f"🏆 Reward: {episode_reward}")

            # Update quantum circuit based on reward
            qfs.unitary_update(episode_action, episode_reward)
            episode_current_state = episode_action
            episode_last_error = episode_error

        # Calculate policy and accuracies
        policy_columns = calculate_policy(number_of_columns, initial_state, qfs)
        policy_accuracy_train, policy_accuracy_test = evaluate_policy(
            learner, policy_columns, data, episode_size
        )

        # Save episode results
        df.loc[len(df)] = {
            'episode': str(i + 1),
            'episode_columns': str(episode_columns),
            'policy_columns': str(policy_columns),
            'policy_accuracy_train': policy_accuracy_train,
            'policy_accuracy_test': policy_accuracy_test
        }

        print(f"\n✅ Episode {i + 1} completed")

def prepare_training_data(data, selected_columns):
    """Prepare training and testing data based on selected columns."""
    X = data.iloc[:, selected_columns]
    y = data.iloc[:, -1]
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

def update_X_train_X_test(columns, data):
    """Update the training and testing datasets based on selected columns."""
    X = data.iloc[:, columns]
    y = data.iloc[:, -1]
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

def calculate_policy(number_of_columns, initial_state, qfs):
    """Calculate the policy based on the quantum-inspired feature selection."""
    policy_columns = []
    policy_available_actions = available_actions(
        number_of_columns, policy_columns, initial_state, initial_state, internal_trashold, 0, qfs
    )
    policy_current_state = initial_state

    while policy_available_actions:  # Changed condition to be more Pythonic
        try:
            probabilities = qfs.measure(policy_available_actions, return_probabilities=True)
            print(f"🔎 Policy Probabilities: {probabilities}")

            # Select the next action using quantum measurement
            policy_action = qfs.measure(policy_available_actions)
            print(f"🎯 Selected action: {policy_action}")

            # Update the selected columns
            policy_columns = update_columns(policy_action, policy_columns)
            print(f"📌 Updated policy columns: {policy_columns}")

            # Update the available actions
            policy_available_actions = available_actions(
                number_of_columns, policy_columns, initial_state, policy_action, internal_trashold, 0, qfs
            )
            print(f"Policy available actions: {policy_available_actions}")

        except (ValueError, ZeroDivisionError) as e:
            print(f"❌ Error during policy calculation: {e}")
            break

    return policy_columns

def evaluate_policy(learner, policy_columns, data, episode_size):
    """Evaluate the policy by calculating train and test accuracies."""
    if not policy_columns:
        return 0, 0

    # Prepare training and testing data
    X_train, X_test, y_train, y_test = prepare_training_data(data, policy_columns)

    # Train the learner on the training data
    learner.fit(X_train, y_train, num_it=50, learning_rate=learning_rate)

    # Calculate train accuracy
    train_predictions = learner.predict(X_train)
    policy_accuracy_train = np.mean(train_predictions == y_train)

    # Calculate test accuracy
    test_predictions = learner.predict(X_test)
    policy_accuracy_test = np.mean(test_predictions == y_test)

    return policy_accuracy_train, policy_accuracy_test

def exploration_explotation(epsilon):
    """
    Decide whether to explore or exploit based on the epsilon value.

    Args:
        epsilon (float): Probability of choosing exploration (value between 0 and 1).

    Returns:
        int: 1 for exploration, 0 for exploitation.
    """
    if np.random.rand() < epsilon:  
        return 1  # Exploration
    else:
        return 0  # Exploitation

def update_columns(action, columns):
    """Add the selected action to the list of columns."""
    if action not in columns:
        columns.append(action)
    return columns

def main():
    print("Welcome to the Quantum Experiment Framework!")

    # Set up experiment folder
    setup_experiment_folder(experiment)

    # Load and preprocess data
    data = initialize_experiment_data()

    # Experiment setup
    number_of_columns = data.shape[1] - 1
    initial_state = number_of_columns
    episodes_number = 10 * len(data.index) / episode_size
    learner = QuantumLearner(num_layers=2)
    qfs = QuantumFeatureSelection(num_features=number_of_columns)
    df = pd.DataFrame(columns=('episode', 'episode_columns', 'policy_columns', 'policy_accuracy_train', 'policy_accuracy_test'))

    # Run experiments
    for e in range(number_of_experiment):
        print(f"\n🔬 Starting Experiment {e + 1}")
        run_episode(learner, qfs, data, number_of_columns, initial_state, episodes_number, df)

    # Save results
    df.to_csv(f'Experiments/{experiment}/results.csv', index=False)
    print("✅ Experiment completed successfully!")

if __name__ == "__main__":
    main()