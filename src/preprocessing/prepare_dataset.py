# Import necessary libraries
import json  # for reading JSON files
import os  # for file operations
import numpy as np  # for working with arrays
import pandas as pd  # for creating CSV files
from sklearn.model_selection import train_test_split  # for splitting data


class DatasetPreprocessor:
    """
    This class processes the raw gesture data and prepares it for training.
    It loads all the JSON files we collected and splits them into training and testing sets.
    """

    def __init__(self, raw_data_dir='data/raw', processed_data_dir='data/processed'):
        """
        Initialize the preprocessor.
        Sets up folders where raw data is and where processed data will go.
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        # Create the processed data folder if it doesn't exist
        os.makedirs(processed_data_dir, exist_ok=True)

    def load_raw_data(self):
        """
        Load all JSON files from the raw data folder.
        Reads each file and extracts the landmarks and gesture name.
        Returns two arrays: one with features (X) and one with labels (y).
        """
        data = []  # will store all the landmark data
        labels = []  # will store all the gesture names

        # Get list of all JSON files in the raw data folder
        json_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.json')]

        print(f"Loading {len(json_files)} samples...")

        # Read each JSON file
        for filename in json_files:
            filepath = os.path.join(self.raw_data_dir, filename)

            # Open and read the JSON file
            with open(filepath, 'r') as f:
                sample = json.load(f)

            # Extract the landmarks (63 numbers) and gesture name
            data.append(sample['landmarks'])
            labels.append(sample['gesture'])

        # Convert lists to numpy arrays for easier processing
        return np.array(data), np.array(labels)

    def prepare_dataset(self, test_size=0.2, random_state=42):
        """
        Main function that prepares the dataset for training.
        Loads data, splits it into train/test sets, and saves as CSV files.
        80% of data goes to training, 20% goes to testing.
        """
        # Load all the raw data
        X, y = self.load_raw_data()

        # Print some info about the dataset
        print(f"\nDataset shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of samples: {X.shape[0]}")

        # Show how many samples we have for each gesture
        unique, counts = np.unique(y, return_counts=True)
        print("\nClass distribution:")
        for gesture, count in zip(unique, counts):
            print(f"  {gesture}: {count} samples")

        # Split the data into training and testing sets
        # test_size=0.2 means 20% for testing, 80% for training
        # stratify=y makes sure each gesture is split evenly
        # random_state=42 makes the split reproducible (same every time)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")

        # Save training data to CSV
        train_df = pd.DataFrame(X_train)  # convert to dataframe
        train_df['gesture'] = y_train  # add gesture labels as last column
        train_df.to_csv(os.path.join(self.processed_data_dir, 'train.csv'), index=False)

        # Save testing data to CSV
        test_df = pd.DataFrame(X_test)
        test_df['gesture'] = y_test
        test_df.to_csv(os.path.join(self.processed_data_dir, 'test.csv'), index=False)

        print(f"\nDatasets saved to {self.processed_data_dir}/")

        return X_train, X_test, y_train, y_test


# This runs when you execute the script directly
if __name__ == "__main__":
    preprocessor = DatasetPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_dataset()
