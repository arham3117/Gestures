# Import necessary libraries
import pandas as pd  # for reading CSV files
import numpy as np  # for numerical operations
from sklearn.ensemble import RandomForestClassifier  # our machine learning model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report  # for evaluating model
import matplotlib.pyplot as plt  # for creating plots
import seaborn as sns  # for nicer looking plots
import joblib  # for saving the trained model
import os  # for file operations


class GestureModelTrainer:
    """
    This class trains a Random Forest model to recognize hand gestures.
    It loads the preprocessed data, trains the model, and evaluates its performance.
    """

    def __init__(self, processed_data_dir='data/processed', model_dir='models'):
        """
        Initialize the trainer.
        Sets up folders for data and where to save the trained model.
        """
        self.processed_data_dir = processed_data_dir
        self.model_dir = model_dir
        # Create models folder if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        self.model = None  # will store our trained model

    def load_data(self):
        """
        Load the training and testing data from CSV files.
        Separates features (X) from labels (y).
        """
        # Read the CSV files we created in preprocessing
        train_df = pd.read_csv(os.path.join(self.processed_data_dir, 'train.csv'))
        test_df = pd.read_csv(os.path.join(self.processed_data_dir, 'test.csv'))

        # Split features and labels for training data
        X_train = train_df.drop('gesture', axis=1).values  # all columns except 'gesture'
        y_train = train_df['gesture'].values  # only the 'gesture' column

        # Split features and labels for testing data
        X_test = test_df.drop('gesture', axis=1).values
        y_test = test_df['gesture'].values

        return X_train, X_test, y_train, y_test

    def train(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Train the Random Forest model.
        n_estimators = number of decision trees (100 means 100 trees)
        max_depth = how deep each tree can be (20 prevents overfitting)
        """
        # Load the data
        X_train, X_test, y_train, y_test = self.load_data()

        print("Training Random Forest Classifier...")
        print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}")

        # Create the Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,  # number of trees
            max_depth=max_depth,  # maximum tree depth
            random_state=random_state,  # for reproducible results
            n_jobs=-1  # use all CPU cores for faster training
        )

        # Train the model on training data
        self.model.fit(X_train, y_train)

        print("\nTraining completed!")

        # Check how well the model performs
        self.evaluate(X_train, y_train, X_test, y_test)

        return self.model

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluate how well the model performs.
        Tests it on both training and testing data.
        Prints accuracy, precision, recall, and F1 score.
        """
        # Check performance on training data
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Check performance on testing data (this is the important one!)
        y_test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)  # how many it got right
        test_precision = precision_score(y_test, y_test_pred, average='weighted')  # how many predictions were correct
        test_recall = recall_score(y_test, y_test_pred, average='weighted')  # how many actual gestures it found
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')  # balance of precision and recall

        # Print all the results
        print("\n" + "=" * 50)
        print("Model Performance")
        print("=" * 50)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy:  {test_accuracy:.4f}")
        print(f"Precision:         {test_precision:.4f}")
        print(f"Recall:            {test_recall:.4f}")
        print(f"F1 Score:          {test_f1:.4f}")
        print("=" * 50)

        # Show detailed results for each gesture
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred))

        # Create confusion matrix (shows which gestures get confused with each other)
        cm = confusion_matrix(y_test, y_test_pred)
        self.plot_confusion_matrix(cm, self.model.classes_)

        # Show which features are most important
        self.plot_feature_importance()

    def plot_confusion_matrix(self, cm, class_names):
        """
        Create a confusion matrix visualization.
        Shows which gestures the model confuses with each other.
        Perfect diagonal means no confusion (all correct).
        """
        plt.figure(figsize=(10, 8))
        # Create heatmap with gesture names
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(self.model_dir, 'confusion_matrix.png')
        plt.savefig(output_path)
        print(f"\nConfusion matrix saved to {output_path}")
        plt.close()

    def plot_feature_importance(self, top_n=20):
        """
        Show which hand landmarks are most important for classification.
        Higher bars mean that landmark is more useful for recognizing gestures.
        """
        importances = self.model.feature_importances_
        # Get the top N most important features
        indices = np.argsort(importances)[::-1][:top_n]

        # Create bar chart
        plt.figure(figsize=(12, 6))
        plt.title(f'Top {top_n} Feature Importances')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), indices, rotation=45)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(self.model_dir, 'feature_importance.png')
        plt.savefig(output_path)
        print(f"Feature importance plot saved to {output_path}")
        plt.close()

    def save_model(self, filename='gesture_classifier.joblib'):
        """
        Save the trained model to a file so we can use it later.
        This way we don't have to retrain every time.
        """
        if self.model is None:
            print("No model to save. Train the model first.")
            return

        model_path = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, model_path)
        print(f"\nModel saved to {model_path}")

    def load_model(self, filename='gesture_classifier.joblib'):
        """
        Load a previously trained model from file.
        """
        model_path = os.path.join(self.model_dir, filename)
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return self.model


# This runs when you execute the script directly
if __name__ == "__main__":
    trainer = GestureModelTrainer()
    model = trainer.train(n_estimators=100, max_depth=20)
    trainer.save_model()