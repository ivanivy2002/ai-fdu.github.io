import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from fea import feature_extraction

from Bio.PDB import PDBParser

TASK_DETAIL = False
# TASK_DETAIL = True


class SVMModel:
    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVC(kernel=kernel, C=C, probability=True)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        return self.model.score(data, targets)


class LRModel:
    # todo: Implement Logistic Regression model
    """
        Initialize Logistic Regression (from sklearn) model.

        Parameters:
        - C (float): Inverse of regularization strength; must be a positive float. Default is 1.0.
    """

    def __init__(self, C=1.0):
        self.model = LogisticRegression(C=C, max_iter=1000)

    """
        Train the Logistic Regression model.

        Parameters:
        - train_data (array-like): Training data.
        - train_targets (array-like): Target values for the training data.
    """

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    """
        Evaluate the performance of the Logistic Regression model.

        Parameters:
        - data (array-like): Data to be evaluated.
        - targets (array-like): True target values corresponding to the data.

        Returns:
        - float: Accuracy score of the model on the given data.
    """

    def evaluate(self, data, targets):
        return self.model.score(data, targets)


class LinearSVMModel:
    # todo
    """
        Initialize Linear SVM (from sklearn) model.

        Parameters:
        - C (float): Inverse of regularization strength; must be a positive float. Default is 1.0.
    """

    def __init__(self, C=1.0):
        self.model = LinearSVC(C=C, max_iter=1000)

    """
        Train and Evaluate are the same.
    """

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        return self.model.score(data, targets)


class LinearSVMModelHand:
    # 手写的线性SVM
    def __init__(self, C=1.0):
        self.C = C
        self.model = None
        
    def train(self, train_data, train_targets):
        self.model = self.fit(train_data, train_targets)
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        alpha = np.zeros(n_samples)
        beta = 0
        lr = 0.0001
        max_iter = 1000
        for _ in range(max_iter):
            for i in range(n_samples):
                if y[i] * (np.dot(X[i], X.T.dot(alpha * y)) + beta) <= 0:
                    alpha[i] += lr
                    beta += lr * y[i]
        return alpha, beta
    
    def predict(self, X):
        return np.sign(np.dot(X, X.T.dot(self.model[0] * y)) + self.model[1])
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def evaluate(self, data, targets):
        return self.score(data, targets)
    

def data_preprocess(args):
    if args.ent == 'Atom':
        # diagrams = feature_extraction()[0]
        diagrams = np.load('./data/atom_matrix.npy')
    elif args.ent == 'Residue':
        # diagrams = residue_extraction()[0]
        diagrams = np.load('./data/residue_matrix.npy')
    elif args.ent == 'Chain':
        # diagrams = chain_extraction()[0]
        # 直接加载fea后的数据
        diagrams = np.load('./data/chain_matrix.npy')
    else:
        diagrams = np.load('./data/diagrams.npy')
    diagrams_row, diagrams_col = diagrams.shape
    # print("Diagrams shape:", diagrams.shape) # 1357*300 (1357 samples, 300 features)
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')  # Load the cast file
    # print("Cast shape:", cast.shape)  # 1357*56 (1357 samples, 56 tasks)
    cast.columns.values[0] = 'protein'  # Rename the first column to 'protein'
    data_list = []
    target_list = []
    for task in range(1, 56):  # Assuming only one task for now
        task_col = cast.iloc[:, task]  # Get the task column
        ## todo: Try to load data/target
        # 利用diagrams的特征维度数据，处理得到train_data和test_data
        # 利用cast的标签数据，处理train_targets和test_targets
        # train_data, test_data, train_targets, test_targets = train_test_split(diagrams, cast.iloc[:, 1:], test_size=0.2, random_state=42)
        train_data = []
        test_data = []
        train_targets = []
        test_targets = []
        for i in range(diagrams_row):
            if task_col[i] == 1:
                train_data.append(diagrams[i])
                train_targets.append(True)
            elif task_col[i] == 2:
                train_data.append(diagrams[i])
                train_targets.append(False)
            elif task_col[i] == 3:
                test_data.append(diagrams[i])
                test_targets.append(True)
            elif task_col[i] == 4:
                test_data.append(diagrams[i])
                test_targets.append(False)
            else:
                raise ValueError("Unknown tag")
        data_list.append((train_data, test_data))
        target_list.append((train_targets, test_targets))
    return data_list, target_list


def main(args):
    path = './out/'

    data_list, target_list = data_preprocess(args)

    task_acc_train = []  # List to store training accuracy for each task
    task_acc_test = []  # List to store testing accuracy for each task

    # Model Initialization based on input argument
    path += f"{args.model_type}/"
    path += f"{args.model_type}_C{args.C}"
    if args.model_type == 'svm':
        model = SVMModel(kernel=args.kernel, C=args.C)
        path += f"_{args.kernel}"
    else:
        print("Attention: Kernel option is not supported")
        if args.model_type == 'linear_svm':
            model = LinearSVMModel(C=args.C)
        elif args.model_type == 'lr':
            model = LRModel(C=args.C)
        else:
            raise ValueError("Unsupported model type")
    path += '.txt'
    with open(path, 'a') as f:
        start_time = pd.Timestamp.now()
        for i in range(len(data_list)): # For each task
            train_data, test_data = data_list[i]
            train_targets, test_targets = target_list[i]
            if TASK_DETAIL:
                print(f"Processing dataset {i + 1}/{len(data_list)}")
            f.write(f"Processing dataset {i + 1}/{len(data_list)}\n")
            # Train the model
            model.train(train_data, train_targets)

            # Evaluate the model
            train_accuracy = model.evaluate(train_data, train_targets)
            test_accuracy = model.evaluate(test_data, test_targets)
            if TASK_DETAIL:
                print(f"Dataset {i + 1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
            f.write(f"Dataset {i + 1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}\n")
            task_acc_train.append(train_accuracy)
            task_acc_test.append(test_accuracy)
        end_time = pd.Timestamp.now()
        cost_time = end_time - start_time
        f.write(args.__str__() + '\n')
        task_acc_train_avg = sum(task_acc_train) / len(task_acc_train)
        task_acc_test_avg = sum(task_acc_test) / len(task_acc_test)
        f.write(f"Training accuracy: {task_acc_train_avg}\n")
        f.write(f"Testing accuracy: {task_acc_test_avg}\n")
        f.write(f"Time taken: {cost_time}\n")
        f.write("--------------------------------------------------\n")
        print(args.__str__() + '\n')
        print("Training accuracy:", task_acc_train_avg)
        print("Testing accuracy:", task_acc_test_avg)
        print("Time taken:", cost_time)
        print("--------------------------------------------------")
    return task_acc_train_avg, task_acc_test_avg, cost_time, task_acc_train, task_acc_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Model Training and Evaluation")
    parser.add_argument('--model_type', type=str, default='svm', choices=['svm', 'linear_svm', 'lr'], help="Model type")
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'],
                        help="Kernel type")
    parser.add_argument('--C', type=float, default=20, help="Regularization parameter")
    # parser.add_argument('--ent', action='store_true',
    #                     help="Load data from a file using a feature engineering function feature_extraction() from fea.py")
    parser.add_argument('--ent', type=str, default='false', choices=['Atom','Residue','Chain','False'],
                        help="Load data from a file using a feature engineering function feature_extraction() from fea.py")
    args = parser.parse_args()
    main(args)
