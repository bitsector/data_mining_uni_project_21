import argparse
import datetime
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


def init() -> argparse.Namespace:
    """
    Initialize and parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a decision tree classifier.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the CSV file containing the data.")
    parser.add_argument(
        "-c",
        "--criterion",
        type=str,
        choices=["gini", "entropy"],
        default="gini",
        help='Criterion for the decision tree: "gini" or "entropy" (default: "gini").',
    )
    parser.add_argument("-v", "--visualise", action="store_true", help="Visualise the decision tree graphically.")
    parser.add_argument(
        "-e", "--export", type=str, help="Export the data to an Excel file. Provide the filename with .xlsx extension."
    )
    parser.add_argument("-d", "--depth", type=int, default=3, help="Maximum depth of the tree.")
    parser.add_argument("-u", "--dump", action="store_true", help="Dump the structure of classifiers to the terminal.")
    return parser.parse_args()


def read_data(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load data from a CSV file using the provided arguments.
    """
    column_names = [
        "age",
        "sex",
        "on_thyroxine",
        "query_on_thyroxine",
        "on_antithyroid_medication",
        "sick",
        "pregnant",
        "thyroid_surgery",
        "i131_treatment",
        "query_hypothyroid",
        "query_hyperthyroid",
        "lithium",
        "goitre",
        "tumor",
        "hypopituitary",
        "psych",
        "tsh_measured",
        "tsh",
        "t3_measured",
        "t3",
        "tt4_measured",
        "tt4",
        "t4u_measured",
        "t4u",
        "fti_measured",
        "fti",
        "tbg_measured",
        "tbg",
        "referral_source",
        "result",
    ]
    data = pd.read_csv(args.input, header=None, names=column_names, usecols=range(len(column_names)))
    return data


def encode_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all categorical columns except 'age' using Label Encoding.
    """
    label_encoder = LabelEncoder()
    categorical_columns = [col for col in data.columns if col != "age"]

    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column])

    # Ensure 'age' is the first column
    age = data["age"]
    data.drop(labels=["age"], axis=1, inplace=True)
    data.insert(0, "age", age)

    return data


def clean_result(data: pd.DataFrame) -> None:
    """
    Clean the 'result' column by removing square brackets and their contents.
    Trim spaces from both sides of the cleaned strings.
    Convert "-" to "f" and any string of letters to "t".
    """
    data["result"] = data["result"].str.replace(r"\s*\[.*?\]\s*", "", regex=True).str.strip()
    data["result"] = data["result"].apply(lambda x: "f" if x == "-" else "t")

    # Count occurrences of "t" and "f"
    result_counts = data["result"].value_counts()
    total_count = len(data)
    t_count = result_counts.get("t", 0)
    f_count = result_counts.get("f", 0)
    t_percentage = (t_count / total_count) * 100
    f_percentage = (f_count / total_count) * 100

    # Print the results
    print(f"Total lines: {total_count}")
    print(f"Number of 't' values: {t_count} ({t_percentage:.2f}%)")
    print(f"Number of 'f' values: {f_count} ({f_percentage:.2f}%)")


def get_majorities(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the most common value for each column.
    """
    return data.mode().iloc[0]


def print_majorities(majority_values: pd.Series) -> None:
    """
    Print the most common values for each column.
    """
    print("Majority values for each column:")
    print(majority_values)


def clear_unknown_majorities_columns(data: pd.DataFrame, majority_values: pd.Series) -> None:
    """
    Identify columns where the majority value is '?' and delete them.
    """
    unknown_columns = [col for col in majority_values.index if majority_values[col] == "?"]
    if unknown_columns:
        print("Columns with '?' as the majority value will be deleted:")
        print(unknown_columns)
        data.drop(columns=unknown_columns, inplace=True)
    else:
        print("No columns with '?' as the majority value were found.")


def remove_columns_manually(column_names: List[str], data: pd.DataFrame) -> None:
    """
    Remove specified columns from the DataFrame.
    """
    data.drop(columns=column_names, inplace=True)
    print(f"Columns {column_names} have been removed from the data.")


def export_xlsx(data: pd.DataFrame, filename: str) -> None:
    """
    Export data to an Excel file.
    """
    data.to_excel(filename, index=False)
    print(f"Data exported to {filename}")


def trim_c45(
    classifier: DecisionTreeClassifier, X_train: pd.DataFrame, y_train: pd.Series, depth: int
) -> DecisionTreeClassifier:
    path = classifier.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[:-1]  # Exclude the maximum alpha

    clfs = []
    for ccp_alpha in ccp_alphas:
        if depth == 0:
            clf = DecisionTreeClassifier(criterion="entropy", ccp_alpha=ccp_alpha)
        else:
            clf = DecisionTreeClassifier(criterion="entropy", ccp_alpha=ccp_alpha, max_depth=depth)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    accuracies = [accuracy_score(y_train, clf.predict(X_train)) for clf in clfs]
    best_clf = clfs[np.argmax(accuracies)]
    return best_clf


def trim_cart(
    classifier: DecisionTreeClassifier, X_train: pd.DataFrame, y_train: pd.Series, depth: int
) -> DecisionTreeClassifier:
    path = classifier.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[:-1]  # Exclude the maximum alpha

    clfs = []
    for ccp_alpha in ccp_alphas:
        if depth == 0:
            clf = DecisionTreeClassifier(criterion="gini", ccp_alpha=ccp_alpha)
        else:
            clf = DecisionTreeClassifier(criterion="gini", ccp_alpha=ccp_alpha, max_depth=depth)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    accuracies = [accuracy_score(y_train, clf.predict(X_train)) for clf in clfs]
    best_clf = clfs[np.argmax(accuracies)]
    return best_clf


def trim_random_forest(
    classifier: RandomForestClassifier, X_train: pd.DataFrame, y_train: pd.Series, depth: int
) -> RandomForestClassifier:
    # Adjust the depth of each tree
    estimators = []
    for tree in classifier.estimators_:
        if depth == 0:
            clf = DecisionTreeClassifier(criterion="gini")
        else:
            clf = DecisionTreeClassifier(criterion="gini", max_depth=depth)
        clf.fit(X_train, y_train)
        estimators.append(clf)

    # Replace old estimators with new ones
    classifier.estimators_ = estimators
    return classifier


def train_c45(data: pd.DataFrame) -> DecisionTreeClassifier:
    X = data.drop("result", axis=1)
    y = data["result"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    c45_classifier = DecisionTreeClassifier(criterion="entropy")
    c45_classifier.fit(X_train, y_train)
    print(f"C4.5 Accuracy: {accuracy_score(y_test, c45_classifier.predict(X_test))}")
    return c45_classifier


def train_cart(data: pd.DataFrame) -> DecisionTreeClassifier:
    X = data.drop("result", axis=1)
    y = data["result"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cart_classifier = DecisionTreeClassifier(criterion="gini")
    cart_classifier.fit(X_train, y_train)
    print(f"CART Accuracy: {accuracy_score(y_test, cart_classifier.predict(X_test))}")
    return cart_classifier


def train_random_forest(data: pd.DataFrame) -> RandomForestClassifier:
    X = data.drop("result", axis=1)
    y = data["result"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_classifier.predict(X_test))}")
    return rf_classifier


def display_c45(c45_classifier, feature_names, tag):
    if c45_classifier is None:
        raise ValueError("Classifier is not trained.")
    start_time = time.time()
    plt.figure(figsize=(20, 10))
    plot_tree(
        c45_classifier, filled=True, feature_names=feature_names, class_names=["Class1", "Class2"]
    )  # Adjust class names as necessary
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig(f"out_c45_{tag}_{timestamp}.png")
    plt.close()
    end_time = time.time()
    print(f"Time taken to plot C4.5 tree: {end_time - start_time} seconds")


def display_cart(cart_classifier, feature_names, tag):
    if cart_classifier is None:
        raise ValueError("Classifier is not trained.")
    start_time = time.time()
    plt.figure(figsize=(20, 10))
    plot_tree(cart_classifier, filled=True, feature_names=feature_names, class_names=True)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig(f"out_cart_{tag}_{timestamp}.png")
    plt.close()
    end_time = time.time()
    print(f"Time taken to plot CART tree: {end_time - start_time} seconds")


def display_random_forest(rf_classifier, feature_names, tag, tree_index=0):
    if rf_classifier is None:
        raise ValueError("Classifier is not trained.")
    start_time = time.time()
    plt.figure(figsize=(20, 10))
    plot_tree(rf_classifier.estimators_[tree_index], filled=True, feature_names=feature_names, class_names=True)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig(f"out_rf_{tag}_{timestamp}.png")
    plt.close()
    end_time = time.time()
    print(f"Time taken to plot Random Forest tree: {end_time - start_time} seconds")


def dump_c45(c45_classifier, tag):
    print(f"C4.5 Classifier Structure ({tag}):")
    print(tree.export_text(c45_classifier))


def dump_cart(cart_classifier, tag):
    print(f"CART Classifier Structure ({tag}):")
    print(tree.export_text(cart_classifier))


def dump_random_forest(rf_classifier, tag):
    print(f"Random Forest Classifier Structure ({tag}):")
    for i, estimator in enumerate(rf_classifier.estimators_):
        print(f"Tree {i} Structure ({tag}):")
        print(tree.export_text(estimator))


def main() -> None:
    args = init()
    data = read_data(args)

    clean_result(data)  # Clean the 'result' column

    majority_values = get_majorities(data)
    print_majorities(majority_values)

    clear_unknown_majorities_columns(data, majority_values)

    # Call remove_columns_manually with specified columns
    columns_to_remove = [
        "tsh_measured",
        "t3_measured",
        "tt4_measured",
        "t4u_measured",
        "fti_measured",
        "tbg_measured",
        "referral_source",
    ]
    remove_columns_manually(columns_to_remove, data)

    data = encode_categorical_columns(data)

    # Train classification models
    c45_classifier = train_c45(data)
    cart_classifier = train_cart(data)
    rf_classifier = train_random_forest(data)

    # Display decision trees if visualise flag is set
    if args.visualise:
        feature_names = data.drop("result", axis=1).columns
        display_c45(c45_classifier, feature_names, "_pre_prune_")
        display_cart(cart_classifier, feature_names, "_pre_prune_")
        display_random_forest(
            rf_classifier, feature_names, "_pre_prune_", tree_index=0
        )  # Display the first tree as an example

    # Prune the classifiers
    depth = args.depth
    c45_classifier = trim_c45(c45_classifier, data.drop("result", axis=1), data["result"], depth)
    cart_classifier = trim_cart(cart_classifier, data.drop("result", axis=1), data["result"], depth)
    rf_classifier = trim_random_forest(rf_classifier, data.drop("result", axis=1), data["result"], depth)

    # Print depths and accuracy rates of pruned classifiers
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("result", axis=1), data["result"], test_size=0.2, random_state=42
    )

    c45_depth = c45_classifier.get_depth()
    c45_accuracy = accuracy_score(y_test, c45_classifier.predict(X_test))
    print(f"Pruned C4.5 Depth: {c45_depth}, Accuracy: {c45_accuracy}")

    cart_depth = cart_classifier.get_depth()
    cart_accuracy = accuracy_score(y_test, cart_classifier.predict(X_test))
    print(f"Pruned CART Depth: {cart_depth}, Accuracy: {cart_accuracy}")

    rf_depths = [tree.get_depth() for tree in rf_classifier.estimators_]
    rf_accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))
    print(f"Pruned Random Forest Depths: {rf_depths}, Accuracy: {rf_accuracy}")

    # Display decision trees if visualise flag is set
    if args.visualise:
        feature_names = data.drop("result", axis=1).columns
        display_c45(c45_classifier, feature_names, "_post_prune_")
        display_cart(cart_classifier, feature_names, "_post_prune_")
        display_random_forest(
            rf_classifier, feature_names, "_post_prune_", tree_index=0
        )  # Display the first tree as an example

    if args.dump:
        dump_c45(c45_classifier, "_post_prune_")
        dump_cart(cart_classifier, "_post_prune_")
        # dump_random_forest(rf_classifier,"_post_prune_")

    if args.export:
        export_xlsx(data, args.export)


if __name__ == "__main__":
    main()
