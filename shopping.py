import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    evidence = []
    labels = []
    months = {
        "jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "june": 5,
        "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11
    }

    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:

            # Organise data into types
            administrative = int(row["Administrative"])
            administrative_duration = float(row["Administrative_Duration"])
            informational = int(row["Informational"])
            informational_duration = float(row["Informational_Duration"])
            productrelated = int(row["ProductRelated"])
            productrelated_duration = float(row["ProductRelated_Duration"])
            bouncerates = float(row["BounceRates"])
            exitrates = float(row["ExitRates"])
            pagevalues = float(row["PageValues"])
            specialday = float(row["SpecialDay"])
            month = months[row["Month"].lower()]
            operatingsystems = int(row["OperatingSystems"])
            browser = int(row["Browser"])
            region = int(row["Region"])
            traffictype = int(row["TrafficType"])
            visitortype = 0 if row["VisitorType"].lower() == 'new_visitor' else 1
            weekend = int(row["Weekend"].lower() == "true")
            revenue = int(row["Revenue"].lower() == "true")


            # Create list of evidence for current row
            e = [
                administrative, 
                administrative_duration, 
                informational, 
                informational_duration, 
                productrelated, 
                productrelated_duration, 
                bouncerates, 
                exitrates, 
                pagevalues, 
                specialday, 
                month, 
                operatingsystems, 
                browser, 
                region, 
                traffictype, 
                visitortype, 
                weekend
            ]
            
            # Append these lists to their respective list
            evidence.append(e)
            labels.append([revenue])
    
    print(evidence, labels)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
