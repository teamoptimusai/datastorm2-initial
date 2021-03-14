from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from model import Multi_Model
from utils import prepare_training_data, prepare_data

data_train = pd.read_csv('Hotel-A-train.csv', index_col=0)

data_test = pd.read_csv('Hotel-A-validation.csv', index_col=0)
data_test['Reservation_Status'] = data_test['Reservation_Status'].map(
    {'Check-In': 0, 'Canceled': 1, 'No-Show': 2})
test_labels = np.array(data_test.pop('Reservation_Status'))

scaler = StandardScaler()
pca = PCA(0.85)

do_validation = False
create_submission = True

train1_features, train1_labels, train2_features, train2_labels, train3_features, train3_labels, val_features, val_labels, class_weight = prepare_training_data(
    data_train, scaler, pca, validation=do_validation)

model1 = DecisionTreeClassifier()
model2 = DecisionTreeClassifier()
model3 = DecisionTreeClassifier()

model = Multi_Model(model1, model2, model3)
model.fit(
    train1_features, train1_labels, train2_features, train2_labels, train3_features, train3_labels
)

if do_validation:
    val_predictions = model.predict(val_features)
    print(accuracy_score(val_labels, val_predictions))
    print(confusion_matrix(val_labels, val_predictions))
    print(classification_report(val_labels, val_predictions))

test_features = prepare_data(data_test, scaler, pca)

test_predictions = model.predict(test_features)
print(accuracy_score(test_labels, test_predictions))
print(confusion_matrix(test_labels, test_predictions))
print(classification_report(test_labels, test_predictions))

if create_submission:
    data_submission = pd.read_csv('Hotel-A-test.csv', index_col=0)
    submission_features = prepare_data(data_submission, scaler, pca)
    submission_predictions = model.predict(submission_features)
    submission_predictions = submission_predictions.tolist()
    submission_predictions = [x + 1 for x in submission_predictions]
    col_drop = data_submission.columns.tolist()
    submission = data_submission.drop(col_drop, 1)
    submission['Reservation_status'] = pd.DataFrame(submission_predictions, columns=[
                                                    'Reservation_status'])['Reservation_status'].values
    submission.to_csv('submission_file.csv')