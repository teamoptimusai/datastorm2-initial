import pandas as pd
import numpy as np


class Multi_Model:
    def __init__(self, model1, model2, model3):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.predictions1 = None
        self.predictions2 = None
        self.with_pred_train3_df = None
        self.with_pred_train3_features = None
        self.with_pred_df = None

    def fit(self, train1_features, train1_labels, train2_features, train2_labels, train3_features, train3_labels):
        # training model1 and model2
        self.model1.fit(train1_features, train1_labels)
        self.model2.fit(train2_features, train2_labels)
        # concatenation
        self.predictions1 = self.model1.predict(train3_features)
        self.predictions2 = self.model2.predict(train3_features)
        self.with_pred_train3_df = pd.DataFrame(train3_features)
        self.with_pred_train3_df['Pred1'] = pd.DataFrame(
            self.predictions1.tolist(), columns=['Pred1'])['Pred1'].values
        self.with_pred_train3_df['Pred2'] = pd.DataFrame(
            self.predictions1.tolist(), columns=['Pred2'])['Pred2'].values
        self.with_pred_train3_features = np.array(self.with_pred_train3_df)
        # training model3
        self.model3.fit(self.with_pred_train3_features, train3_labels)

    def predict(self, features):
        self.with_pred_df = pd.DataFrame(features)
        self.with_pred_df['Pred1'] = pd.DataFrame(self.model1.predict(
            features).tolist(), columns=['Pred1'])['Pred1'].values
        self.with_pred_df['Pred2'] = pd.DataFrame(
            self.model2.predict(features), columns=['Pred2'])['Pred2'].values

        return (self.model3.predict(np.array(self.with_pred_df)))
