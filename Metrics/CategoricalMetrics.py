import statistics
import warnings
import pandas as pd
import numpy as np


class MetricsData:

    def __init__(self, input_data: pd.DataFrame, data_label: str, label_classes: list, true_val):
        self.input_data = input_data
        self.data_label = data_label
        self.label_classes = label_classes
        self.true_val = true_val

    def __get_unique_values_and_labels(self, data_feature: str):
        """
        This function returns the unique values and labels for a given feature
        :param data_feature: for which the unique values and labels are to be calculated
        :return: unique values and labels for the given feature
        """

        unique_values = self.input_data[data_feature].unique()
        unique_labels = self.input_data[self.data_label].unique()

        return unique_values, unique_labels

    def get_labels_distribution(self, data_feature: str):

        """
        This function returns the distribution of labels for a given feature
        :param data_feature:  for which the distribution is to be calculated
        :return: dataframe with the distribution of labels for the given feature having columns as
            - Variable
            - Value
            - columns for each label class
            - Distribution of labels for the given feature
        """
        lst = []
        unique_values, unique_labels = self.__get_unique_values_and_labels(data_feature)
        if set(unique_labels) == set(self.label_classes):
            for val in unique_values:
                classes_count = []
                for label_class in range(len(self.label_classes)):
                    classes_count.append(
                        self.input_data[
                            (self.input_data[data_feature] == val) & (
                                    self.input_data[self.data_label] == self.label_classes[label_class])].count()[
                            data_feature])

                classes_list = [data_feature, val] + classes_count
                lst.append(classes_list)

            label_columns = []
            for label_class in range(len(self.label_classes)):
                label_columns.append(str(self.label_classes[label_class]))

            columns = ['Variable', 'Value'] + label_columns
            label_distribution_data = pd.DataFrame(lst, columns=columns)
            total_classes = []
            for label_class in range(len(self.label_classes)):
                total_classes.append(
                    self.input_data[self.input_data[self.data_label] == self.label_classes[label_class]].count()[
                        data_feature])
                label_distribution_data['Distribution_' + str(self.label_classes[label_class])] = \
                    round(label_distribution_data[str(self.label_classes[label_class])] / total_classes[label_class], 3)

            return label_distribution_data

    def get_precision_recall_f1(self, data_feature: str):

        """
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)

        :param data_feature: feature for which precision, recall and f1 score is to be calculated
        :return: Pandas dataframe with precision, recall and f1 score for each class having columns as
                - if classes are 2:
                    - DataFrame returns columns which are
                        - Variable
                        - Value
                        - Precision columns for each class
                        - Recall columns for each class
                        - F1 columns for each class
                - if classes are more than 2:
                    - DataFrame returns columns which are
                        - Variable
                        - Value
                        - Precision column as None
                        - Recall column as None
                        - F1 column as None
        """

        precision_recall_f1_data = self.get_labels_distribution(data_feature)
        unique_values, unique_labels = self.__get_unique_values_and_labels(data_feature)

        if set(unique_labels) == set(self.label_classes):
            if len(self.label_classes) == 2:
                for label_class in range(len(self.label_classes)):
                    if self.true_val in unique_values:

                        # Precision for 2 classes
                        precision_recall_f1_data['Precision_' + str(self.label_classes[label_class])] = \
                            precision_recall_f1_data['Distribution_' + str(self.label_classes[label_class])].loc[
                                precision_recall_f1_data['Value'] == self.true_val].unique()[0]

                        # Recall for 2 classes
                        if label_class == 0:
                            num = \
                                precision_recall_f1_data[str(self.label_classes[label_class])].loc[
                                    precision_recall_f1_data['Value'] == self.true_val].unique()[
                                    0]
                            den = precision_recall_f1_data[str(self.label_classes[1])].loc[
                                precision_recall_f1_data['Value'] != self.true_val].unique()
                            precision_recall_f1_data['Recall_' + str(self.label_classes[label_class])] = round(
                                num / (num + sum(den)), 3)
                        else:
                            num = \
                                precision_recall_f1_data[str(self.label_classes[label_class])].loc[
                                    precision_recall_f1_data['Value'] == self.true_val].unique()[
                                    0]
                            den = precision_recall_f1_data[str(self.label_classes[0])].loc[
                                precision_recall_f1_data['Value'] != self.true_val].unique()
                            precision_recall_f1_data['Recall_' + str(self.label_classes[label_class])] = round(
                                num / (num + sum(den)), 3)

                        #    f1 score for 2 classes
                        precision_recall_f1_data['f1_' + str(self.label_classes[label_class])] = \
                            round(statistics.harmonic_mean([
                                precision_recall_f1_data['Precision_' + str(self.label_classes[label_class])][0],
                                precision_recall_f1_data['Recall_' + str(self.label_classes[label_class])][0]
                            ]), 3)
                        precision_recall_f1_data = precision_recall_f1_data.replace(
                            {'f1_' + str(self.label_classes[label_class]): {np.inf: 0, -np.inf: 0}})

            else:
                precision_recall_f1_data['Precision'] = None
                precision_recall_f1_data['Recall'] = None
                precision_recall_f1_data['f1'] = None

            relevant_columns = [x for x in precision_recall_f1_data.columns if 'Distribution' not in x]
            precision_recall_f1_data = precision_recall_f1_data[relevant_columns]
            return precision_recall_f1_data

        else:
            raise ValueError("given label classes are inconsistent with the data")

    def get_woe_iv(self, data_feature: str):

        """
        WOE = ln ( (good / bad) / (total_good / total_bad) )
        IV = (good / total_good - bad / total_bad) * WOE

        :param data_feature:
        :return: Pandas DataFrame with WOE and IV values having the following columns:
            - Variable
            - Value
            - Distribution columns of label classes
            - WoE for self.true_val
            - IV for self.true_val
        """

        unique_values, unique_labels = self.__get_unique_values_and_labels(data_feature)

        if set(unique_labels) == set(self.label_classes):
            woe_iv_data = self.get_labels_distribution(data_feature)

            distribution_sum = 0
            for label_class in range(len(self.label_classes)):
                distribution_sum += woe_iv_data['Distribution_' + str(self.label_classes[label_class])]

            woe_iv_data['WoE_' + str(self.true_val)] = round(np.log(
                woe_iv_data['Distribution_' + str(self.label_classes[0])] /
                (distribution_sum - woe_iv_data['Distribution_' + str(self.label_classes[0])])), 3)
            woe_iv_data = woe_iv_data.replace(
                {'WoE_' + str(self.true_val): {np.inf: 0, -np.inf: 0, np.nan: 0}})

            woe_iv_data['IV_' + str(self.true_val)] = round(
                woe_iv_data['WoE_' + str(self.true_val)] *
                (woe_iv_data['Distribution_' + str(self.label_classes[0])] - (
                        distribution_sum - woe_iv_data['Distribution_' + str(self.label_classes[0])])), 3)

            woe_iv_data = woe_iv_data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
            woe_iv_data.index = range(len(woe_iv_data.index))

            relevant_columns = [x for x in woe_iv_data.columns if 'Distribution' not in x]
            woe_iv_data = woe_iv_data[relevant_columns]

            return woe_iv_data
        else:
            raise ValueError("given label classes are inconsistent with the data")

    def get_feature_report(self, data_feature: str):

        """
        :param data_feature: string
        :return: pandas dataframe with the following columns:
            - Variable
            - Value
            - n columns for each label class having the number of observations for each label class
            - Precision columns for each label class
            - Recall columns for each label class
            - f1 columns for each label class
            - WoE for self.true_val
            - IV for self.true_val
        """

        label_distribution_data = self.get_labels_distribution(data_feature)
        precision_recall_f1_data = self.get_precision_recall_f1(data_feature)
        woe_iv_data = self.get_woe_iv(data_feature)

        feature_report = pd.merge(label_distribution_data, precision_recall_f1_data,
                                  on=['Variable', 'Value'] + self.label_classes)
        feature_report = pd.merge(feature_report, woe_iv_data, on=['Variable', 'Value'] + self.label_classes)

        return feature_report

    def get_feature_conclusion(self, data_feature: str):

        """
        :param data_feature: string
        :return: dictionary having the following keys:
            - column
            - label_classes_count
            - feature_variables_count
            - Total columns for each label class having the total number of observations for each label class
            - Precision columns for each label class
            - Recall columns for each label class
            - f1 columns for each label class
            - WoE for self.true_val
            - IV for self.true_val
        """

        unique_values, unique_labels = self.__get_unique_values_and_labels(data_feature)
        data = self.get_feature_report(data_feature)

        if set(unique_labels) == set(self.label_classes):
            feature_conclusion = dict()
            feature_conclusion["column"] = str(data_feature)
            feature_conclusion["label_classes_count"] = str(len(self.label_classes))
            feature_conclusion["feature_variables_count"] = str(len(unique_values))

            for label_class in range(len(self.label_classes)):
                feature_conclusion['Total_' + str(self.label_classes[label_class])] = str(
                    data[self.label_classes[label_class]].sum())

            if len(self.label_classes) == 2:
                for label_class in range(len(self.label_classes)):
                    try:
                        feature_conclusion['Precision_' + str(self.label_classes[label_class])] = \
                            str(round(data['Precision_' + self.label_classes[label_class]].mean(), 3))
                        feature_conclusion['Recall_' + str(self.label_classes[label_class])] = \
                            str(round(data['Recall_' + self.label_classes[label_class]].mean(), 3))
                        feature_conclusion['f1_' + str(self.label_classes[label_class])] = \
                            str(round(data['f1_' + self.label_classes[label_class]].mean(), 3))
                    except Exception as e:
                        print(e)
                        feature_conclusion['Precision_' + str(self.label_classes[label_class])] = str(None)
                        feature_conclusion['Recall_' + str(self.label_classes[label_class])] = str(None)
                        feature_conclusion['f1_' + str(self.label_classes[label_class])] = str(None)


            else:
                feature_conclusion['Precision'] = str(None)
                feature_conclusion['Recall'] = str(None)
                feature_conclusion['f1'] = str(None)

            feature_conclusion['WoE_' + str(self.true_val)] = str(round(
                data['WoE_' + str(self.true_val)].sum(), 3))
            feature_conclusion['IV_' + str(self.true_val)] = str(round(
                data['IV_' + str(self.true_val)].sum(), 3))

            return feature_conclusion

        else:
            raise ValueError("given label classes are inconsistent with the data")


class MetricsBooleanData:
    """
    :Attributes  data_label: column which is treated as label
            label_classes: list of possible value of classes
            preferred_value: value of boolean data preferred for calculation
            value_classes: list of boolean values of column
    """

    def __init__(self, data: pd.DataFrame, data_label: str, label_classes: list, preferred_value: str,
                 value_classes: list):
        self.data = data
        self.data_label = data_label
        self.label_classes = label_classes
        self.preferred_value = preferred_value
        self.value_classes = value_classes

    def get_confusion_matrix(self, data_feature: str):

        data = self.data[self.data[data_feature].isin(self.value_classes)]
        record = dict()
        record['Variable'] = str(data_feature)
        record['Total'] = data.shape[0]
        for label_class in self.label_classes:
            record[label_class] = str(data[data[self.data_label] == label_class].shape[0])
        for value_class in self.value_classes:
            for label_class in self.label_classes:
                record[value_class + '_' + label_class] = str(data[(data[self.data_label] == label_class) & (
                        data[data_feature] == value_class)].shape[0])

        return record

    def get_woe_iv(self, data_feature: str):

        data = self.data[self.data[data_feature].isin(self.value_classes)]
        record = dict()
        record['Variable'] = str(data_feature)
        record['Total'] = data.shape[0]

        for label_class in self.label_classes:
            record[label_class] = data[data[self.data_label] == label_class].shape[0]

        for value_class in self.value_classes:
            for label_class in self.label_classes:
                record[value_class + '_' + label_class] = data[(data[self.data_label] == label_class) & (
                        data[data_feature] == value_class)].shape[0]


        value1_label1 = record[self.value_classes[0] + '_' + self.label_classes[0]]
        label1 = record[self.label_classes[0]]
        value1_label2 = record[self.value_classes[0] + '_' + self.label_classes[1]]
        label2 = record[self.label_classes[1]]

        try:
            woe_before_log = ((value1_label1 / value1_label2) / (label1 / label2))
        except ZeroDivisionError:
            warnings.warn("Zero Division error in WoE_begore_log")
            woe_before_log = 0


        record['WoE' + '_' + str(self.value_classes[0])] = round(np.log(woe_before_log), 3)
        try:
            iv = ((value1_label1 / label1) - (value1_label2 / label2)) * record['WoE' + '_' + str(self.value_classes[0])]
            record['IV' + '_' + str(self.value_classes[0])] = round(iv, 3)
        except ZeroDivisionError:
            warnings.warn("Zero Division error in iv")
            iv = 0
            record['IV' + '_' + str(self.value_classes[0])] = round(iv, 3)



        value1_label1 = record[self.value_classes[1] + '_' + self.label_classes[1]]
        label1 = record[self.label_classes[1]]
        value1_label2 = record[self.value_classes[1] + '_' + self.label_classes[0]]
        label2 = record[self.label_classes[0]]

        try:
            woe_before_log = ((value1_label1 / value1_label2) / (label1 / label2))
        except ZeroDivisionError:
            warnings.warn("Zero Division error in WoE_begore_log")
            woe_before_log = 0

        record['WoE' + '_' + str(self.value_classes[1])] = round(np.log(woe_before_log), 3)
        try:
            iv = ((value1_label1 / label1) - (value1_label2 / label2)) * record[
                'WoE' + '_' + str(self.value_classes[1])]
            record['IV' + '_' + str(self.value_classes[1])] = round(iv, 3)
        except ZeroDivisionError:
            warnings.warn("Zero Division error in iv")
            iv = 0
            record['IV' + '_' + str(self.value_classes[1])] = round(iv, 3)

        record['IV'] = round(record['IV' + '_' + str(self.value_classes[0])] + record['IV' + '_' + str(self.value_classes[1])], 3)

        for key, value in record.items():
            record[key] = str(value)

        return record
