"""
======================================================
Plot individual and voting regression predictions v 01
======================================================

.. currentmodule:: sklearn

A voting regressor is an ensemble meta-estimator that fits several base
regressors, each on the whole dataset. Then it averages the individual
predictions to form a final prediction.
Multiple different regressors will be used to predict the data:
:class:`~ensemble.GradientBoostingRegressor`,
:class:`~ensemble.RandomForestRegressor`,
:class:`~ensemble.AdaBoostRegressor`,
:class:`~ensemble.MLPRegressor`,
:class:`~ensemble.KNeighborsRegressor`, and
:class:`~linear_model.LinearRegression`).
Then the above regressors will be used for the
:class:`~ensemble.VotingRegressor`.

Predictions made by all models will be plotted for comparison.

The dataset consists of a Qubit measurement and the RFU size distribution
from the fragment analyzer (number of features depends on bin size used for
fragment sizes). The target is the molarity for each sample.

"""
print(__doc__)

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def import_data(data_file_name, index_column, sheet_name, skiprows=None, verifyIntegrityFlag=True):

    engine = 'openpyxl'  # Support for xlxs file format
    if data_file_name.split('.')[1] == 'xls':
        engine = 'xlrd'  # Support for xls file format
    df = pd.read_excel(data_file_name,
                       sheet_name=sheet_name,
                       skiprows=skiprows,
                       engine=engine,
                       keep_default_na=False)

    # df.columns = modify_df_column_names(df.columns)

    # Remove blank rows
    index_names = df[df[index_column] == ''].index
    df.drop(index_names, inplace=True)

    # Remove unnamed columns
    unnamedColList = [colName for colName in df.columns if str(colName).startswith('Unnamed')]
    df.drop(labels=unnamedColList, axis='columns', inplace=True)

    # Create custom columns
    # create_custom_columns(df, documentName, data_file_name)
    # df.set_index(index_column, drop=False, inplace=True, verify_integrity=verifyIntegrityFlag)

    return df


def calculate_bin_number(size, binSize, startOffset, maxSize):

    if size < startOffset: return -1
    if size > maxSize: return -1

    return int(size / binSize)


# %%
# Training classifiers
# --------------------------------
#
# First, we will load the diabetes dataset and initiate a gradient boosting
# regressor, a random forest regressor and a linear regression. Next, we will
# use the 3 regressors to build the voting regressor:
def main(args):
    testMode = args.testmode
    randState = args.random_state
    maxIterations = args.max_iterations

    # Load data
    X_diabetes, y_diabetes = load_diabetes(return_X_y=True, as_frame=True)
    # print('X:', X.head(15))
    # print('y:', y.head(15))

    # Set up dataframes for X and y values
    binSize = 50
    startOffset = 20  # This should cut off the LM peak
    maxSize = 5700  # This should cut off the UM peak
    # maxSize = 14000  # This is the maximum size for all samples
    numberOfBins = int(maxSize / binSize)
    # print('Number of bins:', numberOfBins)

    yDF = pd.DataFrame(columns=['Sample', 'target'])
    dfQubit = pd.DataFrame(columns=['Sample', 'value'])
    binColumnNames = []
    for i in range(numberOfBins):
        binColumnNames.append('bin' + str(i))
    X_unnormalized = pd.DataFrame(columns=binColumnNames)

    # Load from ATACseq_molarity_estimation
    data_file_name = 'ATACseq_molarity_estimation.xlsx'
    sheetList = ['3021T', '3013Ta', '3013Tb', '3005T']
    for sheet in sheetList:
        df = import_data(data_file_name, 'Sample', sheet)
        # print(df.head(10))

        for index, row in df.iterrows():
            # Set up target values
            sample_original = str(row['Sample']).strip()
            sample = sample_original.replace(' ', '_')
            sample = sample.replace('-', '_')
            sample = sample.replace(':', '')

            qubit = float(row['Qubit (ng/uL)'])
            nMol_measured = float(row['dPCR (nMol)-measured'])
            nMol_actual = float(row['dPCR (nMol)-actual'])
            nMol_after_sequencing = str(row['nMol after sequencing'])

            # print(sample, qubit, nMol_measured, nMol_actual, nMol_after_sequencing)
            if len(nMol_after_sequencing) > 0: yVal = float(nMol_after_sequencing)
            else: yVal = nMol_actual
            yDF = yDF.append({'Sample': sample, 'target': yVal}, ignore_index=True)

            dfQubit = dfQubit.append({'Sample': sample, 'value': qubit}, ignore_index=True)

            # Divide RFU size distribution into bins
            cumulativeBins = [0] * numberOfBins
            binCounts = [0] * numberOfBins
            for i in range(7, len(row)):
                # print(index, row)
                size = row.index[i]
                rfu = row[size]
                binNumber = calculate_bin_number(size, binSize, startOffset, maxSize)
                if binNumber < 0: continue
                # print(size, binNumber)
                cumulativeBins[binNumber] = cumulativeBins[binNumber] + rfu
                binCounts[binNumber] = binCounts[binNumber] + 1
                # print(size, rfu)
            # print('CumulativeBins:', cumulativeBins)
            # print('binCounts:', binCounts)
            for binNumber in range(len(cumulativeBins)):
                cumulativeBins[binNumber] = float(cumulativeBins[binNumber]) / binCounts[binNumber]
            # print('Adjusted cumulativeBins:', cumulativeBins)
            X_unnormalized.loc[sample] = cumulativeBins

    yDF.set_index('Sample', drop=True, inplace=True, verify_integrity=True)
    # print(yDF.head(25))

    y_array = np.array(yDF.iloc[:, 0: 1]).reshape(-1)
    # yDF_temp = yDF.reset_index(drop=True)
    # y = pd.DataFrame()
    # y = pd.DataFrame(data=yDF_temp['target'])
    # y.loc[0] = yDF_temp['target']
    y = pd.DataFrame(y_array).squeeze()

    # print('y_diabetes:', y_diabetes.head(15))
    # print('y_diabetes shape:', y_diabetes.shape)
    # print('y:', y)
    # print('y shape:', y.shape)

    dfQubit.set_index('Sample', drop=True, inplace=True, verify_integrity=True)
    # print(dfQubit.head(25))
    # print(X_unnormalized.head(25))

    # Normalize data
    X_normalized_array = preprocessing.normalize(X_unnormalized, norm='l2')
    X_normailzed = pd.DataFrame(X_normalized_array, columns=binColumnNames)
    X_normailzed['Sample'] = dfQubit.index
    X_normailzed.set_index('Sample', drop=True, inplace=True, verify_integrity=True)
    X_normailzed['Qubit'] = dfQubit['value']
    X = X_normailzed.reset_index(drop=True)

    # print('X_diabetes:', X_diabetes.head(15))
    # print('X:', X.head(25))

    # Dimensionality reduction
    # //--- omit dimensions with least variance

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = randState)
    # print('tDF:', yDF.head)
    # print('Y:', y)
    # print(y_test)
    y_test_array = np.array(y_test.iloc[:]).reshape(-1)

    print('\nX train shape:', X_train.shape, ' y train shape:', y_train.shape)
    print('X test shape:', X_test.shape, ' y test shape:', y_test.shape)

    # Train classifiers
    reg_gb = GradientBoostingRegressor(random_state=randState)
    reg_rf = RandomForestRegressor(random_state=randState)
    reg_ab = AdaBoostRegressor(random_state=randState)
    reg_mlp = MLPRegressor(random_state=randState, max_iter=maxIterations)
    reg_kn = KNeighborsRegressor()
    reg_lr = LinearRegression()

    reg_gb.fit(X_train, y_train)
    reg_rf.fit(X_train, y_train)
    reg_ab.fit(X_train, y_train)
    reg_mlp.fit(X_train, y_train)
    reg_kn.fit(X_train, y_train)
    reg_lr.fit(X_train, y_train)

    ereg = VotingRegressor([('gb', reg_gb), ('rf', reg_rf), ('ab', reg_ab), ('mlp', reg_mlp), ('kn', reg_kn)])
    ereg.fit(X_train, y_train)

    scores_gb = cross_val_score(reg_gb, X_train, y_train, cv=5)
    scores_rf = cross_val_score(reg_rf, X_train, y_train, cv=5)
    scores_ab = cross_val_score(reg_ab, X_train, y_train, cv=5)
    scores_mlp = cross_val_score(reg_mlp, X_train, y_train, cv=5)
    scores_kn = cross_val_score(reg_kn, X_train, y_train, cv=5)
    scores_lr = cross_val_score(reg_lr, X_train, y_train, cv=5)
    scores_vr = cross_val_score(ereg, X_train, y_train, cv=5)

    print("\nScores_mlp: {}".format(scores_ab))
    print("Mean cross validation score: {}".format(np.mean(scores_ab)))
    print("Score without cross validation: {}".format(reg_ab.score(X_train, y_train)))

    # %%
    # Making predictions
    # --------------------------------
    #
    # Now we will use each of the regressors to make the 20 first predictions.
    # xt = X[:21]
    xt = X_test

    pred_gb = reg_gb.predict(xt)
    pred_rf = reg_rf.predict(xt)
    pred_ab = reg_ab.predict(xt)
    pred_mlp = reg_mlp.predict(xt)
    pred_kn = reg_kn.predict(xt)
    pred_lr = reg_lr.predict(xt)
    pred_vr = ereg.predict(xt)

    print('\ny_test:', y_test_array)
    # print('y_test shape:', y_test_array.shape)
    print('pred_gb:', pred_gb)
    # print('pred_gb shape:', pred_gb.shape)
    print('pred_rf:', pred_rf)
    # print('pred_rf shape:', pred_rf.shape)
    print('pred_ab:', pred_ab)
    # print('pred_ab shape:', pred_ab.shape)
    print('pred_mlp:', pred_mlp)
    print('mlp fold change:', pred_mlp / y_test_array)
    # print('pred_mlp shape:', pred_mlp.shape)
    print('pred_kn:', pred_kn)
    # print('pred_kn shape:', pred_kn.shape)

    # %%
    # Plot the results
    # --------------------------------
    #
    # Finally, we will visualize the test predictions. The red stars show the average
    # prediction made by :class:`~ensemble.VotingRegressor`.
    plt.figure()

    # plt.plot(pred_mlp, 'k.', label='MLPRegressor')
    # plt.plot(pred_gb, 'gd', label='GradientBoostingRegressor')
    # plt.plot(pred_rf, 'b^', label='RandomForestRegressor')
    # plt.plot(pred_ab, 'mh', label='AdaBoostRegressor')
    # plt.plot(pred_kn, 'ys', label='KNeighborsRegressor')
    # plt.plot(pred_lr, 'cp', label='LinearRegression')
    # plt.plot(pred_vr, 'c+', ms=10, label='VotingRegressor')
    # plt.plot(y_test_array, 'r*', ms=10, label='Actual')
    # plt.title('Regressor predictions and their average')

    # Log scale on the y-axis
    plt.semilogy(pred_mlp, 'k.', label='MLPRegressor')
    # plt.semilogy(pred_gb, 'gd', label='GradientBoostingRegressor')
    # plt.semilogy(pred_rf, 'b^', label='RandomForestRegressor')
    # plt.semilogy(pred_ab, 'mh', label='AdaBoostRegressor')
    # plt.semilogy(pred_kn, 'ys', label='KNeighborsRegressor')
    # plt.semilogy(pred_lr, 'cp', label='LinearRegression')
    # plt.semilogy(pred_vr, 'c+', ms=10, label='VotingRegressor')
    plt.semilogy(y_test_array, 'r*', ms=10, label='Actual')
    plt.title('Regressor predictions and their average (log scale)')

    # Fold change
    # plt.plot(pred_mlp / y_test_array, 'k.', label='MLPRegressor')
    # plt.plot(pred_gb / y_test_array, 'gd', label='GradientBoostingRegressor')
    # plt.plot(pred_rf / y_test_array, 'b^', label='RandomForestRegressor')
    # plt.plot(pred_ab / y_test_array, 'mh', label='AdaBoostRegressor')
    # plt.plot(pred_kn / y_test_array, 'ys', label='KNeighborsRegressor')
    # plt.plot(pred_lr / y_test_array, 'cp', label='LinearRegression')
    # plt.plot(pred_vr / y_test_array, 'c+', ms=10, label='VotingRegressor')
    # plt.plot(y_test_array / y_test_array, 'r*', ms=10, label='Actual')
    # plt.title('Regressor predictions fold change')

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel('predicted')
    plt.xlabel('training samples')
    plt.legend(loc="best")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("-c", "--chromo", help="Chromosome: chr1 - chr22, or chrX", default="chr21")
    parser.add_argument("-r", "--random_state", help="Random state initializer", type=int, default=1)
    parser.add_argument("-m", "--max_iterations", help="Maximum number of iterations for MLP regressor", type=int, default=10000)
    parser.add_argument("-t", "--testmode", help="Turn test mode on", action="store_true")
    args = parser.parse_args()

    main(args)
