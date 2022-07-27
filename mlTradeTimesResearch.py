import math
import time
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
import matplotlib.pyplot as plt
from dtreeviz.trees import dtreeviz
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

start_time = time.time()

path = r'C:\Users\mweiss\Python Projects\ML Trade Times\new_600k_trade_data.xlsx'
#path = r'/Users/martinweiss/Documents/Python/MSRB/ML Trade Times/trade_report_time_elapsed_data_all_1.xlsx'
#path = r'/content/trade_report_time_elapsed_data_all_1.xlsx'
outputPath = r'C:\Users\mweiss\Python Projects\ML Trade Times'
data = pd.read_excel(path, sheet_name = 'trade_report_time_elapsed_data')

binary = data['CLASS']
minutesElapsed = data['MINUTES_ELAPSED']
timeOfTrade = data['TOT']
recordType = data['RSRF.record_type']
dealerRank = data['dealer_rank']
dealerQuintile = data['dealer_quintile']
price = data['RSRF.dollar_price']
parAmount = data['RSRF.par']
capacity = data['RSRF.capacity']
msrb_issue_status_ind = data['RSRF.msrb_issue_status_ind']
specialOne = data['RSRF.special_price_reason_1']
specialTwo = data['RSRF.special_price_reason_2']
specialThree = data['RSRF.special_price_reason_3']
settlementDays = data['RSRF.settlement_days']
OFF_SETTLE_DATE_YEAR = data['MCSM.OFF_SETTLE_DATE_YEAR']
MAT_DATE_YEAR = data['MCSM.QS_MAT_DATE_YEAR']
DTD_TO_MTY_YEARS = data['DTD_TO_MTY_YEARS']
YEARS_SINCE_DTD = data['YEARS_SINCE_DTD']
bondAgeFraction = data['FRACTION_OF_LIFE_OF_BOND']
roundedBondAgeFraction = data['ROUNDED_FRACTION_OF_LIFE_OF_BOND']
QS_COUPON_RATE = data['MCSM.QS_COUPON_RATE']
INT_PAY_FREQ_STD_PERIOD = data['MCSM.INT_PAY_FREQ_STD_PERIOD']
INT_ACCRUAL_COUNT = data['MCSM.INT_ACCRUAL_COUNT']
monthsToNextCall = data['MONTHS_TO_NEXT_CALL']
CALL_FREQ_CODE = data['MCSM.CALL_FREQ_CODE']
CALL_TYPE = data['MCSM.CALL_TYPE']
OFF_WI_IND = data['MCSM.OFF_WI_IND']
OFF_SECURITY_TYPE = data['MCSM.OFF_SECURITY_TYPE']
IND_INT_TYPE = data['MCSM.IND_INT_TYPE']
DEFAULT_IND = data['MCSM.DEFAULT_IND']
COMPLETE_CALL_IND = data['MCSM.COMPLETE_CALL_IND']
MAT_DENOM = data['MCSM.AM_MAT_DENOM']
PUT_EXISTS_IND = data['MCSM.PUT_EXISTS_IND']
CSB_ISSUE_TRANS = data['MCSM.CSB_ISSUE_TRANS']
ISSUE_STATUS = data['MCSM.CSB_ISSUE_STATUS']
REVENUE_TYPE = data['MCSM.REVENUE_TYPE']
PURPOSE_CLASS = data['MCSM.PT_PURPOSE_CLASS']

timeOfTrade = timeOfTrade.tolist()
recordType = recordType.tolist()
dealerRank = dealerRank.tolist()
dealerQuintile = dealerQuintile.tolist()
price = price.tolist()
parAmount = parAmount.tolist()
capacity = capacity.tolist()
msrb_issue_status_ind = msrb_issue_status_ind.tolist()
specialOne = specialOne.tolist()
specialTwo = specialTwo.tolist()
specialThree = specialThree.tolist()
settlementDays = settlementDays.tolist()
OFF_SETTLE_DATE_YEAR = OFF_SETTLE_DATE_YEAR.tolist()
MAT_DATE_YEAR = MAT_DATE_YEAR.tolist()
DTD_TO_MTY_YEARS = DTD_TO_MTY_YEARS.tolist()
YEARS_SINCE_DTD = YEARS_SINCE_DTD.tolist()
bondAgeFraction = bondAgeFraction.tolist()
roundedBondAgeFraction = roundedBondAgeFraction.tolist()
QS_COUPON_RATE = QS_COUPON_RATE.tolist()
INT_PAY_FREQ_STD_PERIOD = INT_PAY_FREQ_STD_PERIOD.tolist()
INT_ACCRUAL_COUNT = INT_ACCRUAL_COUNT.tolist()
monthsToNextCall = monthsToNextCall.tolist()
CALL_FREQ_CODE = CALL_FREQ_CODE.tolist()
CALL_TYPE = CALL_TYPE.tolist()
OFF_WI_IND = OFF_WI_IND.tolist()
OFF_SECURITY_TYPE = OFF_SECURITY_TYPE.tolist()
IND_INT_TYPE = IND_INT_TYPE.tolist()
DEFAULT_IND = DEFAULT_IND.tolist()
COMPLETE_CALL_IND = COMPLETE_CALL_IND.tolist()
MAT_DENOM = MAT_DENOM.tolist()
PUT_EXISTS_IND = PUT_EXISTS_IND.tolist()
CSB_ISSUE_TRANS = CSB_ISSUE_TRANS.tolist()
ISSUE_STATUS = ISSUE_STATUS.tolist()
REVENUE_TYPE = REVENUE_TYPE.tolist()
PURPOSE_CLASS = PURPOSE_CLASS.tolist()

timeOfTradeCat = []
for i in range(len(timeOfTrade)):
    timeOfTrade[i] = str(timeOfTrade[i])
    timeOfTrade[i] = timeOfTrade[i][0:5].replace(':', '.')
    timeOfTrade[i] = float(timeOfTrade[i])
    if(timeOfTrade[i] < 10.45):
            timeOfTradeCat.append('Before 10:45')
    if((timeOfTrade[i] >= 10.45) and (timeOfTrade[i] <= 12)):
            timeOfTradeCat.append('Between 10:45 and 12:00')
    if((timeOfTrade[i] > 12) and (timeOfTrade[i] <= 13.15)):
            timeOfTradeCat.append('Between 12:00 and 1:15')
    if(timeOfTrade[i] > 13.15):
            timeOfTradeCat.append('After 1:15')

priceCat = []
for i in range(len(price)):
    if(price[i] < 90):
            priceCat.append('Less than $90')
    if((price[i] >= 90) and (price[i] < 100)):
            priceCat.append('Between $90 and $100')
    if(price[i] == 100):
            priceCat.append('$100')
    if(price[i] > 100):
            priceCat.append('Greater than $100')

specialOneCat = []
for i in range(len(specialOne)):
    if(specialOne[i] == 0):
        specialOneCat.append('Zero')
    else:
        specialOneCat.append('One')

specialTwoCat = []
for i in range(len(specialTwo)):
    if(specialTwo[i] == 0):
        specialTwoCat.append('Zero')
    else:
        specialTwoCat.append('One')

specialThreeCat = []
for i in range(len(specialThree)):
    if(specialThree[i] == 0):
        specialThreeCat.append('Zero')
    else:
        specialThreeCat.append('One')

parAmountCat = []
for i in range(len(parAmount)):
    if(parAmount[i] < 15000):
            parAmountCat.append('Less than 15,000')
    if((parAmount[i] >= 15000) and (parAmount[i] <= 25000)):
            parAmountCat.append('Between 15,000 and 25,000')
    if((parAmount[i] > 25000) and (parAmount[i] <= 74500)):
            parAmountCat.append('Between 25,000 and 74,500')
    if(parAmount[i] > 74500):
            parAmountCat.append('Greater than 74,500')

settlementDaysCat = []
for i in range(len(settlementDays)):
    if(settlementDays[i] < 2):
            settlementDaysCat.append('Less than 2 Settlement Days')
    if(settlementDays[i] == 2):
            settlementDaysCat.append('2 Settlement Days')
    if(settlementDays[i] > 2):
            settlementDaysCat.append('More than 2 Settlement Days')

settleYearCat = []
for i in range(len(OFF_SETTLE_DATE_YEAR)):
    if(OFF_SETTLE_DATE_YEAR[i] < 2017):
            settleYearCat.append('2017 or before')
    if((OFF_SETTLE_DATE_YEAR[i] >= 2017) and (OFF_SETTLE_DATE_YEAR[i] <= 2019)):
            settleYearCat.append('2017 - 2019')
    if(OFF_SETTLE_DATE_YEAR[i] > 2019):
            settleYearCat.append('2019 or later')

yearsToMatureCat = []
for i in range(len(DTD_TO_MTY_YEARS)):
    if(DTD_TO_MTY_YEARS[i] < 9):
            yearsToMatureCat.append('Less than 9')
    if((DTD_TO_MTY_YEARS[i] >= 9) and (DTD_TO_MTY_YEARS[i] <= 12)):
            yearsToMatureCat.append('9 - 12')
    if((DTD_TO_MTY_YEARS[i] > 12) and (DTD_TO_MTY_YEARS[i] <= 19)):
            yearsToMatureCat.append('12 - 19')
    if(DTD_TO_MTY_YEARS[i] > 19):
            yearsToMatureCat.append('Greater than 19')

bondAgeCat = []
for i in range(len(YEARS_SINCE_DTD)):
    if(YEARS_SINCE_DTD[i] < 2):
            bondAgeCat.append('Less than 2')
    if((YEARS_SINCE_DTD[i] >= 2) and (YEARS_SINCE_DTD[i] <= 6)):
            bondAgeCat.append('2 - 6')
    if(YEARS_SINCE_DTD[i] > 6):
            bondAgeCat.append('Greater than 6')

for i in range(len(bondAgeFraction)):
    bondAgeFraction[i] = str(bondAgeFraction[i])
    if(bondAgeFraction[i] == 'nan'):
            bondAgeFraction[i] = 0
    bondAgeFraction[i] = float(bondAgeFraction[i])

bondAgeFractionCat = []
for i in range(len(bondAgeFraction)):
    if(bondAgeFraction[i] <= 0.22):
        bondAgeFractionCat.append('Bond Age Fraction less than 0.23')
    if((bondAgeFraction[i] > 0.22) and (bondAgeFraction[i] < 0.501)):
        bondAgeFractionCat.append('Bond Age Fraction between 0.23 and 0.501')
    if(bondAgeFraction[i] >= 0.501):
        bondAgeFractionCat.append('Bond Age Fraction greater than 0.501')

for i in range(len(QS_COUPON_RATE)):
    QS_COUPON_RATE[i] = str(QS_COUPON_RATE[i])
    if(QS_COUPON_RATE[i] == 'nan'):
            QS_COUPON_RATE[i] = 5
    QS_COUPON_RATE[i] = float(QS_COUPON_RATE[i])

couponRateCat = []
for i in range(len(QS_COUPON_RATE)):
    if(QS_COUPON_RATE[i] < 5):
        couponRateCat.append('Coupon Rate less than 5%')
    if(QS_COUPON_RATE[i] == 5):
        couponRateCat.append('Coupon Rate == 5%')
    if(QS_COUPON_RATE[i] > 5):
        couponRateCat.append('Coupon Rate greater than 5%')

monthsToNextCallCat = []
for i in range(len(monthsToNextCall)):
    monthsToNextCall[i] = str(monthsToNextCall[i])
    if(monthsToNextCall[i] == 'nan'):
        monthsToNextCallCat.append('n/a')
    else:
        monthsToNextCall[i] = float(monthsToNextCall[i])
    if(type(monthsToNextCall[i]) == float):
        if(monthsToNextCall[i] < 50):
                monthsToNextCallCat.append('Less than 50')
        if((monthsToNextCall[i] >= 50) and (monthsToNextCall[i] <= 85)):
                monthsToNextCallCat.append('Between 50 and 85')
        if(monthsToNextCall[i] > 85):
                monthsToNextCallCat.append('More than 85')

bondDenomCat = []
for i in range(len(MAT_DENOM)):
    MAT_DENOM[i] = str(MAT_DENOM[i])
    if(MAT_DENOM[i] == 'nan'):
        bondDenomCat.append('n/a')
    else:
        MAT_DENOM[i] = float(MAT_DENOM[i])
    if(type(MAT_DENOM[i]) == float):
        if(MAT_DENOM[i] < 5000):
                bondDenomCat.append('Less than 5000')
        if(MAT_DENOM[i] == 5000):
                bondDenomCat.append('5000')
        if(MAT_DENOM[i] > 5000):
                bondDenomCat.append('More than 5000')

binaryHold = binary
minutesElapsedThreshold = []
recallScores = []
precisionScores = []
accuracyScores = []
f1Scores = []
onTimeCount= []
lateCount = []
trueNegative = []
trueNegativePercent = []
falseNegative = []
falseNegativePercent = []
falsePostive = []
falsePostivePercent = []
truePostive = []
truePostivePercent = []
lateTradeCount = []

minutesElapsedCount = 0
for minutesElapsedCount in np.arange(0.1, 2.6, 0.1):
    for i in range(len(binary)):
        if(minutesElapsed[i] < minutesElapsedCount):
            binary[i] = 0
    print('Minutes Elapsed Threshold:', minutesElapsedCount)
    cleanedValues = pd.DataFrame({'Minutes Elapsed': minutesElapsed})
    cleanedValues.insert(1, 'Time of Trade', timeOfTradeCat)
    cleanedValues.insert(2, 'Record Type', recordType)
    cleanedValues.insert(3, 'Dealer Quintile', dealerQuintile)
    cleanedValues.insert(4, 'Price', priceCat)
    cleanedValues.insert(5, 'Par Amount', parAmountCat)
    cleanedValues.insert(6, 'Capacity', capacity)
    cleanedValues.insert(7, 'msrb_issue_status_ind', msrb_issue_status_ind)
    cleanedValues.insert(8, 'Special One', specialOneCat)
    cleanedValues.insert(9, 'Special Two', specialTwoCat)
    cleanedValues.insert(10, 'Special Three', specialThreeCat)
    cleanedValues.insert(11, 'Settlement Days', settlementDaysCat)
    cleanedValues.insert(12, 'OFF_SETTLE_DATE_YEAR', settleYearCat)
    cleanedValues.insert(13, 'MAT_DATE_YEAR', MAT_DATE_YEAR)
    cleanedValues.insert(14, 'DTD_TO_MTY_YEARS', yearsToMatureCat)
    cleanedValues.insert(15, 'YEARS_SINCE_DTD', bondAgeCat)
    cleanedValues.insert(16, 'Bond Age Fraction', bondAgeFractionCat)
    cleanedValues.insert(17, 'QS_COUPON_RATE', couponRateCat)
    cleanedValues.insert(18, 'INT_PAY_FREQ_STD_PERIOD', INT_PAY_FREQ_STD_PERIOD)
    cleanedValues.insert(19, 'INT_ACCRUAL_COUNT', INT_ACCRUAL_COUNT)
    cleanedValues.insert(20, 'MONTHS_TO_NEXT_CALL', monthsToNextCallCat)
    cleanedValues.insert(21, 'CALL_FREQ_CODE', CALL_FREQ_CODE)
    cleanedValues.insert(22, 'CALL_TYPE', CALL_TYPE)
    cleanedValues.insert(23, 'OFF_WI_IND', OFF_WI_IND)
    cleanedValues.insert(24, 'OFF_SECURITY_TYPE', OFF_SECURITY_TYPE)
    cleanedValues.insert(25, 'IND_INT_TYPE', IND_INT_TYPE)
    cleanedValues.insert(26, 'DEFAULT_IND', DEFAULT_IND)
    cleanedValues.insert(27, 'COMPLETE_CALL_IND', COMPLETE_CALL_IND)
    cleanedValues.insert(28, 'MAT_DENOM', bondDenomCat)
    cleanedValues.insert(29, 'PUT_EXISTS_IND', PUT_EXISTS_IND)
    cleanedValues.insert(30, 'CSB_ISSUE_TRANS', CSB_ISSUE_TRANS)
    cleanedValues.insert(31, 'ISSUE_STATUS', ISSUE_STATUS)
    cleanedValues.insert(32, 'REVENUE_TYPE', REVENUE_TYPE)
    #cleanedValues.insert(33, 'PURPOSE_CLASS', PURPOSE_CLASS)

    binary = np.array(binary)
    binary = binary.reshape(-1, 1)

    x = cleanedValues.drop(['Minutes Elapsed'], axis = 1)
    y = binary

    featuresToEncode = list(x.select_dtypes(include = ['object']).columns)
    encodedData = pd.get_dummies(x, columns = featuresToEncode)

    featureNames = []
    for col in encodedData.columns:
        #featureNames.append(col.replace('_', ': '))
        featureNames.append(col)

    x_train, x_test, y_train, y_test = train_test_split(encodedData, y.ravel(), test_size = 0.2, random_state = 42)

    #over_x_train, over_x_test, over_y_train, over_y_test = train_test_split(encodedData, y.ravel(), test_size = 0.05, random_state = 42)

    print('Under fit data:')
    y_count = y_train.tolist()
    print('0:', y_count.count(0))
    print('1:', y_count.count(1))

    #undersample = RandomUnderSampler(sampling_strategy = 0.8)
    #x_train, y_train = undersample.fit_resample(over_x_train, over_y_train)
    #x_test, y_test = undersample.fit_resample(over_x_test, over_y_test)
    #
    #print('Resampled:')
    #y_count = y_train.tolist()
    #print('0:', y_count.count(0))
    #print('1:', y_count.count(1))


    clf = RandomForestClassifier()
    parameters = [{'n_estimators': range (100, 150, 25), 'max_depth': range(10, 20, 5), 'min_samples_leaf': range(10, 20, 5), 'max_features': [0.4, 0.5], 'criterion': ['gini', 'entropy']}]
    gs = GridSearchCV(estimator = clf, param_grid = parameters, cv = 2, n_jobs = -1, verbose = 3)

    trees = 5
    #rfc = RandomForestClassifier(n_estimators = trees, criterion = 'gini', max_depth = 24, min_samples_leaf = 60, max_features = 0.5, n_jobs = -1, random_state = 42, verbose = 5)
    rfc = RandomForestClassifier(n_estimators=trees, criterion='gini', max_depth=25, min_samples_leaf=50, max_features=0.5, max_samples=0.5, max_leaf_nodes=20, n_jobs=-1, random_state=42, class_weight='balanced_subsample', verbose=5)
    #rfc = RandomForestClassifier(n_estimators=trees, criterion='gini', max_depth=50, min_samples_leaf=50, max_features=0.4, max_samples=0.4, max_leaf_nodes=25, n_jobs=-1, random_state=42, class_weight='balanced_subsample', oob_score=True, ccp_alpha = 0.02, verbose=5)
    rfc.fit(x_train, y_train)
    rfcPrediction = rfc.predict(x_test)
    print('R2:', r2_score(y_test, rfcPrediction))
    print('Random Forest Regressor MSE:',  mean_squared_error(y_test, rfcPrediction))
    print('Random Forest Regressor MAE:', mean_absolute_error(y_test, rfcPrediction))
    rfcPrediction = rfcPrediction.tolist()
    print('Late trades:', rfcPrediction.count(1))

    #print(gs.best_params_)

    cf_matrix = confusion_matrix(y_test, rfcPrediction)

    '''
    group_names = ['True Negative','False Positive','False Negative', 'True Positive']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    cmPlot = plt.figure()
    sns.heatmap(cf_matrix, annot = labels, fmt = '', cmap = 'Blues', xticklabels = ['On Time Trade', 'Late Trade'], yticklabels = ['On Time Trade', 'Late Trade'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Trade', rotation = 'horizontal')
    plt.ylabel('Actual Trade')
    '''

    print('Recall Score: %.3f' % recall_score(y_test, rfcPrediction))
    print('Precision Score: %.3f' % precision_score(y_test, rfcPrediction))
    print('Accuracy Score: %.3f' % accuracy_score(y_test, rfcPrediction))
    print('F1 Score: %.3f' % f1_score(y_test, rfcPrediction))

    testTotal = cf_matrix[0][0] + cf_matrix[1][0] + cf_matrix[0][1] + cf_matrix[1][1]

    binaryHold = list(binary)
    minutesElapsedThreshold.append(minutesElapsedCount)
    recallScores.append(recall_score(y_test, rfcPrediction))
    precisionScores.append(precision_score(y_test, rfcPrediction))
    accuracyScores.append(accuracy_score(y_test, rfcPrediction))
    f1Scores.append(f1_score(y_test, rfcPrediction))
    onTimeCount.append(binaryHold.count(0))
    lateCount.append(binaryHold.count(1))
    trueNegative.append(cf_matrix[0][0])
    trueNegativePercent.append((cf_matrix[0][0]/testTotal)*100)
    falseNegative.append(cf_matrix[1][0])
    falseNegativePercent.append((cf_matrix[1][0]/testTotal)*100)
    falsePostive.append(cf_matrix[0][1])
    falsePostivePercent.append((cf_matrix[0][1]/testTotal)*100)
    truePostive.append(cf_matrix[1][1])
    truePostivePercent.append((cf_matrix[1][1]/testTotal)*100)
    lateTradeCount.append(rfcPrediction.count(1))

statData = pd.DataFrame({'Minutes Elapsed': minutesElapsedThreshold})
statData.insert(1, 'Recall', recallScores)
statData.insert(2, 'Precision', precisionScores)
statData.insert(3, 'Accuracy', accuracyScores)
statData.insert(4, 'F1', f1Scores)
statData.insert(5, 'On Time Count', onTimeCount)
statData.insert(6, 'Late Count', lateCount)
statData.insert(7, 'True Negative', trueNegative)
statData.insert(8, 'True Negative Percent', trueNegativePercent)
statData.insert(9, 'False Negative', falseNegative)
statData.insert(10, 'False Negative Percent', falseNegativePercent)
statData.insert(11, 'False Positive', falsePostive)
statData.insert(12, 'False Positive Percent', falsePostivePercent)
statData.insert(13, 'True Positive', truePostive)
statData.insert(14, 'True Positive Percent', truePostivePercent)
statData.to_excel(f'{outputPath}/statData.xlsx', index = False)

statPlot = plt.figure()
plt.title('Minutes Elapsed vs Metrics')
plt.xlabel('Minutes Elapsed')
plt.ylabel('Metrics')
plt.plot(minutesElapsedThreshold, recallScores, linewidth = 1, label = 'Recall Score')
plt.plot(minutesElapsedThreshold, precisionScores, linewidth = 1, label = 'Precision Score')
plt.plot(minutesElapsedThreshold, accuracyScores, linewidth = 1, label = 'Accuracy Score')
plt.plot(minutesElapsedThreshold, f1Scores, linewidth = 1, label = 'F1 Score')
plt.legend(loc = 'best', fancybox = True, shadow = True)
plt.xticks(np.arange(min(minutesElapsedThreshold), max(minutesElapsedThreshold) + 0.2, 0.1))

statPlot2 = plt.figure()
plt.title('Minutes Elapsed vs Metrics')
plt.xlabel('Minutes Elapsed')
plt.ylabel('Metrics')
plt.plot(minutesElapsedThreshold, recallScores, linewidth = 1, label = 'Recall Score', marker = 'o', drawstyle = 'steps-post')
plt.plot(minutesElapsedThreshold, precisionScores, linewidth = 1, label = 'Precision Score', marker = 'o', drawstyle = 'steps-post')
plt.plot(minutesElapsedThreshold, accuracyScores, linewidth = 1, label = 'Accuracy Score', marker = 'o', drawstyle = 'steps-post')
plt.plot(minutesElapsedThreshold, f1Scores, linewidth = 1, label = 'F1 Score', marker = 'o', drawstyle = 'steps-post')
plt.legend(loc = 'best', fancybox = True, shadow = True)
plt.xticks(np.arange(min(minutesElapsedThreshold), max(minutesElapsedThreshold) + 0.2, 0.1))

print('--- %s seconds ---' % (time.time() - start_time))

plt.show()
