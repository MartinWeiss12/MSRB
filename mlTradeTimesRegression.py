import math
import time
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
import matplotlib.pyplot as plt
from dtreeviz.trees import dtreeviz
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

start_time = time.time()

#path = r'C:\Users\mweiss\Python Projects\ML Trade Times\new_600k_trade_data.xlsx'
path = r'/Users/martinweiss/Documents/Python/MSRB/ML Trade Times/new_600k_trade_data.xlsx'
#path = r'/content/trade_report_time_elapsed_data_all_1.xlsx'
#outputPath = r'C:\Users\mweiss\Python Projects\ML Trade Times'
outputPath = r'/Users/martinweiss/Documents/Python/MSRB/ML Trade Times'
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

for i in range(len(timeOfTrade)):
    timeOfTrade[i] = str(timeOfTrade[i])
    timeOfTrade[i] = timeOfTrade[i][0:5].replace(':', '.')
    timeOfTrade[i] = float(timeOfTrade[i])

for i in range(len(settlementDays)):
    if(settlementDays[i] < -1):
        settlementDays[i] = 2

for i in range(len(bondAgeFraction)):
  bondAgeFraction[i] = str(bondAgeFraction[i])
  if(bondAgeFraction[i] == 'nan'):
      bondAgeFraction[i] = 0
  bondAgeFraction[i] = float(bondAgeFraction[i])

for i in range(len(QS_COUPON_RATE)):
    QS_COUPON_RATE[i] = str(QS_COUPON_RATE[i])
    if(QS_COUPON_RATE[i] == 'nan'):
        QS_COUPON_RATE[i] = 5
    QS_COUPON_RATE[i] = float(QS_COUPON_RATE[i])

monthsToNextCallHold = []
for i in range(len(monthsToNextCall)):
    monthsToNextCall[i] = str(monthsToNextCall[i])
    if(monthsToNextCall[i] != 'nan'):
        monthsToNextCall[i] = float(monthsToNextCall[i])
        monthsToNextCallHold.append(monthsToNextCall[i])
monthsToNextCallAvg = math.floor(sum(monthsToNextCallHold) / len(monthsToNextCallHold))
for i in range(len(monthsToNextCall)):
    if(monthsToNextCall[i] == 'nan'):
        monthsToNextCall[i] = monthsToNextCallAvg
        monthsToNextCall[i] = float(monthsToNextCall[i])

for i in range(len(CALL_FREQ_CODE)):
    CALL_FREQ_CODE[i] = str(CALL_FREQ_CODE[i])
    if(CALL_FREQ_CODE[i] == 'nan'):
        CALL_FREQ_CODE[i] = 'n/a'

for i in range(len(CALL_TYPE)):
    CALL_TYPE[i] = str(CALL_TYPE[i])
    if(CALL_TYPE[i] == 'nan'):
        CALL_TYPE[i] = 'n/a'

for i in range(len(CSB_ISSUE_TRANS)):
    CSB_ISSUE_TRANS[i] = str(CSB_ISSUE_TRANS[i])
    if(CSB_ISSUE_TRANS[i] == 'nan'):
        CSB_ISSUE_TRANS[i] = 'n/a'

cleanedValues = pd.DataFrame({'Class': binary})
cleanedValues.insert(1, 'Time of Trade', timeOfTrade)
cleanedValues.insert(2, 'Record Type', recordType)
cleanedValues.insert(3, 'Dealer Rank', dealerRank)
#cleanedValues.insert(4, 'Dealer Quintile', dealerQuintile)
cleanedValues.insert(4, 'Price', price)
cleanedValues.insert(5, 'Par Amount', parAmount)
cleanedValues.insert(6, 'Capacity', capacity)
cleanedValues.insert(7, 'msrb_issue_status_ind', msrb_issue_status_ind)
cleanedValues.insert(8, 'Special One', specialOne)
cleanedValues.insert(9, 'Special Two', specialTwo)
cleanedValues.insert(10, 'Special Three', specialThree)
cleanedValues.insert(11, 'Settlement Days', settlementDays)
cleanedValues.insert(12, 'MAT_DATE_YEAR', MAT_DATE_YEAR)
cleanedValues.insert(13, 'OFF_SETTLE_DATE_YEAR', OFF_SETTLE_DATE_YEAR)
cleanedValues.insert(14, 'DTD_TO_MTY_YEARS', DTD_TO_MTY_YEARS)
cleanedValues.insert(15, 'YEARS_SINCE_DTD', YEARS_SINCE_DTD)
cleanedValues.insert(16, 'Bond Age Fraction', bondAgeFraction)
cleanedValues.insert(17, 'QS_COUPON_RATE', QS_COUPON_RATE)
cleanedValues.insert(18, 'INT_PAY_FREQ_STD_PERIOD', INT_PAY_FREQ_STD_PERIOD)
cleanedValues.insert(19, 'INT_ACCRUAL_COUNT', INT_ACCRUAL_COUNT)
cleanedValues.insert(20, 'MONTHS_TO_NEXT_CALL', monthsToNextCall)
cleanedValues.insert(21, 'CALL_FREQ_CODE', CALL_FREQ_CODE)
cleanedValues.insert(22, 'CALL_TYPE', CALL_TYPE)
cleanedValues.insert(23, 'OFF_WI_IND', OFF_WI_IND)
cleanedValues.insert(24, 'OFF_SECURITY_TYPE', OFF_SECURITY_TYPE)
cleanedValues.insert(25, 'IND_INT_TYPE', IND_INT_TYPE)
cleanedValues.insert(26, 'DEFAULT_IND', DEFAULT_IND)
cleanedValues.insert(27, 'COMPLETE_CALL_IND', COMPLETE_CALL_IND)
cleanedValues.insert(28, 'MAT_DENOM', MAT_DENOM)
cleanedValues.insert(29, 'PUT_EXISTS_IND', PUT_EXISTS_IND)
cleanedValues.insert(30, 'CSB_ISSUE_TRANS', CSB_ISSUE_TRANS)
cleanedValues.insert(31, 'ISSUE_STATUS', ISSUE_STATUS)
cleanedValues.insert(32, 'REVENUE_TYPE', REVENUE_TYPE)

minutesElapsed = np.array(minutesElapsed)
minutesElapsed = minutesElapsed.reshape(-1, 1)
binary = np.array(binary)
binary = binary.reshape(-1, 1)

x = cleanedValues.drop(['Class'], axis = 1)
y = minutesElapsed

featuresToEncode = list(x.select_dtypes(include = ['object']).columns)
encodedData = pd.get_dummies(x, columns = featuresToEncode)
featureNames = []
for col in encodedData.columns:
  #featureNames.append(col.replace('_', ': '))
  featureNames.append(col)

x_train, x_test, y_train, y_test = train_test_split(encodedData, y.ravel(), test_size = 0.2, random_state = 42)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_prediction =  lr.predict(x_test)
print('R2:', r2_score(y_test, y_prediction))
print('MSE:', mean_squared_error(y_test, y_prediction))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_prediction)))
print('MAE:', mean_absolute_error(y_test, y_prediction))
coefs = pd.DataFrame(np.transpose(lr.coef_), columns = ['Coefficients'], index = featureNames)
coefs.insert(1, 'Feature', featureNames)
coefs.sort_values(inplace = True, by = 'Coefficients', ascending = False)
coefs.to_excel(f'{outputPath}/coefs.xlsx', index = False)
lrPlot = plt.figure()
coefs.plot.barh(fontsize = 7)
plt.title('Linear Regression Coefficient Bar Plot')
plt.axvline(x = 0, color = 'gray')
plt.xlabel('Raw Coefficient Values')

print('--- %s seconds ---' % (time.time() - start_time))

plt.show()


'''
rfr = RandomForestRegressor()
parameters = [{'n_estimators': range (100, 150, 25), 'max_depth': range(4, 6, 1), 'min_samples_leaf': range(4, 6, 1), 'max_features': [0.4, 0.5], 'criterion': ['poisson', 'squared_error']}]
gs = GridSearchCV(estimator = rfr, param_grid = parameters, cv = 10, n_jobs = -1, verbose = 3)

trees = 3000
rfr = RandomForestRegressor(n_estimators = trees, criterion = 'poisson', max_depth = 6, min_samples_leaf = 20, max_features = 0.4, n_jobs = -1, random_state = 42, verbose = 5)
rfr.fit(x_train, y_train)
rfrPrediction = rfr.predict(x_test)
print('Mean Absolute Error:', mean_absolute_error(y_test, rfrPrediction))
print('Mean Squared Error:', mean_squared_error(y_test, rfrPrediction))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, rfrPrediction)))

#print(gs.best_params_)

estimatorAccuracy = []
for currentEstimator in range(trees):
  estimatorAccuracy.append([currentEstimator, mean_squared_error(y_test, rfr.estimators_[currentEstimator].predict(x_test.values))])
estimatorAccuracy = pd.DataFrame(estimatorAccuracy, columns = ['Estimator Number','Mean Squared Error'])
estimatorAccuracy.sort_values(inplace = True, by = 'Mean Squared Error', ascending = True)
bestDecisionTree = rfr.estimators_[estimatorAccuracy.head(1)['Estimator Number'].values[0]]

print(estimatorAccuracy)

viz = dtreeviz(bestDecisionTree, x_test, y_test, target_name = 'Minutes Elapsed', feature_names = featureNames, title = 'Decision Tree')
'''