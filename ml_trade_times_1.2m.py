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

pathOne = r'C:\Users\mweiss\Python Projects\ML Trade Times\trade_data_A.xlsx'
dataOne = pd.read_excel(pathOne, sheet_name = 'trade_report_time_elapsed_data')
pathTwo = r'C:\Users\mweiss\Python Projects\ML Trade Times\trade_data_B.xlsx'
dataTwo = pd.read_excel(pathTwo, sheet_name = 'trade_report_time_elapsed_data')
outputPath = r'C:\Users\mweiss\Python Projects\ML Trade Times'

binary_one = dataOne['CLASS']
minutesElapsed_one = dataOne['MINUTES_ELAPSED']
timeOfTrade_one = dataOne['TOT']
recordType_one = dataOne['RSRF.record_type']
dealerRank_one = dataOne['dealer_rank']
dealerQuintile_one = dataOne['dealer_quintile']
price_one = dataOne['RSRF.dollar_price']
parAmount_one = dataOne['RSRF.par']
capacity_one = dataOne['RSRF.capacity']
msrb_issue_status_ind_one = dataOne['RSRF.msrb_issue_status_ind']
specialOne_one = dataOne['RSRF.special_price_reason_1']
specialTwo_one = dataOne['RSRF.special_price_reason_2']
specialThree_one = dataOne['RSRF.special_price_reason_3']
settlementDays_one = dataOne['RSRF.settlement_days']
OFF_SETTLE_DATE_YEAR_one = dataOne['MCSM.OFF_SETTLE_DATE_YEAR']
MAT_DATE_YEAR_one = dataOne['MCSM.QS_MAT_DATE_YEAR']
DTD_TO_MTY_YEARS_one = dataOne['DTD_TO_MTY_YEARS']
YEARS_SINCE_DTD_one = dataOne['YEARS_SINCE_DTD']
bondAgeFraction_one = dataOne['FRACTION_OF_LIFE_OF_BOND']
roundedBondAgeFraction_one = dataOne['ROUNDED_FRACTION_OF_LIFE_OF_BOND']
QS_COUPON_RATE_one = dataOne['MCSM.QS_COUPON_RATE']
INT_PAY_FREQ_STD_PERIOD_one = dataOne['MCSM.INT_PAY_FREQ_STD_PERIOD']
INT_ACCRUAL_COUNT_one = dataOne['MCSM.INT_ACCRUAL_COUNT']
#monthsToNextCall_one = dataOne['MONTHS_TO_NEXT_CALL']
CALL_FREQ_CODE_one = dataOne['MCSM.CALL_FREQ_CODE']
CALL_TYPE_one = dataOne['MCSM.CALL_TYPE']
OFF_WI_IND_one = dataOne['MCSM.OFF_WI_IND']
OFF_SECURITY_TYPE_one = dataOne['MCSM.OFF_SECURITY_TYPE']
IND_INT_TYPE_one = dataOne['MCSM.IND_INT_TYPE']
DEFAULT_IND_one = dataOne['MCSM.DEFAULT_IND']
COMPLETE_CALL_IND_one = dataOne['MCSM.COMPLETE_CALL_IND']
MAT_DENOM_one = dataOne['MCSM.AM_MAT_DENOM']
PUT_EXISTS_IND_one = dataOne['MCSM.PUT_EXISTS_IND']
CSB_ISSUE_TRANS_one = dataOne['MCSM.CSB_ISSUE_TRANS']
ISSUE_STATUS_one = dataOne['MCSM.CSB_ISSUE_STATUS']
REVENUE_TYPE_one = dataOne['MCSM.REVENUE_TYPE']

binary_two = dataTwo['CLASS']
minutesElapsed_two = dataTwo['MINUTES_ELAPSED']
timeOfTrade_two = dataTwo['TOT']
recordType_two = dataTwo['RSRF.record_type']
dealerRank_two = dataTwo['dealer_rank']
dealerQuintile_two = dataTwo['dealer_quintile']
price_two = dataTwo['RSRF.dollar_price']
parAmount_two = dataTwo['RSRF.par']
capacity_two = dataTwo['RSRF.capacity']
msrb_issue_status_ind_two = dataTwo['RSRF.msrb_issue_status_ind']
specialOne_two = dataTwo['RSRF.special_price_reason_1']
specialTwo_two = dataTwo['RSRF.special_price_reason_2']
specialThree_two = dataTwo['RSRF.special_price_reason_3']
settlementDays_two = dataTwo['RSRF.settlement_days']
OFF_SETTLE_DATE_YEAR_two = dataTwo['MCSM.OFF_SETTLE_DATE_YEAR']
MAT_DATE_YEAR_two = dataTwo['MCSM.QS_MAT_DATE_YEAR']
DTD_TO_MTY_YEARS_two = dataTwo['DTD_TO_MTY_YEARS']
YEARS_SINCE_DTD_two = dataTwo['YEARS_SINCE_DTD']
bondAgeFraction_two = dataTwo['FRACTION_OF_LIFE_OF_BOND']
roundedBondAgeFraction_two = dataTwo['ROUNDED_FRACTION_OF_LIFE_OF_BOND']
QS_COUPON_RATE_two = dataTwo['MCSM.QS_COUPON_RATE']
INT_PAY_FREQ_STD_PERIOD_two = dataTwo['MCSM.INT_PAY_FREQ_STD_PERIOD']
INT_ACCRUAL_COUNT_two = dataTwo['MCSM.INT_ACCRUAL_COUNT']
#monthsToNextCall_two = dataTwo['MONTHS_TO_NEXT_CALL']
CALL_FREQ_CODE_two = dataTwo['MCSM.CALL_FREQ_CODE']
CALL_TYPE_two = dataTwo['MCSM.CALL_TYPE']
OFF_WI_IND_two = dataTwo['MCSM.OFF_WI_IND']
OFF_SECURITY_TYPE_two = dataTwo['MCSM.OFF_SECURITY_TYPE']
IND_INT_TYPE_two = dataTwo['MCSM.IND_INT_TYPE']
DEFAULT_IND_two = dataTwo['MCSM.DEFAULT_IND']
COMPLETE_CALL_IND_two = dataTwo['MCSM.COMPLETE_CALL_IND']
MAT_DENOM_two = dataTwo['MCSM.AM_MAT_DENOM']
PUT_EXISTS_IND_two = dataTwo['MCSM.PUT_EXISTS_IND']
CSB_ISSUE_TRANS_two = dataTwo['MCSM.CSB_ISSUE_TRANS']
ISSUE_STATUS_two = dataTwo['MCSM.CSB_ISSUE_STATUS']
REVENUE_TYPE_two = dataTwo['MCSM.REVENUE_TYPE']

binary_one = binary_one.tolist()
minutesElapsed_one = minutesElapsed_one.tolist()
timeOfTrade_one = timeOfTrade_one.tolist()
recordType_one = recordType_one.tolist()
dealerRank_one = dealerRank_one.tolist()
dealerQuintile_one = dealerQuintile_one.tolist()
price_one = price_one.tolist()
parAmount_one = parAmount_one.tolist()
capacity_one = capacity_one.tolist()
msrb_issue_status_ind_one = msrb_issue_status_ind_one.tolist()
specialOne_one = specialOne_one.tolist()
specialTwo_one = specialTwo_one.tolist()
specialThree_one = specialThree_one.tolist()
settlementDays_one = settlementDays_one.tolist()
OFF_SETTLE_DATE_YEAR_one = OFF_SETTLE_DATE_YEAR_one.tolist()
MAT_DATE_YEAR_one = MAT_DATE_YEAR_one.tolist()
DTD_TO_MTY_YEARS_one = DTD_TO_MTY_YEARS_one.tolist()
YEARS_SINCE_DTD_one = YEARS_SINCE_DTD_one.tolist()
bondAgeFraction_one = bondAgeFraction_one.tolist()
roundedBondAgeFraction_one = roundedBondAgeFraction_one.tolist()
QS_COUPON_RATE_one = QS_COUPON_RATE_one.tolist()
INT_PAY_FREQ_STD_PERIOD_one = INT_PAY_FREQ_STD_PERIOD_one.tolist()
INT_ACCRUAL_COUNT_one = INT_ACCRUAL_COUNT_one.tolist()
#monthsToNextCall_one = monthsToNextCall_one.tolist()
CALL_FREQ_CODE_one = CALL_FREQ_CODE_one.tolist()
CALL_TYPE_one = CALL_TYPE_one.tolist()
OFF_WI_IND_one = OFF_WI_IND_one.tolist()
OFF_SECURITY_TYPE_one = OFF_SECURITY_TYPE_one.tolist()
IND_INT_TYPE_one = IND_INT_TYPE_one.tolist()
DEFAULT_IND_one = DEFAULT_IND_one.tolist()
COMPLETE_CALL_IND_one = COMPLETE_CALL_IND_one.tolist()
MAT_DENOM_one = MAT_DENOM_one.tolist()
PUT_EXISTS_IND_one = PUT_EXISTS_IND_one.tolist()
CSB_ISSUE_TRANS_one = CSB_ISSUE_TRANS_one.tolist()
ISSUE_STATUS_one = ISSUE_STATUS_one.tolist()
REVENUE_TYPE_one = REVENUE_TYPE_one.tolist()

binary_two = binary_two.tolist()
minutesElapsed_two = minutesElapsed_two.tolist()
timeOfTrade_two = timeOfTrade_two.tolist()
recordType_two = recordType_two.tolist()
dealerRank_two = dealerRank_two.tolist()
dealerQuintile_two = dealerQuintile_two.tolist()
price_two = price_two.tolist()
parAmount_two = parAmount_two.tolist()
capacity_two = capacity_two.tolist()
msrb_issue_status_ind_two = msrb_issue_status_ind_two.tolist()
specialOne_two = specialOne_two.tolist()
specialTwo_two = specialTwo_two.tolist()
specialThree_two = specialThree_two.tolist()
settlementDays_two = settlementDays_two.tolist()
OFF_SETTLE_DATE_YEAR_two = OFF_SETTLE_DATE_YEAR_two.tolist()
MAT_DATE_YEAR_two = MAT_DATE_YEAR_two.tolist()
DTD_TO_MTY_YEARS_two = DTD_TO_MTY_YEARS_two.tolist()
YEARS_SINCE_DTD_two = YEARS_SINCE_DTD_two.tolist()
bondAgeFraction_two = bondAgeFraction_two.tolist()
roundedBondAgeFraction_two = roundedBondAgeFraction_two.tolist()
QS_COUPON_RATE_two = QS_COUPON_RATE_two.tolist()
INT_PAY_FREQ_STD_PERIOD_two = INT_PAY_FREQ_STD_PERIOD_two.tolist()
INT_ACCRUAL_COUNT_two = INT_ACCRUAL_COUNT_two.tolist()
#monthsToNextCall_two = monthsToNextCall_two.tolist()
CALL_FREQ_CODE_two = CALL_FREQ_CODE_two.tolist()
CALL_TYPE_two = CALL_TYPE_two.tolist()
OFF_WI_IND_two = OFF_WI_IND_two.tolist()
OFF_SECURITY_TYPE_two = OFF_SECURITY_TYPE_two.tolist()
IND_INT_TYPE_two = IND_INT_TYPE_two.tolist()
DEFAULT_IND_two = DEFAULT_IND_two.tolist()
COMPLETE_CALL_IND_two = COMPLETE_CALL_IND_two.tolist()
MAT_DENOM_two = MAT_DENOM_two.tolist()
PUT_EXISTS_IND_two = PUT_EXISTS_IND_two.tolist()
CSB_ISSUE_TRANS_two = CSB_ISSUE_TRANS_two.tolist()
ISSUE_STATUS_two = ISSUE_STATUS_two.tolist()
REVENUE_TYPE_two = REVENUE_TYPE_two.tolist()

minutesElapsed = minutesElapsed_one + minutesElapsed_two
binary = binary_one + binary_two
timeOfTrade = timeOfTrade_one + timeOfTrade_two
recordType = recordType_one + recordType_two
dealerRank = dealerRank_one + dealerRank_two
dealerQuintile = dealerQuintile_one + dealerQuintile_two
price = price_one + price_two
parAmount = parAmount_one + parAmount_two
capacity = capacity_one + capacity_two
msrb_issue_status_ind = msrb_issue_status_ind_one + msrb_issue_status_ind_two
specialOne = specialOne_one + specialOne_two
specialTwo = specialTwo_one + specialTwo_two
specialThree = specialThree_one + specialThree_two
settlementDays = settlementDays_one + settlementDays_two
OFF_SETTLE_DATE_YEAR = OFF_SETTLE_DATE_YEAR_one + OFF_SETTLE_DATE_YEAR_two
MAT_DATE_YEAR = MAT_DATE_YEAR_one + MAT_DATE_YEAR_two
DTD_TO_MTY_YEARS = DTD_TO_MTY_YEARS_one + DTD_TO_MTY_YEARS_two
YEARS_SINCE_DTD = YEARS_SINCE_DTD_one + YEARS_SINCE_DTD_two
bondAgeFraction = bondAgeFraction_one + bondAgeFraction_two
roundedBondAgeFraction = roundedBondAgeFraction_one + roundedBondAgeFraction_two
QS_COUPON_RATE = QS_COUPON_RATE_one + QS_COUPON_RATE_two
INT_PAY_FREQ_STD_PERIOD = INT_PAY_FREQ_STD_PERIOD_one + INT_PAY_FREQ_STD_PERIOD_two
INT_ACCRUAL_COUNT = INT_ACCRUAL_COUNT_one + INT_ACCRUAL_COUNT_two
#monthsToNextCall = monthsToNextCall_one + monthsToNextCall_two
CALL_FREQ_CODE = CALL_FREQ_CODE_one + CALL_FREQ_CODE_two
CALL_TYPE = CALL_TYPE_one + CALL_TYPE_two
OFF_WI_IND = OFF_WI_IND_one + OFF_WI_IND_two
OFF_SECURITY_TYPE = OFF_SECURITY_TYPE_one + OFF_SECURITY_TYPE_two
IND_INT_TYPE = IND_INT_TYPE_one + IND_INT_TYPE_two
DEFAULT_IND = DEFAULT_IND_one + DEFAULT_IND_two
COMPLETE_CALL_IND = COMPLETE_CALL_IND_one + COMPLETE_CALL_IND_two
MAT_DENOM = MAT_DENOM_one + MAT_DENOM_two
PUT_EXISTS_IND = PUT_EXISTS_IND_one + PUT_EXISTS_IND_two
CSB_ISSUE_TRANS = CSB_ISSUE_TRANS_one + CSB_ISSUE_TRANS_two
ISSUE_STATUS = ISSUE_STATUS_one + ISSUE_STATUS_two
REVENUE_TYPE = REVENUE_TYPE_one + REVENUE_TYPE_two

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

# monthsToNextCallHold = []
# for i in range(len(monthsToNextCall)):
#     monthsToNextCall[i] = str(monthsToNextCall[i])
#     if(monthsToNextCall[i] != 'nan'):
#         monthsToNextCall[i] = float(monthsToNextCall[i])
#         monthsToNextCallHold.append(monthsToNextCall[i])
# monthsToNextCallAvg = math.floor(sum(monthsToNextCallHold) / len(monthsToNextCallHold))
# for i in range(len(monthsToNextCall)):
#     if(monthsToNextCall[i] == 'nan'):
#         monthsToNextCall[i] = monthsToNextCallAvg
#         monthsToNextCall[i] = float(monthsToNextCall[i])

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
cleanedValues.insert(12, 'MAT DATE YEAR', MAT_DATE_YEAR)
cleanedValues.insert(13, 'OFF SETTLE DATE YEAR', OFF_SETTLE_DATE_YEAR)
cleanedValues.insert(14, 'DTD TO MTY YEARS', DTD_TO_MTY_YEARS)
cleanedValues.insert(15, 'YEARS SINCE DTD', YEARS_SINCE_DTD)
cleanedValues.insert(16, 'Bond Age Fraction', bondAgeFraction)
cleanedValues.insert(17, 'QS COUPON RATE', QS_COUPON_RATE)
cleanedValues.insert(18, 'INT_PAY_FREQ_STD_PERIOD', INT_PAY_FREQ_STD_PERIOD)
cleanedValues.insert(19, 'INT_ACCRUAL_COUNT', INT_ACCRUAL_COUNT)
#cleanedValues.insert(20, 'MONTHS TO NEXT CALL', monthsToNextCall)
cleanedValues.insert(20, 'CALL_FREQ_CODE', CALL_FREQ_CODE)
cleanedValues.insert(21, 'CALL_TYPE', CALL_TYPE)
cleanedValues.insert(22, 'OFF_WI_IND', OFF_WI_IND)
cleanedValues.insert(23, 'OFF_SECURITY_TYPE', OFF_SECURITY_TYPE)
cleanedValues.insert(24, 'IND_INT_TYPE', IND_INT_TYPE)
cleanedValues.insert(25, 'DEFAULT_IND', DEFAULT_IND)
cleanedValues.insert(26, 'COMPLETE_CALL_IND', COMPLETE_CALL_IND)
cleanedValues.insert(27, 'MAT DENOM', MAT_DENOM)
cleanedValues.insert(28, 'PUT_EXISTS_IND', PUT_EXISTS_IND)
cleanedValues.insert(29, 'CSB_ISSUE_TRANS', CSB_ISSUE_TRANS)
cleanedValues.insert(30, 'ISSUE_STATUS', ISSUE_STATUS)
cleanedValues.insert(31, 'REVENUE_TYPE', REVENUE_TYPE)

minutesElapsed = np.array(minutesElapsed)
minutesElapsed = minutesElapsed.reshape(-1, 1)
binary = np.array(binary)
binary = binary.reshape(-1, 1)

x = cleanedValues.drop(['Class'], axis = 1)
y = minutesElapsed

featuresToEncode = list(x.select_dtypes(include = ['object']).columns)
encodedData = pd.get_dummies(x, columns = featuresToEncode)

vtcList = []
commonFNList = []
dfLen = len(encodedData)
for col in encodedData.columns:
  appList = []
  valueToCount = (encodedData[col].value_counts().idxmax())
  maxCount = encodedData[encodedData[col] == valueToCount].shape[0]
  if (maxCount/dfLen > 0.95):
    #appList.append
    commonFNList.append(col)

encodedData = encodedData.drop(commonFNList, axis = 1)

featureNames = []
for col in encodedData.columns:
  featureNames.append(col)

x_train, x_test, y_train, y_test = train_test_split(encodedData, y.ravel(), test_size = 0.2, random_state = 42)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_prediction =  lr.predict(x_test)
print('R2:', r2_score(y_test, y_prediction))
print('MSE:', mean_squared_error(y_test, y_prediction))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_prediction)))
print('MAE:', mean_absolute_error(y_test, y_prediction))
rawCoefs = pd.DataFrame(np.transpose(lr.coef_), columns = ['Coefficients'], index = featureNames)
rawCoefs.insert(1, 'Feature', featureNames)
rawCoefs.sort_values(inplace = True, by = 'Coefficients', ascending = False)
lrPlot = plt.figure()
rawCoefs.plot.barh(fontsize = 7)
plt.title('Linear Regression Coefficient Bar Plot')
plt.axvline(x = 0, color = 'gray')
plt.xlabel('Raw Coefficient Values')

tradeCount = []
allFeatureNames = []
for (columnName, columnData) in encodedData.iteritems():
  dataHoldList = list(columnData.values)
  if('_' in columnName):
    tradeCount.append(dataHoldList.count(1))
    allFeatureNames.append(columnName)
  else:
    tradeCount.append(len(dataHoldList))
    allFeatureNames.append(columnName)

column_names = ['Feature Name', 'Coefficient', 'Trade Count']
coefs = pd.DataFrame(columns = column_names)
for featureName in range(len(allFeatureNames)):
  for coef in range(len(rawCoefs)):
    if(rawCoefs.iloc[coef, 1] == allFeatureNames[featureName]):
      coefs = coefs.append({'Feature Name': allFeatureNames[featureName], 'Coefficient': round(rawCoefs.iloc[coef, 0], 2), 'Trade Count': tradeCount[featureName], '%': round((tradeCount[featureName]/len(dataHoldList))*100, 2)}, ignore_index = True)

coefs = coefs.sort_values(by = ['Coefficient'], ascending = True)
#coefs.to_excel(f'{outputPath}/coefs.xlsx', index = False)

#plt.show()

print('--- %s seconds ---' % (time.time() - start_time))

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

# monthsToNextCallCat = []
# for i in range(len(monthsToNextCall)):
#     monthsToNextCall[i] = str(monthsToNextCall[i])
#     if(monthsToNextCall[i] == 'nan'):
#         monthsToNextCallCat.append('n/a')
#     else:
#         monthsToNextCall[i] = float(monthsToNextCall[i])
#     if(type(monthsToNextCall[i]) == float):
#         if(monthsToNextCall[i] < 50):
#                 monthsToNextCallCat.append('Less than 50')
#         if((monthsToNextCall[i] >= 50) and (monthsToNextCall[i] <= 85)):
#                 monthsToNextCallCat.append('Between 50 and 85')
#         if(monthsToNextCall[i] > 85):
#                 monthsToNextCallCat.append('More than 85')

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
minutesElapsed = minutesElapsed_one + minutesElapsed_two
minutesElapsedSeries = pd.Series(minutesElapsed)

for minutesElapsedCount in np.arange(0.1, 2.6, 0.1):
    minutesElapsedThreshold.append(minutesElapsedCount)
    for i in range(len(binary)):
        if(minutesElapsed[i] < minutesElapsedCount):
            binary[i] = 0
    print('Minutes Elapsed Threshold:', minutesElapsedCount)
    cleanedValues = pd.DataFrame({'Minutes Elapsed': minutesElapsedSeries})
    cleanedValues.insert(1, 'Time of Trade', timeOfTradeCat)
    cleanedValues.insert(2, 'Record Type', recordType)
    cleanedValues.insert(3, 'Dealer Quintile', dealerQuintile)
    cleanedValues.insert(4, 'Price', priceCat)
    cleanedValues.insert(5, 'Par Amount', parAmountCat)
    cleanedValues.insert(6, 'Capacity', capacity)
    cleanedValues.insert(7, 'msrb_issue_status_ind', msrb_issue_status_ind)
    cleanedValues.insert(8, 'Special One', specialOne)
    cleanedValues.insert(9, 'Special Three', specialThree)
    cleanedValues.insert(10, 'Settlement Days', settlementDaysCat)
    cleanedValues.insert(11, 'MAT_DATE_YEAR', MAT_DATE_YEAR)
    cleanedValues.insert(12, 'DTD_TO_MTY_YEARS', yearsToMatureCat)
    cleanedValues.insert(13, 'YEARS_SINCE_DTD', bondAgeCat)
    cleanedValues.insert(14, 'Bond Age Fraction', bondAgeFractionCat)
    cleanedValues.insert(15, 'QS_COUPON_RATE', couponRateCat)
    cleanedValues.insert(16, 'INT_PAY_FREQ_STD_PERIOD', INT_PAY_FREQ_STD_PERIOD)
    cleanedValues.insert(17, 'INT_ACCRUAL_COUNT', INT_ACCRUAL_COUNT)
    #cleanedValues.insert(18, 'MONTHS_TO_NEXT_CALL', monthsToNextCallCat)
    cleanedValues.insert(18, 'CALL_FREQ_CODE', CALL_FREQ_CODE)
    cleanedValues.insert(19, 'CALL_TYPE', CALL_TYPE)
    cleanedValues.insert(20, 'OFF_WI_IND', OFF_WI_IND)
    cleanedValues.insert(21, 'OFF_SECURITY_TYPE', OFF_SECURITY_TYPE)
    cleanedValues.insert(22, 'IND_INT_TYPE', IND_INT_TYPE)
    cleanedValues.insert(23, 'DEFAULT_IND', DEFAULT_IND)
    cleanedValues.insert(24, 'COMPLETE_CALL_IND', COMPLETE_CALL_IND)
    cleanedValues.insert(25, 'PUT_EXISTS_IND', PUT_EXISTS_IND)
    cleanedValues.insert(26, 'CSB_ISSUE_TRANS', CSB_ISSUE_TRANS)
    cleanedValues.insert(27, 'ISSUE_STATUS', ISSUE_STATUS)
    cleanedValues.insert(28, 'REVENUE_TYPE', REVENUE_TYPE)

    binary = np.array(binary)
    binary = binary.reshape(-1, 1)

    x = cleanedValues.drop(['Minutes Elapsed'], axis = 1)
    y = binary

    featuresToEncode = list(x.select_dtypes(include = ['object']).columns)
    encodedData = pd.get_dummies(x, columns = featuresToEncode)

    featureNames = []
    for col in encodedData.columns:
        featureNames.append(col)

    over_x_train, over_x_test, over_y_train, over_y_test = train_test_split(encodedData, y.ravel(), test_size = 0.2, random_state = 42)

    print('Under fit data:')
    y_count = over_y_train.tolist()
    print('0:', y_count.count(0))
    print('1:', y_count.count(1))

    undersample = RandomUnderSampler(sampling_strategy = 1)
    x_train, y_train = undersample.fit_resample(over_x_train, over_y_train)
    x_test, y_test = undersample.fit_resample(over_x_test, over_y_test)

    print('Resampled:')
    y_count = y_train.tolist()
    print('0:', y_count.count(0))
    print('1:', y_count.count(1))

    # clf = RandomForestClassifier()
    # parameters = [{'n_estimators': range (100, 150, 25), 'max_depth': range(10, 20, 5), 'min_samples_leaf': range(10, 20, 5), 'max_features': [0.4, 0.5], 'criterion': ['gini', 'entropy']}]
    # gs = GridSearchCV(estimator = clf, param_grid = parameters, cv = 2, n_jobs = -1, verbose = 3)

    trees = 5
    rfc = RandomForestClassifier(n_estimators=trees, criterion='gini', max_depth=25, min_samples_leaf=50, max_features=0.5, max_samples=0.5, max_leaf_nodes=20, n_jobs=-1, random_state=42, class_weight='balanced_subsample')
    rfc.fit(x_train, y_train)
    rfcPrediction = rfc.predict(x_test)
    # print('R2:', r2_score(y_test, rfcPrediction))
    # print('Random Forest Regressor MSE:',  mean_squared_error(y_test, rfcPrediction))
    # print('Random Forest Regressor MAE:', mean_absolute_error(y_test, rfcPrediction))
    rfcPrediction = rfcPrediction.tolist()
    # print('Late trades:', rfcPrediction.count(1))

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

    # print('Recall Score: %.3f' % recall_score(y_test, rfcPrediction))
    # print('Precision Score: %.3f' % precision_score(y_test, rfcPrediction))
    # print('Accuracy Score: %.3f' % accuracy_score(y_test, rfcPrediction))
    # print('F1 Score: %.3f' % f1_score(y_test, rfcPrediction))

    testTotal = cf_matrix[0][0] + cf_matrix[1][0] + cf_matrix[0][1] + cf_matrix[1][1]

    binaryHold = list(binary)
    recallScores.append(recall_score(y_test, rfcPrediction))
    precisionScores.append(precision_score(y_test, rfcPrediction))
    accuracyScores.append(accuracy_score(y_test, rfcPrediction))
    f1Scores.append(f1_score(y_test, rfcPrediction))
    onTimeCount.append(y_count.count(0))
    lateCount.append(y_count.count(1))
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

'''
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
'''
print('--- %s seconds ---' % (time.time() - start_time))

plt.show()
