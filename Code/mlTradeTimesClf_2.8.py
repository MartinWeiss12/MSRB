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

path = r'C:\Users\mweiss\Python Projects\ML Trade Times\trade_report_time_elapsed_data.xlsx'
#path = r'/Users/martinweiss/Desktop/Python/MSRB/ML Trade Times/trade_report_time_elapsed_data.xlsx'
#path = r'/content/trade_report_time_elapsed_data.xlsx'
data = pd.read_excel(path, sheet_name = 'trade_report_time_elapsed_data')

binary = data['CLASS']
minutesElapsed = data['MINUTES_ELAPSED']
timeOfTrade = data['TOT']
recordType = data['RSRF.record_type']
dealerRank = data['dealer_rank']
dealerQuintile = data['dealer_quintile']
price = data['dollar_price']
bondsTraded = data['RSRF.par']
capacity = data['RSRF.capacity']
issueType = data['RSRF.msrb_issue_status_ind']
specialOne = data['RSRF.special_price_reason_1']
specialTwo = data['RSRF.special_price_reason_2']
specialThree = data['RSRF.special_price_reason_3']
settlementDays = data['RSRF.settlement_days']
issueYear = data['MCSM.OFF_SETTLE_DATE_YEAR']
yearsToMature = data['DTD_TO_MTY_YEARS']
bondAge = data['YEARS_SINCE_DTD']
bondAgeFraction = data['FRACTION_OF_LIFE_OF_BOND']
roundedBondAgeFraction = data['ROUNDED_FRACTION_OF_LIFE_OF_BOND']
interestRate = data['MCSM.QS_COUPON_RATE']
interestPaymentFreq = data['MCSM.INT_PAY_FREQ_STD_PERIOD']
interestMethod = data['MCSM.INT_ACCRUAL_COUNT']
monthsToNextCall = data['MONTHS_TO_NEXT_CALL']
callFreq = data['MCSM.CALL_FREQ_CODE']
callType = data['MCSM.CALL_TYPE']
vendorIssueType = data['MCSM.OFF_WI_IND']
securityType = data['MCSM.OFF_SECURITY_TYPE']
interestType = data['MCSM.IND_INT_TYPE']
defaultBond = data['MCSM.DEFAULT_IND']
completedCall = data['MCSM.COMPLETE_CALL_IND']
bondDenom = data['MCSM.AM_MAT_DENOM']
bondPuttable = data['MCSM.PUT_EXISTS_IND']
bondTrans = data['MCSM.CSB_ISSUE_TRANS']
bondStatus = data['MCSM.CSB_ISSUE_STATUS']
revenueType = data['MCSM.REVENUE_TYPE']

timeOfTrade = timeOfTrade.tolist()
recordType = recordType.tolist()
dealerRank = dealerRank.tolist()
dealerQuintile = dealerQuintile.tolist()
price = price.tolist()
bondsTraded = bondsTraded.tolist()
capacity = capacity.tolist()
issueType = issueType.tolist()
specialOne = specialOne.tolist()
specialTwo = specialTwo.tolist()
specialThree = specialThree.tolist()
settlementDays = settlementDays.tolist()
issueYear = issueYear.tolist()
yearsToMature = yearsToMature.tolist()
bondAge = bondAge.tolist()
bondAgeFraction = bondAgeFraction.tolist()
roundedBondAgeFraction = roundedBondAgeFraction.tolist()
interestRate = interestRate.tolist()
interestPaymentFreq = interestPaymentFreq.tolist()
interestMethod = interestMethod.tolist()
monthsToNextCall = monthsToNextCall.tolist()
callFreq = callFreq.tolist()
callType = callType.tolist()
vendorIssueType = vendorIssueType.tolist()
securityType = securityType.tolist()
interestType = interestType.tolist()
defaultBond = defaultBond.tolist()
completedCall = completedCall.tolist()
bondDenom = bondDenom.tolist()
bondPuttable = bondPuttable.tolist()
bondTrans = bondTrans.tolist()
bondStatus = bondStatus.tolist()
revenueType = revenueType.tolist()

for i in range(len(timeOfTrade)):
    timeOfTrade[i] = str(timeOfTrade[i])
    timeOfTrade[i] = timeOfTrade[i][0:5].replace(':', '.')
    timeOfTrade[i] = float(timeOfTrade[i])
    # if(timeOfTrade[i] < 10.45):
    #     timeOfTrade[i] = 1
    # if((timeOfTrade[i] >= 10.45) and (timeOfTrade[i] <= 12)):
    #     timeOfTrade[i] = 2
    # if((timeOfTrade[i] > 12) and (timeOfTrade[i] <= 13.15)):
    #     timeOfTrade[i] = 3
    # if(timeOfTrade[i] > 13.15):
    #     timeOfTrade[i] = 4

for i in range(len(recordType)):
    if(recordType[i] == 'CB'):
        recordType[i] = 1
    if(recordType[i] == 'CS'):
        recordType[i] = 2
    if(recordType[i] == 'ID'):
        recordType[i] = 3

# for i in range(len(dealerRank)):
#     if(dealerRank[i] < 4):
#         dealerRank[i] = 1
#     if((dealerRank[i] >= 4) and (dealerRank[i] <= 9)):
#         dealerRank[i] = 2
#     if((dealerRank[i] > 9) and (dealerRank[i] <= 21)):
#         dealerRank[i] = 3
#     if(dealerRank[i] > 21):
#         dealerRank[i] = 4
# dealerRank0List = []
# dealerRank1List = []
# dealerRank01 = 0
# dealerRank02 = 0
# dealerRank03 = 0
# dealerRank04 = 0
# dealerRank11 = 0
# dealerRank12 = 0
# dealerRank13 = 0
# dealerRank14 = 0
# for i in range(len(binary)):
#     if ((binary[i] == 0) and (dealerRank[i] == 1)):
#         dealerRank01 = dealerRank01 + 1
#     if ((binary[i] == 0) and (dealerRank[i] == 2)):
#         dealerRank02 = dealerRank02 + 1
#     if ((binary[i] == 0) and (dealerRank[i] == 3)):
#         dealerRank03 = dealerRank03 + 1
#     if ((binary[i] == 0) and (dealerRank[i] == 4)):
#         dealerRank04 = dealerRank04 + 1
#     if ((binary[i] == 1) and (dealerRank[i] == 1)):
#         dealerRank11 = dealerRank11 + 1
#     if ((binary[i] == 1) and (dealerRank[i] == 2)):
#         dealerRank12 = dealerRank12 + 1
#     if ((binary[i] == 1) and (dealerRank[i] == 3)):
#         dealerRank13 = dealerRank13 + 1
#     if ((binary[i] == 1) and (dealerRank[i] == 4)):
#         dealerRank14 = dealerRank14 + 1
# dealerRank0List.append(dealerRank01)
# dealerRank0List.append(dealerRank02)
# dealerRank0List.append(dealerRank03)
# dealerRank0List.append(dealerRank04)
# dealerRank1List.append(dealerRank11)
# dealerRank1List.append(dealerRank12)
# dealerRank1List.append(dealerRank13)
# dealerRank1List.append(dealerRank14)

for i in range(len(dealerQuintile)):
    if(dealerQuintile[i] == 'Top 1%'):
        dealerQuintile[i] = 1
    if(dealerQuintile[i] == 'Top 5%'):
        dealerQuintile[i] = 2
    if(dealerQuintile[i] == 'Top 20%'):
        dealerQuintile[i] = 3
    if((dealerQuintile[i] == '20% to 40%') or (dealerQuintile[i] == '40% to 60%') or (dealerQuintile[i] == '60% to 80%') or (dealerQuintile[i] == 'Bottom 20%')):
        dealerQuintile[i] = 4
dealerQuintile0List = []
dealerQuintile1List = []
dealerQuintile01 = 0
dealerQuintile02 = 0
dealerQuintile03 = 0
dealerQuintile04 = 0
dealerQuintile11 = 0
dealerQuintile12 = 0
dealerQuintile13 = 0
dealerQuintile14 = 0
for i in range(len(binary)):
    if ((binary[i] == 0) and (dealerQuintile[i] == 1)):
        dealerQuintile01 = dealerQuintile01 + 1
    if ((binary[i] == 0) and (dealerQuintile[i] == 2)):
        dealerQuintile02 = dealerQuintile02 + 1
    if ((binary[i] == 0) and (dealerQuintile[i] == 3)):
        dealerQuintile03 = dealerQuintile03 + 1
    if ((binary[i] == 0) and (dealerQuintile[i] == 4)):
        dealerQuintile04 = dealerQuintile04 + 1
    if ((binary[i] == 1) and (dealerQuintile[i] == 1)):
        dealerQuintile11 = dealerQuintile11 + 1
    if ((binary[i] == 1) and (dealerQuintile[i] == 2)):
        dealerQuintile12 = dealerQuintile12 + 1
    if ((binary[i] == 1) and (dealerQuintile[i] == 3)):
        dealerQuintile13 = dealerQuintile13 + 1
    if ((binary[i] == 1) and (dealerQuintile[i] == 4)):
        dealerQuintile14 = dealerQuintile14 + 1
dealerQuintile0List.append(dealerQuintile01)
dealerQuintile0List.append(dealerQuintile02)
dealerQuintile0List.append(dealerQuintile03)
dealerQuintile0List.append(dealerQuintile04)
dealerQuintile1List.append(dealerQuintile11)
dealerQuintile1List.append(dealerQuintile12)
dealerQuintile1List.append(dealerQuintile13)
dealerQuintile1List.append(dealerQuintile14)

#for i in range(len(price)):
    # if(price[i] < 100):
    #     price[i] = 1
    # if((price[i] >= 100) and (price[i] <= 103.36)):
    #     price[i] = 2
    # if((price[i] > 103.36) and (price[i] <= 107.625)):
    #     price[i] = 3
    # if(price[i] > 107.625):
    #     price[i] = 4

# for i in range(len(bondsTraded)):
#     if(bondsTraded[i] < 14999):
#         bondsTraded[i] = 1
#     if((bondsTraded[i] >= 14999) and (bondsTraded[i] <= 25000)):
#         bondsTraded[i] = 2
#     if((bondsTraded[i] > 25000) and (bondsTraded[i] <= 74500)):
#         bondsTraded[i] = 3
#     if(bondsTraded[i] > 74500):
#         bondsTraded[i] = 4
# bondsTraded0List = []
# bondsTraded1List = []
# bondsTraded01 = 0
# bondsTraded02 = 0
# bondsTraded03 = 0
# bondsTraded04 = 0
# bondsTraded11 = 0
# bondsTraded12 = 0
# bondsTraded13 = 0
# bondsTraded14 = 0
# for i in range(len(binary)):
#     if ((binary[i] == 0) and (bondsTraded[i] == 1)):
#         bondsTraded01 = bondsTraded01 + 1
#     if ((binary[i] == 0) and (bondsTraded[i] == 2)):
#         bondsTraded02 = bondsTraded02 + 1
#     if ((binary[i] == 0) and (bondsTraded[i] == 3)):
#         bondsTraded03 = bondsTraded03 + 1
#     if ((binary[i] == 0) and (bondsTraded[i] == 4)):
#         bondsTraded04 = bondsTraded04 + 1
#     if ((binary[i] == 1) and (bondsTraded[i] == 1)):
#         bondsTraded11 = bondsTraded11 + 1
#     if ((binary[i] == 1) and (bondsTraded[i] == 2)):
#         bondsTraded12 = bondsTraded12 + 1
#     if ((binary[i] == 1) and (bondsTraded[i] == 3)):
#         bondsTraded13 = bondsTraded13 + 1
#     if ((binary[i] == 1) and (bondsTraded[i] == 4)):
#         bondsTraded14 = bondsTraded14 + 1
# bondsTraded0List.append(bondsTraded01)
# bondsTraded0List.append(bondsTraded02)
# bondsTraded0List.append(bondsTraded03)
# bondsTraded0List.append(bondsTraded04)
# bondsTraded1List.append(bondsTraded11)
# bondsTraded1List.append(bondsTraded12)
# bondsTraded1List.append(bondsTraded13)
# bondsTraded1List.append(bondsTraded14)

for i in range(len(capacity)):
    if(capacity[i] == 'A'):
        capacity[i] = 1
    if(capacity[i] == 'P'):
        capacity[i] = 2

for i in range(len(issueType)):
    if(issueType[i] == 'RW'):
        issueType[i] = 1
    if(issueType[i] == 'WI'):
        issueType[i] = 2

for i in range(len(settlementDays)):
    if(settlementDays[i] < 2):
        settlementDays[i] = 1
    if(settlementDays[i] == 2):
        settlementDays[i] = 2
    if(settlementDays[i] > 2):
        settlementDays[i] = 3

for i in range(len(issueYear)):
    if(issueYear[i] < 2017):
        issueYear[i] = 1
    if((issueYear[i] >= 2017) and (issueYear[i] <= 2019)):
        issueYear[i] = 2
    if(issueYear[i] > 2019):
        issueYear[i] = 3

# for i in range(len(yearsToMature)):
#     if(yearsToMature[i] < 9):
#         yearsToMature[i] = 1
#     if((yearsToMature[i] >= 9) and (yearsToMature[i] <= 12)):
#         yearsToMature[i] = 2
#     if((yearsToMature[i] > 12) and (yearsToMature[i] <= 19)):
#         yearsToMature[i] = 3
#     if(yearsToMature[i] > 19):
#         yearsToMature[i] = 4

# for i in range(len(bondAge)):
#     if(bondAge[i] < 2):
#         bondAge[i] = 1
#     if((bondAge[i] >= 2) and (bondAge[i] <= 6)):
#         bondAge[i] = 2
#     if(bondAge[i] > 6):
#         bondAge[i] = 3

for i in range(len(bondAgeFraction)):
    bondAgeFraction[i] = str(bondAgeFraction[i])
    if(bondAgeFraction[i] == 'nan'):
        bondAgeFraction[i] = 0
    bondAgeFraction[i] = float(bondAgeFraction[i])

for i in range(len(interestRate)):
    interestRate[i] = str(interestRate[i])
    if(interestRate[i] == 'nan'):
        interestRate[i] = -1
    else:
        interestRate[i] = float(interestRate[i])
    if((interestRate[i] < 4) and (interestRate[i] > -1)):
        interestRate[i] = 1
    if((interestRate[i] >= 4) and (interestRate[i] <= 4.99)):
        interestRate[i] = 2
    if(interestRate[i] > 4.99):
        interestRate[i] = 3
    if(interestRate[i] == -1):
        interestRate[i] = 4

for i in range(len(interestPaymentFreq)):
    if(interestPaymentFreq[i] != 'SA'):
        interestPaymentFreq[i] = 1
    if(interestPaymentFreq[i] == 'SA'):
        interestPaymentFreq[i] = 2

for i in range(len(interestMethod)):
    if(interestMethod[i] != '030/360'):
        interestMethod[i] = 1
    if(interestMethod[i] == '030/360'):
        interestMethod[i] = 2

for i in range(len(monthsToNextCall)):
    if(monthsToNextCall[i] < 50):
        monthsToNextCall[i] = 1
    if((monthsToNextCall[i] >= 50) and (monthsToNextCall[i] <= 85)):
        monthsToNextCall[i] = 2
    if(monthsToNextCall[i] > 85):
        monthsToNextCall[i] = 3
    monthsToNextCall[i] = str(monthsToNextCall[i])
    if(monthsToNextCall[i] == 'nan'):
        monthsToNextCall[i] = 4
    monthsToNextCall[i] = float(monthsToNextCall[i])

for i in range(len(callFreq)):
    callFreq[i] = str(callFreq[i])
    if((callFreq[i] != 'AT') and (callFreq[i] != 'nan')):
        callFreq[i] = 1
    if(callFreq[i] == 'AT'):
        callFreq[i] = 2
    if(callFreq[i] == 'nan'):
        callFreq[i] = 3

for i in range(len(callType)):
    callType[i] = str(callType[i])
    if((callType[i] != 'CALL') and (callType[i] != 'NONC')):
        callType[i] = 1
    if(callType[i] == 'CALL'):
        callType[i] = 2
    if(callType[i] == 'NONC'):
        callType[i] = 3

for i in range(len(vendorIssueType)):
    if(vendorIssueType[i] == 'Y'):
        vendorIssueType[i] = 1
    if(vendorIssueType[i] == 'N'):
        vendorIssueType[i] = 2

for i in range(len(securityType)):
    if(securityType[i] != 'BOND'):
        securityType[i] = 1
    if(securityType[i] == 'BOND'):
        securityType[i] = 2

for i in range(len(interestType)):
    if(interestType[i] != 'F'):
        interestType[i] = 1
    if(interestType[i] == 'F'):
        interestType[i] = 2

for i in range(len(defaultBond)):
    if(defaultBond[i] == 'Y'):
        defaultBond[i] = 1
    if(defaultBond[i] == 'N'):
        defaultBond[i] = 2

for i in range(len(completedCall)):
    if(completedCall[i] == 'Y'):
        completedCall[i] = 1
    if(completedCall[i] == 'N'):
        completedCall[i] = 2

for i in range(len(bondDenom)):
    if(bondDenom[i] < 5000):
        bondDenom[i] = 1
    if(bondDenom[i] == 5000):
        bondDenom[i] = 2
    if(bondDenom[i] > 5000):
        bondDenom[i] = 3
    bondDenom[i] = str(bondDenom[i])
    if(bondDenom[i] == 'nan'):
        bondDenom[i] = 4
    bondDenom[i] = float(bondDenom[i])

for i in range(len(bondPuttable)):
    bondPuttable[i] = str(bondPuttable[i])
    if(bondPuttable[i] == 'Y'):
        bondPuttable[i] = 1
    if(bondPuttable[i] == 'N'):
        bondPuttable[i] = 2

for i in range(len(bondTrans)):
    if((bondTrans[i] != 'A') and (bondTrans[i] != 'L') and (bondTrans[i] != 'M')):
        bondTrans[i] = 1
    if(bondTrans[i] == 'A'):
        bondTrans[i] = 2
    if(bondTrans[i] == 'L'):
        bondTrans[i] = 3
    if(bondTrans[i] == 'M'):
        bondTrans[i] = 4

for i in range(len(bondStatus)):
    if(bondStatus[i] != 'A'):
        bondStatus[i] = 1
    if(bondStatus[i] == 'A'):
        bondStatus[i] = 2

for i in range(len(revenueType)):
    if((revenueType[i] != 'GO') and (revenueType[i] != 'REV')):
        revenueType[i] = 1
    if(revenueType[i] == 'GO'):
        revenueType[i] = 2
    if(revenueType[i] == 'REV'):
        revenueType[i] = 3

cleanedValues = pd.DataFrame({'Class': binary})
cleanedValues.insert(1, 'Time of Trade', timeOfTrade)
cleanedValues.insert(2, 'Record Type', recordType)
cleanedValues.insert(3, 'Dealer Rank', dealerRank)
cleanedValues.insert(4, 'Dealer Quintile', dealerQuintile)
cleanedValues.insert(5, 'Price', price)
cleanedValues.insert(6, 'Bonds Traded', bondsTraded)
#cleanedValues.insert(7, 'Capacity', capacity)
#cleanedValues.insert(8, 'Issue Type', issueType)
#cleanedValues.insert(9, 'Special One', specialOne)
#cleanedValues.insert(10, 'Special Two', specialTwo)
cleanedValues.insert(7, 'Special Three', specialThree)
cleanedValues.insert(8, 'Settlement Days', settlementDays)
cleanedValues.insert(9, 'Issue Year', issueYear)
cleanedValues.insert(10, 'Years to Mature', yearsToMature)
cleanedValues.insert(11, 'Bond Age', bondAge)
cleanedValues.insert(12, 'Bond Age Fraction', bondAgeFraction)
cleanedValues.insert(13, 'Interest Rate', interestRate)
#cleanedValues.insert(18, 'Interest Payment Frequency', interestPaymentFreq)
#cleanedValues.insert(19, 'Interest Method', interestMethod)
cleanedValues.insert(14, 'Months To Next Call', monthsToNextCall)
cleanedValues.insert(15, 'Call Frequency', callFreq)
cleanedValues.insert(16, 'Call Type', callType)
#cleanedValues.insert(21, 'Vendor Issue Type', vendorIssueType)
#cleanedValues.insert(22, 'Security Type', securityType)
#cleanedValues.insert(23, 'Interest Type', interestType)
#cleanedValues.insert(24, 'Default Bond', defaultBond)
#cleanedValues.insert(25, 'Completed Call', completedCall)
#cleanedValues.insert(17, 'Bond Denomination', bondDenom)
#cleanedValues.insert(29, 'Bond Puttable (Y/N)', bondPuttable)
cleanedValues.insert(17, 'Bond Transaction', bondTrans)
#cleanedValues.insert(28, 'Bond Status', bondStatus)
cleanedValues.insert(18, 'Revenue Type', revenueType)

'''
barPlot1 = plt.figure()
plt.bar(['Less than $100', '$100 - $103.35', '$103.36 - $107.625', 'Greater than $107.626'], price0List, 0.6, label = 'On Time')
plt.bar(['Less than $100', '$100 - $103.35', '$103.36 - $107.625', 'Greater than $107.626'], price1List, 0.6, bottom = price0List, label = 'Late')
plt.xlabel('Price (per $100)')
plt.ylabel('Count')
plt.title('Price Stacked Bar Chart')
plt.legend(loc = 1)

barPlot2 = plt.figure()
plt.bar(['Top 1%', 'Top 5%', 'Top 20%', 'Bottom 80%'], dealerQuintile0List, 0.6, label = 'On Time')
plt.bar(['Top 1%', 'Top 5%', 'Top 20%', 'Bottom 80%'], dealerQuintile1List, 0.6, bottom = dealerQuintile0List, label = 'Late')
plt.xlabel('Dealer Quintile')
plt.ylabel('Count')
plt.title('Dealer Quintile Stacked Bar Chart')
plt.legend(loc = 1)

barPlot3 = plt.figure()
plt.bar(['Less than 14,999', ' 14,999 - 25,000', ' 25,001 - 74,500', 'Greater than 74,501'], bondsTraded0List, 0.6, label = 'On Time')
plt.bar(['Less than 14,999', ' 14,999 - 25,000', ' 25,001 - 74,500', 'Greater than 74,501'], bondsTraded1List, 0.6, bottom = bondsTraded0List, label = 'Late')
plt.xlabel('Bonds Traded')
plt.ylabel('Count')
plt.title('Bonds Traded Stacked Bar Chart')
plt.legend(loc = 1)

# barPlot4 = plt.figure()
# plt.bar(['Under 3.6 months', '3.6 months - 6 months', 'Greater than 6 months'], binnedBondAgeFraction0List, 0.6, label = 'On Time')
# plt.bar(['Under 3.6 months', '3.6 months - 6 months', 'Greater than 6 months'], binnedBondAgeFraction1List, 0.6, bottom = binnedBondAgeFraction0List, label = 'Late')
# plt.xlabel('Bond Age Fraction')
# plt.ylabel('Count')
# plt.title('Bond Age Fraction Stacked Bar Chart')
# plt.legend(loc = 1)

barPlot5 = plt.figure()
plt.bar(['1 - 3', '4 - 9', '10 - 21', '22 - 475'], dealerRank0List, 0.6, label = 'On Time')
plt.bar(['1 - 3', '4 - 9', '10 - 21', '22 - 475'], dealerRank1List, 0.6, bottom = dealerRank0List, label = 'Late')
plt.xlabel('Dealer Rank')
plt.ylabel('Count')
plt.title('Dealer Rank Stacked Bar Chart')
plt.legend(loc = 1)
'''

minutesElapsed = np.array(minutesElapsed)
minutesElapsed = minutesElapsed.reshape(-1, 1)

binary = np.array(binary)
binary = binary.reshape(-1, 1)

x = cleanedValues.drop(['Class'], axis = 1)
y = minutesElapsed

x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), test_size = 0.1, random_state = 42)




df = pd.DataFrame({'DTD_TO_MTY_YEARS': yearsToMature})
df.insert(1, 'Bond Age Fraction', roundedBondAgeFraction)
df.insert(2, 'Class', binary)
groupedData = df.groupby(['DTD_TO_MTY_YEARS', 'Bond Age Fraction'], as_index = False).agg({'Class': ['sum', 'count']})
groupedData.insert(2, 'sum', groupedData['Class']['sum'])
groupedData.insert(3, 'count', groupedData['Class']['count'])
groupedData.insert(4, 'metric', ((groupedData['Class']['sum']/groupedData['Class']['count']) * 100) - 5)
#groupedData.insert(4, 'metric', min((((groupedData['Class']['sum']/groupedData['Class']['count']) * 100) - 5), 10))
for i in range(len(groupedData.index)):
    if(groupedData['metric'][i] > 10):
        groupedData.drop([i], axis = 0, inplace = True)
sns.set()
plot = sns.scatterplot(x = 'DTD_TO_MTY_YEARS', y = 'Bond Age Fraction', size = 'count', hue = 'metric', palette = 'icefire',  sizes = (0, 500), data = groupedData)
norm = plt.Normalize(groupedData['metric'].min(), groupedData['metric'].max())
sm = plt.cm.ScalarMappable(cmap = 'icefire', norm = norm)
sm.set_array([])
plot.get_legend().remove()
plot.figure.colorbar(sm, orientation = 'vertical', label = 'Percent of Delayed Reports')
plot.axis([-0.5, 36.5, -0.05, 1.05])
plt.title('DTD_TO_MTY_YEARS vs Bond Age Fraction Bubble Chart')
plt.show()


'''
lr = LinearRegression()
lr.fit(x_train, y_train)
y_prediction =  lr.predict(x_test)
print('R2:', r2_score(y_test, y_prediction))
print('MSE:', mean_squared_error(y_test, y_prediction))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_prediction)))
print('MAE:', mean_absolute_error(y_test, y_prediction))
coefs = pd.DataFrame(np.transpose(lr.coef_), columns = ['Coefficients'], index = x.columns)
print(coefs)
lrPlot = plt.figure()
coefs.plot.barh(figsize = (9, 7))
plt.title('Linear Regression Coefficient Bar Plot')
plt.axvline(x = 0, color = '.5')
plt.xlabel('Raw Coefficient Values')
plt.subplots_adjust(left = 0.3)

scatterPlot = plt.figure()
plt.title('Bond Age Fraction vs Minutes Elapsed')
plt.scatter(bondAgeFraction, minutesElapsed, s = 0.4, color = 'blue')
plt.yscale('log')
a, b = np.polyfit(bondAgeFraction, y, 1)
plt.plot(bondAgeFraction, a * bondAgeFraction + b, color = 'red', linestyle ='--', linewidth = 0.8)
plt.axhline(y = 2.8, color = 'green',  linewidth = 1.5)
plt.xlabel('Bond Age Fraction')
plt.ylabel('Minutes Elapsed (log scale)')
a = str(a).replace('[','')
b = str(b).replace('[','')
lobf = ('Line of best fit: ' + str(a).replace(']', '') + 'x + ' + str(b).replace(']', ''))
plt.text(0.95, 700, lobf, fontsize = 10)
print('Line of best fit: ' + str(a).replace(']', '') + 'x + ' + str(b).replace(']', ''))


#plt.show()


y = binary
over_x_train, over_x_test, over_y_train, over_y_test = train_test_split(x, y.ravel(), test_size = 0.1, random_state = 42)

#print('Under fit data:')
#y_count = over_y_train.tolist()
#print('0:', y_count.count(0))
#print('1:', y_count.count(1))
#print('Resampled:')

undersample = RandomUnderSampler(sampling_strategy = 0.8)
x_train, y_train = undersample.fit_resample(over_x_train, over_y_train)
x_test, y_test = undersample.fit_resample(over_x_test, over_y_test)

#print('Better fit data:')
#y_count = y_train.tolist()
#print('0:', y_count.count(0))
#print('1:', y_count.count(1))

trees = 1000
rfc = RandomForestClassifier(n_estimators = trees, criterion = 'gini', max_depth = 24, min_samples_leaf = 60, max_features = 0.5, n_jobs = -1, random_state = 42)
rfc.fit(x_train, y_train)
rfcPrediction = rfc.predict(x_test)
#print('R2:', r2_score(y_test, rfcPrediction))
#print('Random Forest Regressor MSE:',  mean_squared_error(y_test, rfcPrediction))
#print('Random Forest Regressor MAE:', mean_absolute_error(y_test, rfcPrediction))
#rfcPrediction = rfcPrediction.tolist()
#print('Late trades:', rfcPrediction.count(1))

cf_matrix = confusion_matrix(y_test, rfcPrediction)
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

print('Recall Score: %.3f' % recall_score(y_test, rfcPrediction))
print('Precision Score: %.3f' % precision_score(y_test, rfcPrediction))
print('Accuracy Score: %.3f' % accuracy_score(y_test, rfcPrediction))
print('F1 Score: %.3f' % f1_score(y_test, rfcPrediction))


estimatorAccuracy = []
for currentEstimator in range(trees):
    estimatorAccuracy.append([currentEstimator, accuracy_score(y_test, rfc.estimators_[currentEstimator].predict(x_test.values))])
    print(rfc.estimators_[currentEstimator])
estimatorAccuracy = pd.DataFrame(estimatorAccuracy, columns = ['Estimator Number','Accuracy'])
estimatorAccuracy.sort_values(inplace = True, by = 'Accuracy', ascending = False)
bestDecisionTree = rfc.estimators_[estimatorAccuracy.head(1)['Estimator Number'].values[0]]
#viz = dtreeviz(bestDecisionTree, x_test, y_test, target_name = 'Bond Class', feature_names = list(x.columns), class_names = '01', title = 'Best Decision Tree')
#viz
#viz.save('tree.svg')




v1Plot = plt.figure()
plt.title('Dealer Quintile Violin Plot')
sns.violinplot(x = 'Class', y = 'Dealer Quintile', data = cleanedValues)
v2Plot = plt.figure()
plt.title('Bonds Traded Violin Plot')
sns.violinplot(x = 'Class', y = 'Bonds Traded', data = cleanedValues)
cleanedValues.insert(19, 'Rounded Bond Age Fraction', roundedBondAgeFraction)
v3Plot = plt.figure()
plt.title('Bond Age Fraction Violin Plot')
sns.violinplot(x = 'Class', y = 'Rounded Bond Age Fraction', data = cleanedValues)
v4Plot = plt.figure()
plt.title('Bond Age Violin Plot')
sns.violinplot(x = 'Class', y = 'Bond Age', data = cleanedValues)
v5Plot = plt.figure()
plt.title('Dealer Rank Violin Plot')
sns.violinplot(x = 'Class', y = 'Dealer Rank', data = cleanedValues)
v6Plot = plt.figure()
plt.title('Time of Trade Violin Plot')
sns.violinplot(x = 'Class', y = 'Time of Trade', data = cleanedValues)

# parameters = [{'n_estimators': range (1000, 3000, 1000), 'criterion': ['gini'], 'max_depth': range(25, 40, 5), 'min_samples_leaf': range(60, 75, 5), 'max_features': [0.45, 0.5, 0.55, 0.6], 'class_weight': [None]}]
# gs = GridSearchCV(estimator = rfc, param_grid = parameters, cv = 10, n_jobs = -1)
# gs.fit(x_train, y_train)
# gsPrediction = gs.predict(x_test)
# print('------------------------------------------------------------')
# print('Recall Score: %.3f' % recall_score(y_test, gsPrediction))
# print('Precision Score: %.3f' % precision_score(y_test, gsPrediction))
# print('Accuracy Score: %.3f' % accuracy_score(y_test, gsPrediction))
# print('F1 Score: %.3f' % f1_score(y_test, gsPrediction))
# print(gs.best_params_)

print('--- %s seconds ---' % (time.time() - start_time))

plt.show()
'''
