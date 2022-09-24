import os
import pandas as pd
import string
import tldextract
import phonenumbers
import jaro

#path = r'C:\\Users\\mweiss\\Python Projects\\Matching Algorthm\\Curated Data' #path of folder from zip file
#outputPath = 'C:\\Users\\mweiss\\Python Projects\\Matching Algorthm' #where master excel file where be written to

path = r'/Users/martinweiss/Desktop/Python/MSRB/Curated Data/' #path of folder from zip file
outputPath = '/Users/martinweiss/Desktop/Python/MSRB/' #where master excel file where be written to

master = pd.DataFrame()
files = os.listdir(path)
files.sort()
fileSource = []

#loop through all files in folder
for file in files:
	if file.endswith('xlsx'):
		excelFile = pd.ExcelFile(f'{path}/{file}')
		sheets = excelFile.sheet_names
		#loop through all sheets
		for sheet in sheets:
			sheetData = excelFile.parse(sheet_name = sheet)
			sheetLen = sheetData.shape[0]
			master = master.append(sheetData)
			master.reset_index()
		#loop through all sheets and take the name and add to a list called fileSource
		for i in range(sheetLen):
			fileSource.append(file.replace('_dbe_curated.xlsx', '').upper())

#add state source column to master file
master.insert(16, 'State Source', fileSource, True)
#drop all columns after State Source column, column Q
master.drop(master.columns[17:], axis = 1, inplace = True)
#write the data to the master and output to outputPath directory line 13
master.to_excel(f'{outputPath}/master_dbe_directory.xlsx', index = False) #output name of master file. outputPath = location of master excel file, change on line 13
print('Sheet Created. \nPath:' + outputPath + '\nSheet Name: master_dbe_directory.xlsx')

#masterPath = r'C:\\Users\\mweiss\\Python Projects\\Matching Algorthm\\master_dbe_directory.xlsx'
#msrbPath = r'C:\\Users\\mweiss\\Python Projects\\Matching Algorthm\\MSRB_registrants_20220519.xlsx'

masterPath = r'/Users/martinweiss/Desktop/Python/MSRB/master_dbe_directory.xlsx'
msrbPath = r'/Users/martinweiss/Desktop/Python/MSRB/MSRB_registrants_20220519.xlsx'
vendorPath = r'/Users/martinweiss/Desktop/Python/MSRB/MSRB Vendor Universe.xlsx'

masterFile = pd.read_excel(masterPath)
msrbFile = pd.read_excel(msrbPath, sheet_name = 'Registration Report')
vendorFile = pd.read_excel(vendorPath, sheet_name = 'Vendor Universe')

allFirmNames = masterFile['Firm Name'] #A
allContactNames = masterFile['Contact Name'] #B
allPhoneNumbers = masterFile['Phone'] #C
allAddresses = masterFile['Address'] #D
allCities = masterFile['City'] #D
allStates = masterFile['State'] #F
allCounties = masterFile['County'] #F
allZipCodes = masterFile['Zip'] #H
allEmails = masterFile['Email'] #I
allWebsites = masterFile['Website'] #J
allCertifications = masterFile['Certification'] #K
allAgencyCertifications = masterFile['Agency Certification'] #L
allBusinessTypes = masterFile['Business Type'] #M
allEthnicities = masterFile['Ethnicity'] #N
allGenders = masterFile['Gender'] #O
allNAICSCodes = masterFile['NAICS Codes'] #P
allStateSources = masterFile['State Source'] #Q
#allDomains #R

masterLen = masterFile.shape[0] #length of master file

allFirmNames = allFirmNames.tolist() #A
allContactNames = allContactNames.tolist() #B
allPhoneNumbers = allPhoneNumbers.tolist() #C
allAddresses = allAddresses.tolist() #D
allCities = allCities.tolist() #D
allStates = allStates.tolist() #F
allCounties = allCounties.tolist() #F
allZipCodes = allZipCodes.tolist() #H
allEmails = allEmails.tolist() #I
allWebsites = allWebsites.tolist() #J
allCertifications = allCertifications.tolist() #K
allAgencyCertifications = allAgencyCertifications.tolist() #L
allBusinessTypes = allBusinessTypes.tolist() #M
allEthnicities = allEthnicities.tolist() #N
allGenders = allGenders.tolist() #O
allNAICSCodes = allNAICSCodes.tolist() #P
allStateSources = allStateSources.tolist() #Q
#allDomains #R

msrbMatchedFirmNames = [] #A
msrbMatchedContactNames = [] #B
msrbMatchedPhoneNumbers = [] #C
msrbMatchedAddresses = [] #D
msrbMatchedCities = [] #D
msrbMatchedStates = [] #F
msrbMatchedCounties = [] #F
msrbMatchedZipCodes = [] #H
msrbMatchedEmails = [] #I
msrbMatchedWebsites = [] #J
msrbMatchedCertifications = [] #K
msrbMatchedAgencyCertifications = [] #L
msrbMatchedBusinessTypes = [] #M
msrbMatchedEthnicities = [] #N
msrbMatchedGenders = [] #O
msrbMatchedNAICSCodes = [] #P
msrbMatchedStateSources = [] #Q
msrbMatchedDomains = [] #R

#columns from msrb registrants file
msrbFirmNames = msrbFile['Organization Name']
msrbContactNames = msrbFile['Contact Name']
msrbEmails = msrbFile['Primary Reg Contact Email']
msrbWebsites = msrbFile['Organization URL']
msrbPhoneNumbers = msrbFile['Primary Reg Contact Phone Number']
msrbID = msrbFile['MSRB ID']
msrbType = msrbFile['Registration Type']
msrbLen = msrbFile.shape[0] #length of msrb registrants file

#setting pandas data frames to lists
msrbFirmNames = msrbFirmNames.tolist()
msrbContactNames = msrbContactNames.tolist()
msrbEmails = msrbEmails.tolist()
msrbWebsites = msrbWebsites.tolist()
msrbPhoneNumbers = msrbPhoneNumbers.tolist()
msrbID = msrbID.tolist()
msrbType = msrbType.tolist()

#columns from vendor file
vendorNames = vendorFile['Vendor']
vendorLen = vendorFile.shape[0] #length of vendor file

#setting pandas data frames to lists
vendorNames = vendorNames.tolist()

#creating empty lists to add data to later on
allCleanedEmailDomains = []
allCleanedWebsiteDomains = []
allMatchedDomains = []
allCleanedPhoneNumbers = []

msrbCleanedEmailDomains = []
msrbCleanedWebsiteDomains = []
msrbDomainHits = []
msrbCleanedPhoneNumbers = []
msrbIDList = []
msrbTypeList = []

naicsMatchedFirms = []
cleanedMatchedNaicsCodes = []
cleanedNaicsCodes = []
dealerList = []
maList = []
dealerMaList = []
dealerNaicsCodes = []
maNaicsCodes = []
dealerMaNaicsCodes = []
cleanedDealerNaicsCodes = []
cleanedMaNaicsCodes = []
cleanedDealerMaNaicsCodes = []
uniqueDealerNaicsCodes = []
uniqueMaNaicsCodes = []
uniqueDealerMaNaicsCodes = []

matchedByFirm = []
matchedByContact = []
matchedByDomain = []
matchedByPhone= []

matchedByFirmBool = []
matchedByContactBool = []
matchedByDomainBool = []
matchedByPhoneBool = []

totalMatchedFirms = []
contactList = []
domainList = []
phoneList = []
matchCountList = []

#initializing counts for summation
firmMatchedCount = 0
contactMatchedCount = 0
domainMatchedCount = 0
phoneMatchedCount = 0
totalDealerCount = 0
totalMaCount = 0
totalDealerMaCount = 0
matchedDealerCount = 0
matchedMaCount = 0
matchedDealerMaCount = 0
totalMatchedCount = 0
matchedNaicsCount = 0
vendorCount = 0
totalNaicsCodes = 6

#master firm names
for i in range(masterLen):
	if(type(allFirmNames[i]) == str):
		for character in string.punctuation:
			allFirmNames[i] = allFirmNames[i].replace(character, '')
			allFirmNames[i] = allFirmNames[i].strip()
			allFirmNames[i] = allFirmNames[i].lower()

#msrb firm names
for i in range(msrbLen):
	if(type(msrbFirmNames[i]) == str):
		for character in string.punctuation:
			msrbFirmNames[i] = msrbFirmNames[i].replace(character, '')
			msrbFirmNames[i] = msrbFirmNames[i].strip()
			msrbFirmNames[i] = msrbFirmNames[i].lower()
			
#vendor firm names
for i in range(vendorLen):
	if(type(vendorNames[i]) == str):
		for character in string.punctuation:
			vendorNames[i] = vendorNames[i].replace(character, '')
			vendorNames[i] = vendorNames[i].strip()
			vendorNames[i] = vendorNames[i].lower()

for j in range(masterLen):
	for i in range(vendorLen):
		#if statment to see if one of the four criteria match: firm name, contact name, domain name, or phone number
		if((vendorNames[i] == allFirmNames[j])):
			
			vendorCount = vendorCount + 1
			
print('Matched Vendor Count:', vendorCount)
			
#master contact names
for i in range(masterLen):
	if(type(allContactNames[i]) == str):
		allContactNames[i] = allContactNames[i].strip()
		allContactNames[i] = allContactNames[i].lower()

#msrb contact names
for i in range(msrbLen):
	if(type(msrbContactNames[i]) == str):
		msrbContactNames[i] = msrbContactNames[i].strip()
		msrbContactNames[i] = msrbContactNames[i].lower()

#create list of cleaned email domains from the master file
for i in range(masterLen):
	if((type(allEmails[i]) == str) and ('@' in allEmails[i])):
		allEmails[i] = allEmails[i].lower()
		allCleanedEmailDomains.append(allEmails[i][allEmails[i].index('@') + 1 : ])
	else:
		allCleanedEmailDomains.append('No Email')

#create list of cleaned website domains from master file
for i in range(masterLen):
	if(type(allWebsites[i]) == str):
		allWebsites[i] = allWebsites[i].lower()
		masterWebsiteDomainName = tldextract.extract(allWebsites[i])
		allCleanedWebsiteDomains.append(masterWebsiteDomainName.domain)
	else:
		allCleanedWebsiteDomains.append('No Domain')

#replace .com, .net and common domains in cleaned email domains
for i in range(len(allCleanedEmailDomains)):
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('.com', ''))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('.net', ''))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('.org', ''))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('gmail', 'No Domain'))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('yahoo', 'No Domain'))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('att', 'No Domain'))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('aol', 'No Domain'))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('msn', 'No Domain'))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('verizon', 'No Domain'))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('comcast', 'No Domain'))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('usbank', 'No Domain'))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('hotmail', 'No Domain'))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('bellsouth', 'No Domain'))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('lpl', 'No Domain'))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('irr', 'No Domain'))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('bellsw', 'No Domain'))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('outlook', 'No Domain'))
	allCleanedEmailDomains[i] = (allCleanedEmailDomains[i].replace('sbcglobal', 'No Domain'))

#create list of matching email and website domains. If there is a match, copy the website domain into the list
for i in range(len(allCleanedWebsiteDomains)):
	if (((allCleanedWebsiteDomains[i] in allCleanedEmailDomains) or (allCleanedEmailDomains[i] in allCleanedEmailDomains)) and (allCleanedWebsiteDomains[i] != 'none')):
		allMatchedDomains.append(allCleanedWebsiteDomains[i])
	else:
		allMatchedDomains.append(allCleanedEmailDomains[i])
	if (allCleanedWebsiteDomains[i] == 'No Domain'):
		allMatchedDomains[i] = allCleanedEmailDomains[i]

#create list of cleaned email domains from the msrb file
for i in range(msrbLen):
	if((type(msrbEmails[i]) == str) and ('@' in msrbEmails[i])):
		msrbEmails[i] = msrbEmails[i].lower()
		msrbCleanedEmailDomains.append(msrbEmails[i][msrbEmails[i].index('@') + 1 : ])
	else:
		msrbCleanedEmailDomains.append('No Email')

#create list of cleaned website domains from msrb file
for i in range(msrbLen):
	if(type(msrbWebsites[i]) == str):
		msrbWebsites[i] = msrbWebsites[i].lower()
		msrbWebsiteDomainName = tldextract.extract(msrbWebsites[i])
		msrbCleanedWebsiteDomains.append(msrbWebsiteDomainName.domain)
	else:
		msrbCleanedWebsiteDomains.append('No Domain')

#replace .com, .net and common domains in cleaned email domains
for i in range(len(msrbCleanedEmailDomains)):
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('.com', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('.net', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('.org', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('gmail', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('yahoo', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('att', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('aol', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('msn', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('verizon', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('comcast', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('usbank', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('hotmail', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('bellsouth', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('lpl', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('irr', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('bellsw', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('outlook', 'No Domain'))
	msrbCleanedEmailDomains[i] = (msrbCleanedEmailDomains[i].replace('sbcglobal', 'No Domain'))

#create list of matching email and website domains. If there is a match, copy the website domain into the list
for i in range(len(msrbCleanedWebsiteDomains)):
	if (((msrbCleanedWebsiteDomains[i] in msrbCleanedEmailDomains) or (msrbCleanedEmailDomains[i] in msrbCleanedEmailDomains)) and (msrbCleanedWebsiteDomains[i] != 'none')):
		msrbDomainHits.append(msrbCleanedWebsiteDomains[i])
	else:
		msrbDomainHits.append(msrbCleanedEmailDomains[i])
	if ((msrbCleanedWebsiteDomains[i] == 'No Domain') or (msrbCleanedWebsiteDomains[i] == 'none')):
		msrbDomainHits[i] = msrbCleanedEmailDomains[i].replace('No Domain', '')
	if(msrbDomainHits[i] == ''):
		msrbDomainHits[i] = 'NA'

print('Finished Domain Matching From Master File')

#create a list of phone numbers from the master file
for i in range(masterLen):
	if ((type(allPhoneNumbers[i]) == str) and (len(allPhoneNumbers[i]) > 9)):
		allCleanedPhoneNumbers.append(phonenumbers.parse(allPhoneNumbers[i], 'US').national_number)
	else:
		allCleanedPhoneNumbers.append('No Phone')

#create a list of phone numbers from the msrb file
for i in range(msrbLen):
	if ((type(msrbPhoneNumbers[i]) == str)):
		msrbCleanedPhoneNumbers.append(phonenumbers.parse(msrbPhoneNumbers[i], 'US').national_number)
	else:
		msrbCleanedPhoneNumbers.append('No Phone')

#core logic of the matching algorthm
#nested for loop will get every entry of a matced firm across all 46 states present in the master files
for j in range(masterLen):
	for i in range(msrbLen):
		#if statment to see if one of the four criteria match: firm name, contact name, domain name, or phone number
		if((msrbFirmNames[i] == allFirmNames[j]) or (msrbContactNames[i] == allContactNames[j]) or (msrbDomainHits[i] == allMatchedDomains[j]) or (msrbCleanedPhoneNumbers[i] == allCleanedPhoneNumbers[j])):
			matchCount = 0
			msrbType[i] = msrbType[i].strip()
			#checking if the firm name from the msrb file is present in the master file
			if(msrbFirmNames[i] in allFirmNames):
				matchedByFirm.append(msrbFirmNames[i])
				matchedByFirmBool.append('Y')
				firmMatchedCount = firmMatchedCount + 1
				matchCount = matchCount + 1
			else:
				matchedByFirm.append('')
				matchedByFirmBool.append('N')

			#checking if the contact name from the msrb file is present in the master file
			if(msrbContactNames[i] in allContactNames):
				matchedByContact.append(msrbFirmNames[i])
				matchedByContactBool.append('Y')
				contactList.append(msrbContactNames[i])
				contactMatchedCount = contactMatchedCount + 1
				matchCount = matchCount + 1
			else:
				matchedByContact.append('')
				matchedByContactBool.append('N')
				contactList.append('')

			#checking if the domain name from the msrb file is present in the master file
			if(msrbDomainHits[i] in allMatchedDomains):
				matchedByDomain.append(msrbFirmNames[i])
				matchedByDomainBool.append('Y')
				domainList.append(msrbDomainHits[i])
				domainMatchedCount = domainMatchedCount + 1
				matchCount = matchCount + 1
			else:
				matchedByDomain.append('')
				matchedByDomainBool.append('N')
				domainList.append('')

			#checking if the phone number from the msrb file is present in the master file
			if(msrbCleanedPhoneNumbers[i] in allCleanedPhoneNumbers):
				matchedByPhone.append(msrbFirmNames[i])
				matchedByPhoneBool.append('Y')
				phoneList.append(msrbCleanedPhoneNumbers[i])
				phoneMatchedCount = phoneMatchedCount + 1
				matchCount = matchCount + 1
			else:
				matchedByPhone.append('')
				matchedByPhoneBool.append('N')
				phoneList.append('')

			#appending lists with matched information from the master file and appending total count with total number of matches
			matchCountList.append(matchCount)
			totalMatchedCount = totalMatchedCount + 1
			msrbIDList.append(msrbID[i])
			msrbTypeList.append(msrbType[i])
			totalMatchedFirms.append(msrbFirmNames[i])
			msrbMatchedFirmNames.append(allFirmNames[j]) #A
			msrbMatchedContactNames.append(allContactNames[j]) #B
			msrbMatchedPhoneNumbers.append(allPhoneNumbers[j]) #C
			msrbMatchedAddresses.append(allAddresses[j]) #D
			msrbMatchedCities.append(allCities[j]) #D
			msrbMatchedStates.append(allStates[j]) #F
			msrbMatchedCounties.append(allCounties[j]) #F
			msrbMatchedZipCodes.append(allZipCodes[j]) #H
			msrbMatchedEmails.append(allEmails[j]) #I
			msrbMatchedWebsites.append(allWebsites[j]) #J
			msrbMatchedCertifications.append(allCertifications[j]) #K
			msrbMatchedAgencyCertifications.append(allAgencyCertifications[j]) #L
			msrbMatchedBusinessTypes.append(allBusinessTypes[j]) #M
			msrbMatchedEthnicities.append(allEthnicities[j]) #N
			msrbMatchedGenders.append(allGenders[j]) #O
			msrbMatchedNAICSCodes.append(allNAICSCodes[j]) #P
			msrbMatchedStateSources.append(allStateSources[j]) #Q
			msrbMatchedDomains.append(allMatchedDomains[j]) #R

#rewrite the master excel sheet with updated domain column
masterFile.insert(17, 'Domain', allMatchedDomains)
#export the updated master file
masterFile.to_excel(f'{outputPath}/master_dbe_directory.xlsx', index = False) #output name of master file. outputPath = location of master excel file, change on line 13

print('Updated Master Sheet with Domain Names')

#new pandas data frame that stores all matched information
matchedFile = pd.DataFrame({'ALL MATCHED FIRMS': totalMatchedFirms})
matchedFile.insert(1, 'Firm Name', msrbMatchedFirmNames)
matchedFile.insert(2, 'Contact Name', msrbMatchedContactNames)
matchedFile.insert(3, 'Phone', msrbMatchedPhoneNumbers)
matchedFile.insert(4, 'Address', msrbMatchedAddresses)
matchedFile.insert(5, 'City', msrbMatchedCities)
matchedFile.insert(6, 'State', msrbMatchedStates)
matchedFile.insert(7, 'County', msrbMatchedCounties)
matchedFile.insert(8, 'Zip', msrbMatchedZipCodes)
matchedFile.insert(9, 'Email', msrbMatchedEmails)
matchedFile.insert(10, 'Website', msrbMatchedWebsites)
matchedFile.insert(11, 'Certification', msrbMatchedCertifications)
matchedFile.insert(12, 'Agency Certification', msrbMatchedAgencyCertifications)
matchedFile.insert(13, 'Business Type', msrbMatchedBusinessTypes)
matchedFile.insert(14, 'Ethnicity', msrbMatchedEthnicities)
matchedFile.insert(15, 'Gender', msrbMatchedGenders)
matchedFile.insert(16, 'NAICS Codes', msrbMatchedNAICSCodes)
matchedFile.insert(17, 'State Source', msrbMatchedStateSources)
matchedFile.insert(18, 'Domain', msrbMatchedDomains)
matchedFile.insert(19, 'MASTER Matched by Firm Name', matchedByFirm)
matchedFile.insert(20, 'Firm Name Match Y/N', matchedByFirmBool)
matchedFile.insert(21, 'MASTER Matched by Contact Name', matchedByContact)
matchedFile.insert(22, 'Contact Name Match Y/N', matchedByContactBool)
matchedFile.insert(23, 'MASTER Matched Contact Name', contactList)
matchedFile.insert(24, 'Matched by Domain Name', matchedByDomain)
matchedFile.insert(25, 'Domain Name Match Y/N', matchedByDomainBool)
matchedFile.insert(26, 'MASTER Domain Name', domainList)
matchedFile.insert(27, 'Matched by Phone Number', matchedByPhone)
matchedFile.insert(28, 'Phone Match Y/N', matchedByPhoneBool)
matchedFile.insert(29, 'MASTER Phone Number', phoneList)
matchedFile.insert(30, 'MSRB ID', msrbIDList)
matchedFile.insert(31, 'MSRB Registration Type', msrbTypeList)
matchedFile.insert(32, 'Match Count', matchCountList)

#second part of core matching algorthm
#this will loop through the entire matched list and remove matches that only match one criteria. Matches with 2, 3, or 4 matched criteria will be kept.
for i in range(totalMatchedCount):
	if(matchCountList[i] == 1):
		#drop rows that only have one matched criteria
		matchedFile.drop([i], axis = 0, inplace = True)
	else:
		naicsMatchedFirms.append(totalMatchedFirms[i])
		cleanedMatchedNaicsCodes.append(msrbMatchedNAICSCodes[i])
		#count of dealers, municipal advisors, and dealer + municipal advisors
		if (msrbTypeList[i] == 'DEALER'):
			matchedDealerCount = matchedDealerCount + 1
			dealerList.append('DEALER')
			dealerNaicsCodes.append(msrbMatchedNAICSCodes[i])
		else:
			dealerList.append('')
			dealerNaicsCodes.append('')
		if (msrbTypeList[i] == 'MA'):
			matchedMaCount = matchedMaCount + 1
			maList.append('MA')
			maNaicsCodes.append(msrbMatchedNAICSCodes[i])
		else:
			maList.append('')
			maNaicsCodes.append('')
		if (msrbTypeList[i] == 'DEALER and MA'):
			matchedDealerMaCount = matchedDealerMaCount + 1
			dealerMaList.append('DEALER and MA')
			dealerMaNaicsCodes.append(msrbMatchedNAICSCodes[i])
		else:
			dealerMaList.append('')
			dealerMaNaicsCodes.append('')

for i in range(msrbLen):
	if (msrbType[i] == 'DEALER'):
		totalDealerCount = totalDealerCount + 1
	if (msrbType[i] == 'MA'):
		totalMaCount = totalMaCount + 1
	if (msrbType[i] == 'DEALER and MA'):
		totalDealerMaCount = totalDealerMaCount + 1

#print revlent counts for later use in a report/talking points
print('Firm Matched Count:', firmMatchedCount)
print('Contact Matched Count:', contactMatchedCount)
print('Domain Matched Count:', domainMatchedCount)
print('Phone Matched Count:', phoneMatchedCount)
print('Total Matched Count:', totalMatchedCount)
print('Matched Dealer Count:', matchedDealerCount)
print('Total Dealer Count:', totalDealerCount)
print('Matched MA Count:', matchedMaCount)
print('Total MA Count:', totalMaCount)
print('Matched Dealer and MA Count:', matchedDealerMaCount)
print('Total Dealer and MA Count:', totalDealerMaCount)

#export the matched file to excel
matchedFile.to_excel(f'{outputPath}/registrant_matched_dbe_directory.xlsx', index = False)

for i in range(len(naicsMatchedFirms)):
	cleanedMatchedNaicsCodes[i] = str(cleanedMatchedNaicsCodes[i])
	cleanedNaicsCodes.append([int(j) for j in cleanedMatchedNaicsCodes[i].split() if j.isdigit()])
	
	dealerNaicsCodes[i] = str(dealerNaicsCodes[i])
	dealerNaicsCodes[i] = ([int(j) for j in dealerNaicsCodes[i].split() if j.isdigit()])
	
	maNaicsCodes[i] = str(maNaicsCodes[i])
	maNaicsCodes[i] = ([int(j) for j in maNaicsCodes[i].split() if j.isdigit()])
	
	dealerMaNaicsCodes[i] = str(dealerMaNaicsCodes[i])
	dealerMaNaicsCodes[i] = ([int(j) for j in dealerMaNaicsCodes[i].split() if j.isdigit()])
	
for i in range(len(cleanedNaicsCodes)):
	cleanedNaicsCodes[i] = str(cleanedNaicsCodes[i])
	cleanedNaicsCodes[i] = cleanedNaicsCodes[i].replace('[', '')
	cleanedNaicsCodes[i] = cleanedNaicsCodes[i].replace(']', '')
	cleanedNaicsCodes[i] = cleanedNaicsCodes[i].replace('2022', '')
	cleanedNaicsCodes[i] = cleanedNaicsCodes[i].replace(',', '')
	dealerNaicsCodes[i] = str(dealerNaicsCodes[i])
	dealerNaicsCodes[i] = dealerNaicsCodes[i].replace('[', '')
	dealerNaicsCodes[i] = dealerNaicsCodes[i].replace(']', '')
	dealerNaicsCodes[i] = dealerNaicsCodes[i].replace('2022', '')
	dealerNaicsCodes[i] = dealerNaicsCodes[i].split(',')
	maNaicsCodes[i] = str(maNaicsCodes[i])
	maNaicsCodes[i] = maNaicsCodes[i].replace('[', '')
	maNaicsCodes[i] = maNaicsCodes[i].replace(']', '')
	maNaicsCodes[i] = maNaicsCodes[i].replace('2022', '')
	maNaicsCodes[i] = maNaicsCodes[i].split(',')
	dealerMaNaicsCodes[i] = str(dealerMaNaicsCodes[i])
	dealerMaNaicsCodes[i] = dealerMaNaicsCodes[i].replace('[', '')
	dealerMaNaicsCodes[i] = dealerMaNaicsCodes[i].replace(']', '')
	dealerMaNaicsCodes[i] = dealerMaNaicsCodes[i].replace('2022', '')
	dealerMaNaicsCodes[i] = dealerMaNaicsCodes[i].split(',')
	
uniqueNaicsCodes = ['523930', '541211', '541611', '524298', '52393', '541612', '523120', '94638', '94656', '94675', '84121800', '94630', '93151605', '42512', '92113', '541330', '541370', '541213', '561110', '523940', '523110', '523999', '5239', '531390', '541320', '541614', '541620', '91849', '541618', '541990', '20991', '91806', '96258', '91890', '95816', '95877', '12', '541350', '523920', '523991', '541990', '522291', '921130', '523150', '94625', '94648', '84111700', '93151600', '541611', '523920']

for j in range(masterLen):
	allNAICSCodes[j] = str(allNAICSCodes[j])
	if(len(allNAICSCodes[j]) > 3):
		totalNaicsCodes = totalNaicsCodes + 1
	for i in range(len(uniqueNaicsCodes)):
		if(uniqueNaicsCodes[i] == allNAICSCodes[j]):
			matchedNaicsCount = matchedNaicsCount + 1
			
percentage = (matchedNaicsCount/totalNaicsCodes)*100
print('Matched NAICS codes from master file using NAICS codes from matched MSRB registrants:', matchedNaicsCount)
print('Percentage of matched NAICS codes from master file using NAICS codes from matched MSRB registrants:', percentage)

for i in range(len(cleanedNaicsCodes)):	
	dealerNaicsCodes[i] = str(dealerNaicsCodes[i])
	dealerNaicsCodes[i] = dealerNaicsCodes[i].split(',')
	cleanedDealerNaicsCodes.append(dealerNaicsCodes[i])
	
	maNaicsCodes[i] = str(maNaicsCodes[i])
	maNaicsCodes[i] = maNaicsCodes[i].split(',')
	cleanedMaNaicsCodes.append(maNaicsCodes[i])
	
	dealerMaNaicsCodes[i] = str(dealerMaNaicsCodes[i])
	dealerMaNaicsCodes[i] = dealerMaNaicsCodes[i].split(',')
	cleanedDealerMaNaicsCodes.append(dealerMaNaicsCodes[i])
	
for i in range(len(cleanedNaicsCodes)):
	cleanedDealerNaicsCodes[i] = str(cleanedDealerNaicsCodes[i])
	cleanedDealerNaicsCodes[i] = cleanedDealerNaicsCodes[i].replace('[', '')
	cleanedDealerNaicsCodes[i] = cleanedDealerNaicsCodes[i].replace(']', '')
	cleanedDealerNaicsCodes[i] = cleanedDealerNaicsCodes[i].replace('\'', '')
	cleanedDealerNaicsCodes[i] = cleanedDealerNaicsCodes[i].replace('\"', '')
	cleanedDealerNaicsCodes[i] = cleanedDealerNaicsCodes[i].replace(',', '')
	
	cleanedMaNaicsCodes[i] = str(cleanedMaNaicsCodes[i])
	cleanedMaNaicsCodes[i] = cleanedMaNaicsCodes[i].replace('[', '')
	cleanedMaNaicsCodes[i] = cleanedMaNaicsCodes[i].replace(']', '')
	cleanedMaNaicsCodes[i] = cleanedMaNaicsCodes[i].replace('\'', '')
	cleanedMaNaicsCodes[i] = cleanedMaNaicsCodes[i].replace('\"', '')
	cleanedMaNaicsCodes[i] = cleanedMaNaicsCodes[i].replace(',', '')
	
	cleanedDealerMaNaicsCodes[i] = str(cleanedDealerMaNaicsCodes[i])
	cleanedDealerMaNaicsCodes[i] = cleanedDealerMaNaicsCodes[i].replace('[', '')
	cleanedDealerMaNaicsCodes[i] = cleanedDealerMaNaicsCodes[i].replace(']', '')
	cleanedDealerMaNaicsCodes[i] = cleanedDealerMaNaicsCodes[i].replace('\'', '')
	cleanedDealerMaNaicsCodes[i] = cleanedDealerMaNaicsCodes[i].replace('\"', '')
	cleanedDealerMaNaicsCodes[i] = cleanedDealerMaNaicsCodes[i].replace(',', '')
	
naicsFile = pd.DataFrame({'ALL MATCHED FIRMS': naicsMatchedFirms})
naicsFile.insert(1, 'NAICS Codes', cleanedMatchedNaicsCodes)
naicsFile.insert(2, 'Dealer', dealerList)
naicsFile.insert(3, 'MA', maList)
naicsFile.insert(4, 'Dealer and MA', dealerMaList)
naicsFile.insert(5, 'Cleaned NAICS Codes', cleanedNaicsCodes)
naicsFile.insert(6, 'Dealer NAICS Codes', cleanedDealerNaicsCodes)
naicsFile.insert(7, 'MA NAICS Codes', cleanedMaNaicsCodes)
naicsFile.insert(8, 'Dealer and MA NAICS Codes', cleanedDealerMaNaicsCodes)

naicsFile.to_excel(f'{outputPath}/naics_matched_dbe_directory.xlsx', index = False)
	
uniqueDealerNaicsCodes = list(dict.fromkeys(cleanedDealerNaicsCodes))
uniqueMaNaicsCodes = list(dict.fromkeys(cleanedMaNaicsCodes))
uniqueDealerMaNaicsCodes = list(dict.fromkeys(cleanedDealerMaNaicsCodes))

print('Dealer NAICS codes:', uniqueDealerNaicsCodes)
print('MA NAICS codes:', uniqueMaNaicsCodes)
print('Dealer and MA NAICS codes:', uniqueDealerMaNaicsCodes)
