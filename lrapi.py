from flask import Flask 
import json
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

# pre-processing
inputGraphResults = pd.DataFrame(columns = ['model', 'numFeat', 'mse', 'selectedFeat'])

dataset = pd.read_csv('https://raw.githubusercontent.com/kdhartmann/LinearModels/master/SaratogaHousesClean.csv')
y = dataset['price']
X = dataset.iloc[:, 1:9]

scaler = StandardScaler()
cv = KFold(n_splits=5, shuffle=False)
reg = LinearRegression(fit_intercept = True)

XScaleTransform = scaler.fit_transform(X)
XScaled = pd.DataFrame(XScaleTransform, columns = X.columns)


livingArea = np.array(XScaled['livingArea'])
livingArea = livingArea.reshape(-1,1)

mseList = []
splitList = []
XTrainList = []
yTrainList = []
XTestList = []
yTestList = []
split = 1

for train_index, test_index in cv.split(livingArea):
	X_train, X_test, y_train, y_test = livingArea[train_index], livingArea[test_index], y[train_index], y[test_index]
	reg.fit(X_train, y_train)
	y_pred = reg.predict(X_test)
	MSE = metrics.mean_squared_error(y_test, y_pred)
	mseList.append(MSE)

	X_trainCorrect = []
	for elem in X_train:
		X_trainCorrect.append(elem[0])
	X_testCorrect = []
	for elem in X_test:
		X_testCorrect.append(elem[0])

	XTrainList.append(X_trainCorrect)
	yTrainList.append(np.array(y_train))
	XTestList.append(X_testCorrect)
	yTestList.append(np.array(y_test))
	splitList.append(split)
	split +=1


# function to get dataframe of regression results 
def getResultsDF(model, features):
    df = pd.DataFrame()
    names = ['intercept']
    coefs = [model.intercept_]
    for elem in features:
        names.append(elem)
    for elem in model.coef_:
        coefs.append(elem)
    df['feature'] = names
    df['coef'] = coefs
    return df.reindex(df.coef.abs().sort_values(ascending = False).index)

app = Flask(__name__)

@app.route('/roomsUnscaled')
def roomsUnscaled():
	rooms = X[['rooms']]
	rooms_json = rooms.to_json(orient='records')
	return rooms_json

@app.route('/roomsScaled')
def roomsScaled():
	roomsScaled = XScaled[['rooms']]
	roomsScaled_json = roomsScaled.to_json(orient='records')
	return roomsScaled_json

@app.route('/trainTestSplitMSE')
def trainTestSplitMSE():
	livingArea = XScaled[['livingArea']]
	X_train, X_test, y_train, y_test = train_test_split(livingArea, y, test_size=.2, random_state=1)
	reg.fit(X_train, y_train)
	y_pred = reg.predict(X_test)
	trainTestMSE = metrics.mean_squared_error(y_test, y_pred)
	trainTestMSE_dict = {"trainTestMSE": trainTestMSE}
	trainTestMSE_json = json.dumps(trainTestMSE_dict)
	return trainTestMSE_json

@app.route('/kfoldmse')
def kfoldmse():
	# livingArea = np.array(XScaled['livingArea'])
	# livingArea = livingArea.reshape(-1,1)

	# mseList = []
	# splitList = []
	# XTrainList = []
	# yTrainList = []
	# XTestList = []
	# yTestList = []
	# split = 1

	# for train_index, test_index in cv.split(livingArea):
	# 	X_train, X_test, y_train, y_test = livingArea[train_index], livingArea[test_index], y[train_index], y[test_index]
	# 	reg.fit(X_train, y_train)
	# 	y_pred = reg.predict(X_test)
	# 	MSE = metrics.mean_squared_error(y_test, y_pred)
	# 	mseList.append(MSE)

	# 	X_trainCorrect = []
	# 	for elem in X_train:
	# 		X_trainCorrect.append(elem[0])
	# 	X_testCorrect = []
	# 	for elem in X_test:
	# 		X_testCorrect.append(elem[0])

	# 	XTrainList.append(X_trainCorrect)
	# 	yTrainList.append(np.array(y_train))
	# 	XTestList.append(X_testCorrect)
	# 	yTestList.append(np.array(y_test))
	# 	splitList.append(split)
	# 	split +=1
    
	results = pd.DataFrame()
	results['fold'] = splitList
	results['mse'] = mseList
	results_json = results.to_json(orient='records')
	return results_json

@app.route('/kfoldTrain/<fold>')
def kfoldTrain(fold):
	fold = int(fold)
	trainDF = pd.DataFrame()
	trainDF['XTrain'] = XTrainList[fold-1]
	trainDF['yTrain'] = yTrainList[fold-1]
	train_json = trainDF.to_json(orient='records')
	return train_json

@app.route('/kfoldTest/<fold>')
def kfoldTest(fold):
	fold = int(fold)
	testDF = pd.DataFrame()
	testDF['XTest'] = XTestList[fold-1]
	testDF['yTest'] = yTestList[fold-1]
	test_json = testDF.to_json(orient='records')
	return test_json


@app.route('/inputGraphs/<selectedFeats>')
def inputGraphs(selectedFeats):
	# selectedFeats = selectedFeats.split(',')
	# numFeats = len(selectedFeats)
	# selectedFeatsDF = pd.DataFrame()
	# for elem in selectedFeats:
	# 	selectedFeatsDF[elem] = XScaled[elem]
	# MSE = (cross_val_score(reg, selectedFeatsDF, y, cv=cv, scoring='neg_mean_squared_error'))*-1
	# results = pd.DataFrame()
	# results['numFeat'] = [numFeats]
	# results['mse'] = [np.mean(MSE)]
	# results['selectedFeats'] = [sorted(selectedFeats)]
	# results_json = results.to_json(orient='records')
	# return results_json
	selectedFeats = selectedFeats.split(",")
	numFeat = len(selectedFeats)
	selectedFeatDF = pd.DataFrame()
	for elem in selectedFeats:
		selectedFeatDF[elem] = XScaled[elem]
	MSE = (cross_val_score(reg, selectedFeatDF, y, cv=cv, scoring='neg_mean_squared_error'))*-1
	toConcatDF = pd.DataFrame()
	toConcatDF['numFeat'] = [numFeat]
	toConcatDF['mse'] = [np.mean(MSE)]
	toConcatDF['selectedFeat'] = [selectedFeats]
	global inputGraphResults
	toConcatDF['model'] = (inputGraphResults['selectedFeat'].shape[0] + 1)
	inputGraphResults = pd.concat([inputGraphResults, toConcatDF], sort = True)
	inputGraphResults_json = inputGraphResults.to_json(orient='records')
	return inputGraphResults_json

@app.route('/lowestMSEByFeatureCount')
def lowestMSEByFeatureCount():
	numFeat = 1
	numFeatList = []
	mseList = []
	features = []
	while numFeat <= 8:
		KBest = SelectKBest(f_regression, k=numFeat)
		Kfit = KBest.fit_transform(XScaled, y)
		column_names = np.array(XScaled.columns[KBest.get_support()])
		MSE = (cross_val_score(reg, Kfit, y, cv=cv, scoring='neg_mean_squared_error'))*-1
		numFeatList.append(numFeat)
		mseList.append(np.mean(MSE))
		features.append(column_names)
		numFeat+=1
	df = pd.DataFrame()
	df['numFeat'] = numFeatList
	df['mse'] = mseList
	df['features'] = features
	df_json = df.to_json(orient='records')
	return df_json

@app.route('/linearResults')
def linearResults():
	reg.fit(XScaled, y)
	results = getResultsDF(reg, np.array(XScaled.columns))
	results_json = results.to_json(orient='records')
	return results_json

@app.route('/featureSelectionResults/<numFeat>')
def featureSelectionResults(numFeat):
	numFeat = int(numFeat)
	KBest = SelectKBest(f_regression, k=numFeat)
	Kfit = KBest.fit_transform(XScaled, y)
	column_names = np.array(XScaled.columns[KBest.get_support()])
	reg.fit(Kfit,y)
	results = getResultsDF(reg, column_names)
	results_json = results.to_json(orient='records')
	return results_json

@app.route('/lassoResults/<_lambda>')
def lassoResults(_lambda):
	_lambda = float(_lambda)
	lasso = Lasso(alpha=_lambda)
	lasso.fit(XScaled, y)
	results = getResultsDF(lasso, np.array(XScaled.columns))
	results_json = results.to_json(orient='records')
	return results_json

@app.route('/ridgeResults/<_lambda>')
def ridgeResults(_lambda):
	_lambda = float(_lambda)
	ridge = Ridge(alpha=_lambda)
	ridge.fit(XScaled, y)
	results = getResultsDF(ridge, np.array(XScaled.columns))
	results_json = results.to_json(orient='records')
	return results_json

if __name__ == '__main__':
    app.run(debug=False)