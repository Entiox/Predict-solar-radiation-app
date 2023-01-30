from pandas import read_csv
from pickle import dump
from data_modification import formatTime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from column_names import unix_time, data_, radiation
from os import path

dataset_path = path.join("..", "resources", "SolarPrediction.csv")

def prepareDataForTraining():
    data = read_csv(dataset_path)
    data.drop([unix_time, data_], inplace=True, axis=1)
    data = formatTime(data)

    X = data.drop(radiation, axis=1)
    y = data[radiation]

    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, shuffle=True)

    scaler = MinMaxScaler()
    global X_train_normalized
    X_train_normalized = scaler.fit_transform(X_train)
    global X_test_normalized 
    X_test_normalized = scaler.transform(X_test)
    dump(scaler, file=open(path.join("..", "resources", "scaler.pkl"), "wb"))

def trainAndSaveModel():
    prepareDataForTraining()
    rf = ExtraTreesRegressor(n_estimators=128, max_depth=32, min_samples_split=1, min_samples_leaf=1)
    dump(rf.fit(X_train_normalized, y_train), file=open(path.join("..", "resources", "solar_radiation_prediction_model.pkl"), "wb"))

def printStatistics(algorithmName, y_pred, y_test):
    mae = mean_absolute_error(y_pred, y_test)
    mse = mean_squared_error(y_pred, y_test)
    rmse = mean_squared_error(y_pred, y_test, squared=False)
    r2 = r2_score(y_pred, y_test)
    print(f"{algorithmName}:")
    print(f"Mean Absolute Error: {mae}\nMean Squared Error: {mse}\nRoot Mean Squared Error: {rmse}\nCoefficient of Determination: {r2}\n")

def trainAndTestModels():
    rf = RandomForestRegressor(n_estimators=64, max_depth=64, min_samples_split=1, min_samples_leaf=1)
    printStatistics(algorithmName="Random Forest Regression", 
        y_pred=rf.fit(X_train_normalized, y_train).predict(X_test_normalized), y_test=y_test)

    et = ExtraTreesRegressor(n_estimators=128, max_depth=32, min_samples_split=1, min_samples_leaf=1)
    printStatistics(algorithmName="Extra Trees Regression",
        y_pred=et.fit(X_train_normalized, y_train).predict(X_test_normalized), y_test=y_test)

    gb = GradientBoostingRegressor(learning_rate=0.1, n_estimators=64, max_depth=128, min_samples_split=50, min_samples_leaf=1)
    printStatistics(algorithmName="Gradient Boosting Regression",
        y_pred=gb.fit(X_train_normalized, y_train).predict(X_test_normalized), y_test=y_test)

    hb = HistGradientBoostingRegressor(learning_rate=0.1, max_iter=5000, max_leaf_nodes=100, min_samples_leaf=100)
    printStatistics(algorithmName="Hist Gradient Boosting Regression",
        y_pred=hb.fit(X_train_normalized, y_train).predict(X_test_normalized), y_test=y_test)

    nn = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100), max_iter=500, learning_rate="constant")
    printStatistics(algorithmName="Neural Network Regression",
        y_pred=nn.fit(X_train_normalized, y_train).predict(X_test_normalized), y_test=y_test)

if(__name__ == "__main__"):
    prepareDataForTraining()
    print("Results of training:\n")
    trainAndTestModels()
