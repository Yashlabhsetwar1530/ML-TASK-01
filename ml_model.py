from pandas import read_csv
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = read_csv("Salary_Data.csv")         #From this command one can load the data from the pandas frame
X = df["YearsExperience"].values.reshape(30,1)
y = df["Salary"]

                                          #Here we need to make it into 2-dimensional by the help of reshape function. This is the preprocessing of the data 
                                          #which is very important step 
X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    #To split the data to 2 parts that is training and testing.
mind = LinearRegression()                 #Start the machine learning model
mind.fit(X_train ,y_train)                #Train the ML model
dump(mind ,"Salary_model.pkl")              #save as pickle file
