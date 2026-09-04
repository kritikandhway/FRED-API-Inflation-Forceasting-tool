import requests
import numpy as np
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent/".env")
datasets = {}
class data_gathering:
    def __init__(self,series):
        self.series = series
        self.data_collection()
        
    def data_collection(self):
        for name,series_id in self.series.items():
            URL = "https://api.stlouisfed.org/fred/series/observations"
            FRED_API = os.getenv("FRED_API_KEY")
            parameter = {"series_id":series_id,"api_key":FRED_API,"file_type":"json"}
            response = requests.get(URL,params=parameter)
            file = response.json()
            observation = file["observations"]
            df = pd.DataFrame(observation)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"],errors="coerce")
            df = df.set_index("date")
            datasets[name] = df   
     

class data_processing:
    def __init__(self,datasets):
            self.new_dataset = {}
            self.datasets = datasets
            self.variables = []
            self.final_list = []
            self.drop_list = []
            
            self.process()
    def process(self):
            self.transformation()
            self.set()
            self.concat()
            self.columns()
            self.drop_var()
            self.drop_lables()
            self.drop_na()
            self.csv()
    def transformation(self):
        if "Oil_price" in self.datasets:
                self.datasets["Oil_month_P"] = (self.datasets["Oil_price"]["value"].resample("MS").mean()).to_frame()
        if "Money_Supply_M2" in self.datasets:
                self.datasets["Money_Supply_M2"]["value"] = self.datasets["Money_Supply_M2"]["value"].pct_change(12)*100
        else:
            pass
    def set(self):
        for name,df in self.datasets.items():
            if name == "Oil_price":
                continue
            new_df = df.loc["2010-01-01":]
            self.new_dataset[name] = new_df
            self.variables.append(name)
            self.final_list.append(new_df)
    def concat(self):
        self.data_final = pd.concat(self.final_list,axis=1,join="inner")
    def columns(self):
        self.cols = list(self.data_final.columns)
        i = 0
        for name in self.variables:
            if name == "Oil_month_P":
                self.cols[i] = name+"_"+self.cols[i]
                i+=1
            else:
                self.cols[i]=name+"_"+self.cols[i]
                self.cols[i+1]=name+"_"+self.cols[i+1]
                self.cols[i+2]=name+"_"+self.cols[i+2]
                i+=3
            self.data_final.columns = self.cols
    def drop_var(self):
        for name,df in self.datasets.items():
            self.drop_list.append(name+"_realtime_start")
            self.drop_list.append(name+"_realtime_end")
    def drop_lables(self):
        self.data_ana = self.data_final.drop(labels=self.drop_list,axis = 1,errors="ignore")
    def drop_na(self):
        self.data_analysis = self.data_ana.dropna()
    def csv(self):
        self.data_analysis.to_csv("inflation_data10.csv",index=True)

series = {"sentiment":"UMCSENT","retail_sales":"RSAFS","fed_funds":"FEDFUNDS","core_cpi":"CPILFESL",
                       "ppi":"PPIACO","unemployment":"UNRATE","cpi":"CPIAUCSL","Money_Supply_M2":"M2SL","Oil_price":"DCOILWTICO",
                       "housing_price":"CSUSHPINSA","inflation_exp":"MICH","Energy_price":"CPIENGSL","IND_pro":"INDPRO",
                       "wage":"CES0500000003","rent":"CUSR0000SAH1"}
data = data_gathering(series)
processed_data = data_processing(datasets)

import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
data_analysis = processed_data.data_analysis
data_analysis["inflation_value"] = data_analysis["cpi_value"].pct_change()*100
data_analysis["rolling_inflation_value"] = (data_analysis["inflation_value"].rolling(5).mean())
data_analysis["Oil_month_P_value"] = (data_analysis["Oil_month_P_value"].pct_change()*100)
data_analysis["inflation_exp_value"] = (data_analysis["inflation_exp_value"].pct_change()*100).diff(1)
data_analysis["unemployment_value"] = data_analysis["unemployment_value"] - data_analysis["unemployment_value"].rolling(2).mean()
data_analysis["unemployment_value"] = data_analysis["unemployment_value"]
data_analysis["fed_funds_value"] = data_analysis["fed_funds_value"].rolling(6).mean()
data_analysis["retail_sales_value"] = (data_analysis["retail_sales_value"].pct_change()*100).rolling(3).mean()
data_analysis["core_cpi_value"] = (data_analysis["core_cpi_value"].pct_change()*100)
data_analysis["ppi_value"] = (data_analysis["ppi_value"].pct_change()*100).diff(1)
data_analysis["wage_value"] = (data_analysis["wage_value"].pct_change()*100)
data_analysis["housing_price_value"] = data_analysis["housing_price_value"].pct_change()*100
data_analysis["housing_price_value"] = data_analysis["housing_price_value"] - data_analysis["housing_price_value"].rolling(3).mean()
data_analysis["rent_value"] = data_analysis["rent_value"].pct_change()*100


data_analysis["inflation_next_month"] = data_analysis["inflation_value"].shift(-1)

data_analysis["inflation_lag1"] = data_analysis["inflation_value"].shift(1)

unemp_Model = pd.DataFrame()
for i in range(1,1):
    unemp_Model[f"unemp_lag{i}"] =  data_analysis["unemployment_value"].shift(i)


rent_Model = pd.DataFrame()
for i in range(1,3):
    rent_Model[f"rent_lag{i}"] =  data_analysis["rent_value"].shift(i)


ARL_Model = pd.DataFrame()
for i in range(1,1):
    ARL_Model[f"ARL_lag{i}"] =  data_analysis["IND_pro_value"].shift(i)


m2_Model = pd.DataFrame()
for i in range(1,2):
    m2_Model[f"m2_lag{i}"] =  data_analysis["Money_Supply_M2_value"].shift(i)

inexp_Model = pd.DataFrame()
for i in range(1,1):
    inexp_Model[f"inexp_lag{i}"] =  data_analysis["inflation_exp_value"].shift(i)

core_cpi_Model = pd.DataFrame()
for i in range(1,4):
    core_cpi_Model[f"corecpi_lag{i}"] =  data_analysis["core_cpi_value"].shift(i)

Senti_Model = pd.DataFrame()
for i in range(1,2):
    Senti_Model[f"Senti_lag{i}"] =  data_analysis["sentiment_value"].shift(i)

AR_Model = pd.DataFrame()
AR_Model[["inflation_next_month","lag0"]] =  data_analysis[["inflation_next_month","inflation_value"]]

for i in range(1,2):
    AR_Model[f"lag{i}"] =  data_analysis["inflation_value"].shift(i)



AR_Model3 = pd.concat((AR_Model,data_analysis["sentiment_value"],data_analysis["Oil_month_P_value"],
                       core_cpi_Model,Senti_Model,m2_Model,data_analysis["Money_Supply_M2_value"],
                       inexp_Model,ARL_Model,data_analysis["fed_funds_value"],data_analysis["rolling_inflation_value"]
                       ,data_analysis["wage_value"],rent_Model,unemp_Model,data_analysis["IND_pro_value"]
                    ,data_analysis["core_cpi_value"],data_analysis["unemployment_value"],
                    data_analysis["ppi_value"],data_analysis["retail_sales_value"]
                    ,data_analysis["inflation_exp_value"],data_analysis["rent_value"],data_analysis["housing_price_value"]
                    ),axis=1,join="outer")

future_data = AR_Model3.loc[[AR_Model3.index[-1]]]
X_future = future_data.iloc[:,1:]
month_date = AR_Model3.index[-1]+pd.DateOffset(months=1)
next_month = (month_date).strftime("%B %Y")

AR_clean3 = AR_Model3.dropna()

Y = AR_clean3.iloc[:,0]
X = AR_clean3.iloc[:,1:]

tscv = TimeSeriesSplit(n_splits=5)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
model = Ridge()
all_predictions = []
all_actual = []
for train_index, test_index in tscv.split(X):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_train = Y.iloc[train_index]
    Y_test = Y.iloc[test_index]
    all_actual.append(Y_test)
    model.fit(X_train,Y_train)
    Y_predictor = model.predict(X_test)
    all_predictions.append(Y_predictor.tolist())
all_actual = np.concatenate(all_actual)
all_predictions = np.concatenate(all_predictions)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
MAE = mean_absolute_error(all_actual,all_predictions)
MSE = mean_squared_error(all_actual,all_predictions)
RMSE = (MSE)**0.5
R_squared = r2_score(all_actual,all_predictions)
#print(MAE)
#print(RMSE)
#print(R_squared)
actual_direction = np.sign(Y_test.values - data_analysis.loc[Y_test.index,"inflation_value"].values)
predicted_direction = np.sign(Y_predictor - data_analysis.loc[Y_test.index,"inflation_value"].values)
directional_accuracy = np.mean(actual_direction==predicted_direction)*100
#print(directional_accuracy)
#print(AR_clean3.shape)
prediction_mom = model.predict(X_future)
print("change in MoM% cpi "+str(next_month)+" is "+str(prediction_mom))
graph_data = pd.concat([data_analysis["inflation_value"].iloc[-6:],pd.Series([prediction_mom[0]],index=[month_date])])
plt.plot(graph_data.index, graph_data.values)
plt.xlabel("Month")
plt.ylabel("Monthly Inflation(%)")
for x,y in graph_data.items():
    plt.annotate(f"{y:.2f}%",(x,y))
    plt.scatter(x,y,marker="o")
plt.show()

