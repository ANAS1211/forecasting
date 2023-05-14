import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error

color_pal=sns.color_palette()

df=pd.read_csv("AEP_hourly.csv")
print(df.head())
#changer la colonne des indices par la colonne choisie
df=df.set_index("Datetime")
print(df.head())
# print("index est:",df.index)
df.index=pd.to_datetime(df.index) #mettre en forme la barre des absices 
# print(df.head())
df.plot(style='.',figsize=(15,5),color=color_pal[0], title="PJME Energy Use in MW ")

plt.style.use('fivethirtyeight')
# Train test split 
#pour l'entrainement on a choisi les dates < à janvier 2015
# print("indices",df.index)
train=df.loc[df.index < '01-01-2015']
test=df.loc[df.index>= '01-01-2015']
fig, ax = plt.subplots(figsize=(15,5))
train.plot(ax=ax,label="training set",title="Data Train/Test Split")
test.plot(ax=ax,label="testing set")
#ajouter une ligne noir coupée pour séparation
ax.axvline('01-01-2015', color='black', ls='--')
ax.legend(['Training set','Testing set'])
plt.show()
#tracer une semaine de consommation
df.loc[(df.index >'01-01-2010') & (df.index < '01-08-2010')].plot(figsize=(15,5), title="Week of data")
plt.show()

# Create features
def create_features(df):
    """create time series features based on time series indexes"""
    df=df.copy()
    df['hour']=df.index.hour
    df['dayofweek']=df.index.dayofweek 
    df['quarter']=df.index.quarter #spliter l'année en quadrimestre
    df['month']=df.index.month
    df['year']=df.index.year
    df["dayOfyear"]=df.index.dayofyear
    return df
df=create_features(df)
print(df.head())

#Visualize our Feature/Target relationship
fig, ax =plt.subplots(figsize=(10,8))
sns.boxplot(data=df, x='hour', y="AEP_MW")
ax.set_title('MW by hour')
plt.show()
sns.boxplot(data=df, x='month', y="AEP_MW", color="black")
ax.set_title('MW by month')
plt.show()

# Create the model

#créer un modèle de regression utilisant xgb regressor
train=create_features(train)
test=create_features(test)
print(df.columns)

#Séparer les features de la cible
FEATURES= [ 'hour', 'dayofweek', 'quarter', 'month', 'year', 'dayOfyear']
TARGET='AEP_MW'
#entrainter X et y 
X_train=train[FEATURES]
y_train=train[TARGET]
X_test=test[FEATURES]
y_test=test[TARGET]
reg= xgb.XGBRegressor(n_estimators=1000,early_stopping_rounds=50,learning_rate=0.01)
#créer le modèle de regression sur la base de X_train et y_train
reg.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test,y_test)],verbose=100)
#early_stopping_rounds=50,verbose=True
#si le jeu de test ne s'améliore pas après 50 arbres (voir le principe XGB) on doit mettre le verbose

#Feature importance

fi=pd.DataFrame(data=reg.feature_importances_,index=reg.feature_names_in_,columns=['importance'])
fi.sort_values('importance').plot(kind="barh",title="Feature Importance")
plt.show()

#Forecast on test

test['predictions'] = reg.predict(X_test)
print(df.head())
#fusionner le jeu de données de test avec les prédictions deux dataframes 
df=df.merge(test["predictions"], how="left",left_index=True,right_index=True)
ax=df[["AEP_MW"]].plot(figsize=(15,5))
df['predictions'].plot(ax=ax, style='.')
plt.legend(["Truth data","Legend"])
ax.set_title('Raw data and prediction')
plt.show()

df.loc[(df.index >'04-01-2018') & (df.index <= '04-08-2018')]["AEP_MW"]\
    .plot(figsize=(15,5), title="Week of data")
df.loc[(df.index >'04-01-2018') & (df.index <= '04-08-2018')]["predictions"]\
    .plot(style='.')
plt.legend(['Truth Data','Prediction'])
plt.show()

score=np.sqrt(mean_squared_error(test["predictions"],test["AEP_MW"]))
print(f"RMSE SCORE ON TEST SET :{score:0.2f}")

# Calculer l'erreur
#look 
test["error"]=np.abs(test[TARGET]-test['predictions'])
test["date"]=test.index.duplicated

test.groupby("date")["error"].mean().sort_values(ascending=False).head(10)

# Next steps
#more robust cross validation
#add more features