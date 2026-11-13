from pathlib import Path
import json, joblib, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"data"/"travel_cost_training_data.csv")
features=["destination","destination_type","distance_km","travellers","days","travel_mode","hotel_category","food_preference","season"]
cat=["destination","destination_type","travel_mode","hotel_category","food_preference","season"]
num=["distance_km","travellers","days"]
X=df[features]; y=df["estimated_total_cost"]
pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num",StandardScaler(),num)])
model=Pipeline([("preprocess",pre),("model",RandomForestRegressor(n_estimators=220,max_depth=18,min_samples_leaf=2,random_state=42,n_jobs=-1))])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
model.fit(X_train,y_train)
metrics={"mae":round(float(mean_absolute_error(y_test,model.predict(X_test))),2),
         "r2":round(float(r2_score(y_test,model.predict(X_test))),4),
         "train_rows":len(X_train),"test_rows":len(X_test)}
joblib.dump(model,ROOT/"ml_model"/"travel_cost_random_forest.pkl")
catalog=pd.read_csv(ROOT/"data"/"destination_catalog.csv")
knn_features=["distance_km","hotel_per_night","food_per_person_day","local_per_person_day"]
scaler=StandardScaler(); scaled=scaler.fit_transform(catalog[knn_features])
knn=NearestNeighbors(n_neighbors=5,metric="euclidean").fit(scaled)
joblib.dump({"model":knn,"scaler":scaler,"catalog":catalog,"features":knn_features},ROOT/"ml_model"/"destination_knn.pkl")
(ROOT/"ml_model"/"model_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
print("Models retrained successfully.")
print(json.dumps(metrics,indent=2))
