# pylint: disable=no-member

from metaflow import FlowSpec, step, card, Parameter, environment, current
import pandas as pd
import mlflow
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from metaflow.cards import VegaChart, Markdown

class HousingPriceFlow(FlowSpec):
    DATA_PATH = Parameter(
        'data_path',
        default='data/housing.csv',
        help='Path to housing dataset'
    )
    
    HYPEROPT_ITER = Parameter(
        'hyperopt_iter',
        default=50,
        type=int,
        help='Number of Hyperopt iterations'
    )

    @environment(vars={
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
    })
    @card
    @step
    def start(self):
        """Initialize MLflow and setup models"""
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
        mlflow.set_experiment("Housing Price Prediction")
        
        self.model_configs = [
            {'name': 'RandomForest', 'model': RandomForestRegressor, 'tune': True},
            {'name': 'GradientBoosting', 'model': GradientBoostingRegressor, 'tune': True},
            {'name': 'LinearRegression', 'model': LinearRegression, 'tune': False},
            {'name': 'KNeighbors', 'model': KNeighborsRegressor, 'tune': True}
        ]
        
        self.next(self.load_data)

    @card
    @step
    def load_data(self):
        """Load and log dataset"""
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Housing Price Prediction")
        self.df = pd.read_csv(self.DATA_PATH)
        
        with mlflow.start_run(run_name="data_loading") as run:
            dataset = mlflow.data.from_pandas(
                self.df,
                source=self.DATA_PATH,
                name="Housing Prices",
                targets="price"
            )
            
            mlflow.log_input(dataset, context="training")
            mlflow.log_artifact(self.DATA_PATH)
            mlflow.log_dict(self.df.describe().to_dict(), "data_stats.json")
            
            self.dataset_info = {
                'run_id': run.info.run_id,
                'dataset': dataset.to_dict()
            }
        
        self.next(self.eda)

    @card
    @step
    def eda(self):
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Housing Price Prediction")
        """Exploratory Data Analysis for Housing Prices"""
        df_encoded = self.df.copy()
        encoder = LabelEncoder()
        categorical_cols = [
            'mainroad', 'guestroom', 'basement',
            'hotwaterheating', 'airconditioning', 'prefarea',
            'furnishingstatus'
        ]
        
        for col in categorical_cols:
            df_encoded[col] = encoder.fit_transform(df_encoded[col])

        current.card.append(Markdown("# Dataset Statistics"))
        current.card.append(Markdown(
            f"* Number of samples: {len(df_encoded)}\n"
            f"* Number of features: {len(df_encoded.columns)}\n"
            f"* Missing values: {df_encoded.isna().sum().sum()}\n"
            f"* Duplicated rows: {df_encoded.duplicated().sum()}"
        ))
        
        current.card.append(Markdown("## Price Distribution"))
        current.card.append(VegaChart({
            "data": {"values": self.df[['price']].to_dict('records')},
            "mark": {"type": "bar", "binSpacing": 0},
            "encoding": {
                "x": {"field": "price", "bin": {"maxbins": 30}, "title": "Price"},
                "y": {"aggregate": "count", "title": "Count"}
            }
        }))
        
        cat_features = ['bedrooms', 'bathrooms', 'stories', 'parking']
        
        for feature in cat_features:
            counts = self.df[feature].value_counts().reset_index()
            current.card.append(Markdown(f"## {feature} Distribution"))
            current.card.append(VegaChart({
                "data": {"values": counts.to_dict('records')},
                "mark": "bar",
                "encoding": {
                    "x": {"field": "index", "type": "nominal", "title": feature},
                    "y": {"field": feature, "type": "quantitative", "title": "Count"},
                    "color": {"field": "index", "type": "nominal"}
                }
            }))
        
        relationships = [
            ('bedrooms', 'parking'),
            ('bedrooms', 'bathrooms'),
            ('bedrooms', 'stories'),
            ('parking', 'furnishingstatus'),
            ('stories', 'furnishingstatus'),
            ('bathrooms', 'furnishingstatus'),
            ('bathrooms', 'prefarea'),
            ('stories', 'prefarea'),
            ('parking', 'prefarea')
        ]
        
        for x_feature, hue_feature in relationships:
            data = df_encoded.groupby([x_feature, hue_feature]).size().reset_index(name='count')
            current.card.append(Markdown(f"## {x_feature} vs {hue_feature}"))
            current.card.append(VegaChart({
                "data": {"values": data.to_dict('records')},
                "mark": "bar",
                "encoding": {
                    "x": {"field": x_feature, "type": "ordinal"},
                    "y": {"field": "count", "type": "quantitative"},
                    "color": {"field": hue_feature, "type": "nominal"}
                }
            }))
        
        corr_data = []
        corr_matrix = df_encoded.corr()
        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                corr_data.append({
                    "var1": col1,
                    "var2": col2,
                    "correlation": corr_matrix.loc[col1, col2]
                })
        
        current.card.append(Markdown("## Feature Correlation Matrix"))
        current.card.append(VegaChart({
            "data": {"values": corr_data},
            "mark": "rect",
            "encoding": {
                "x": {"field": "var1", "type": "nominal"},
                "y": {"field": "var2", "type": "nominal"},
                "color": {
                    "field": "correlation", 
                    "type": "quantitative",
                    "scale": {"domain": [-1, 1]}
                }
            },
            "config": {
                "axis": {"labelAngle": -45},
                "view": {"stroke": None}
            }
        }))

        self.next(self.preprocess)

    @card
    @step
    def preprocess(self):
        """Data preprocessing pipeline"""
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Housing Price Prediction")
        encoder = LabelEncoder()
        categorical_cols = [
            'mainroad', 'guestroom', 'basement',
            'hotwaterheating', 'airconditioning', 'prefarea',
            'furnishingstatus'
        ]
        
        for col in categorical_cols:
            self.df[col] = encoder.fit_transform(self.df[col])
        
        self.X = self.df.drop('price', axis=1)
        self.y = self.df['price'].values.reshape(-1, 1)
        
        self.scaler = MinMaxScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.y = self.scaler.fit_transform(self.y)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        self.next(self.hyperparameter_tuning, foreach='model_configs')

    @card
    @step
    def hyperparameter_tuning(self):
        """Hyperparameter optimization with Hyperopt"""
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Housing Price Prediction")
        config = self.input
        model_class = config['model']
        model_name = config['name']
        tune = config['tune']
        
        if tune:
            def objective(params):
                with mlflow.start_run(nested=True):
                    if model_name == 'RandomForest':
                        model_params = {
                            'n_estimators': int(params['n_estimators']),
                            'max_depth': int(params['max_depth']),
                            'min_samples_split': params['min_samples_split']
                        }
                    elif model_name == 'GradientBoosting':
                        model_params = {
                            'n_estimators': int(params['n_estimators']),
                            'learning_rate': params['learning_rate'],
                            'max_depth': int(params['max_depth'])
                        }
                    elif model_name == 'KNeighbors':
                        model_params = {
                            'n_neighbors': int(params['n_neighbors']),
                            'weights': ['uniform', 'distance'][int(params['weights'])],  
                            'p': int(params['p'])
                        }
                    else:
                        raise ValueError(f"Unsupported model for tuning: {model_name}")

                    model = model_class(**model_params)
                    scores = cross_val_score(model, self.X_train, self.y_train.ravel(),
                                           cv=5, scoring='r2')
                    mean_r2 = np.mean(scores)
                    
                    mlflow.log_params(model_params)
                    mlflow.log_metric("cv_r2", mean_r2)
                    
                    return {'loss': -mean_r2, 'status': STATUS_OK}
            
            if model_name == 'RandomForest':
                space = {
                    'n_estimators': hp.quniform('n_estimators', 50, 500, 50),
                    'max_depth': hp.quniform('max_depth', 3, 15, 1),
                    'min_samples_split': hp.uniform('min_samples_split', 0.01, 1)
                }
            elif model_name == 'GradientBoosting':
                space = {
                    'n_estimators': hp.quniform('n_estimators', 50, 500, 50),
                    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
                    'max_depth': hp.quniform('max_depth', 3, 10, 1)
                }
            elif model_name == 'KNeighbors':
                space = {
                    'n_neighbors': hp.quniform('n_neighbors', 3, 15, 1),
                    'weights': hp.choice('weights', [0, 1]),  # 0: uniform, 1: distance
                    'p': hp.quniform('p', 1, 2, 1)
                }
            else:
                raise ValueError(f"Unsupported model for space definition: {model_name}")

            trials = Trials()
            best = fmin(fn=objective,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=self.HYPEROPT_ITER,
                        trials=trials)
            
            self.best_params = best
        else:
            with mlflow.start_run(nested=True):
                model = model_class()
                scores = cross_val_score(model, self.X_train, self.y_train.ravel(),
                                       cv=5, scoring='r2')
                mean_r2 = np.mean(scores)
                mlflow.log_params(model.get_params())
                mlflow.log_metric("cv_r2", mean_r2)
                self.best_params = model.get_params()
        
        self.model_name = model_name
        self.next(self.join_tuning)

    @step
    def join_tuning(self, inputs):
        """Aggregate tuning results"""
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Housing Price Prediction")
        self.X_train = inputs[0].X_train
        self.X_test = inputs[0].X_test
        self.y_train = inputs[0].y_train
        self.y_test = inputs[0].y_test

        self.best_configs = {}
        for inp in inputs:
            self.best_configs[inp.model_name] = {
                'params': inp.best_params,
                'model_class': inp.input['model']
            }
        self.next(self.train_final)

    @card
    @step
    def train_final(self):
        """Final model training with best parameters"""
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Housing Price Prediction")
        self.models = []
        
        for model_name, config in self.best_configs.items():
            with mlflow.start_run(run_name=f"best_{model_name}"):
                params = self._convert_params(config['params'], model_name)
                model = config['model_class'](**params)
                model.fit(self.X_train, self.y_train.ravel())
                
                y_pred = model.predict(self.X_test)
                r2 = r2_score(self.y_test, y_pred)
                mse = mean_squared_error(self.y_test, y_pred)
                
                mlflow.log_params(params)
                mlflow.log_metrics({
                    'test_r2': r2,
                    'test_mse': mse
                })
                mlflow.sklearn.log_model(model, f"{model_name}_model")
                
                self.models.append({
                    'name': model_name,
                    'model': model,
                    'r2': r2,
                    'mse': mse
                })
        
        self.next(self.end)

    def _convert_params(self, params, model_name):
        if model_name == 'RandomForest':
            return {
                'n_estimators': int(params['n_estimators']),
                'max_depth': int(params['max_depth']),
                'min_samples_split': params['min_samples_split']
            }
        elif model_name == 'GradientBoosting':
            return {
                'n_estimators': int(params['n_estimators']),
                'learning_rate': params['learning_rate'],
                'max_depth': int(params['max_depth'])
            }
        elif model_name == 'KNeighbors':
            return {
                'n_neighbors': int(params['n_neighbors']),
                'weights': ['uniform', 'distance'][int(params['weights'])],  # Fix here
                'p': int(params['p'])
            }
        elif model_name == 'LinearRegression':
            return {}
        else:
            raise ValueError(f"Unsupported model for param conversion: {model_name}")

    @card
    @step
    def end(self):
        """Final results comparison"""
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Housing Price Prediction")
        model_names = [m['name'] for m in self.models]
        r2_scores = [m['r2'] for m in self.models]
        
        chart = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {
                "values": [
                    {"model": name, "r2": score}
                    for name, score in zip(model_names, r2_scores)
                ]
            },
            "mark": "bar",
            "encoding": {
                "x": {"field": "model", "type": "nominal"},
                "y": {"field": "r2", "type": "quantitative"}
            }
        }
        
        best_model = max(self.models, key=lambda x: x['r2'])
        
        current.card.append(Markdown("# Final Results"))
        current.card.append(Markdown(f"## Best Model: {best_model['name']}"))
        current.card.append(Markdown(f"**R2 Score**: {best_model['r2']:.4f}"))
        current.card.append(Markdown(f"**MSE**: {best_model['mse']:.2f}"))
        current.card.append(VegaChart(chart))
        
        print(f"Pipeline completed. Best model: {best_model['name']}")

if __name__ == '__main__':
    HousingPriceFlow()