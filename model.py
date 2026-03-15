import os
import logging
import pandas as pd
import numpy as np
import joblib
import fire
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from clearml import Task

# --- Настройка логирования (Требование задания) ---
os.makedirs('./data', exist_ok=True)
logging.basicConfig(
    filename='./data/log_file.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class My_Classifier_Model:
    def __init__(self):
        self.categorical_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
        self.numeric_cols = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 
                             'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']

    def _preprocess(self, df, is_train=True):
        """Внутренний метод для обработки данных"""
        df = df.copy()
        
        # Обработка числовых признаков
        for col in self.numeric_cols:
            if col in df.columns:
                val = df[col].median()
                df[col] = df[col].fillna(val)
                df[f'{col}_missing'] = df[col].isnull().astype(int)

        # Обработка категориальных признаков
        for col in self.categorical_cols:
            if col in df.columns:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val).astype(str)
        
        return df

    def train(self, dataset: str):
        """Метод обучения с использованием Optuna и ClearML"""
        task = Task.init(project_name='Cirrhosis_Prediction', task_name='Optuna_CatBoost_Training')
        task.upload_artifact('train_dataset', artifact_object=dataset)
        
        try:
            logger.info(f"Загрузка данных для обучения: {dataset}")
            train_df = pd.read_csv(dataset)
            
            processed_df = self._preprocess(train_df)
            X = processed_df.drop(['id', 'Status'], axis=1)
            y = processed_df['Status']
            
            cat_features = [X.columns.get_loc(col) for col in self.categorical_cols if col in X.columns]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            logger.info("Запуск Optuna...")
            def objective(trial):
                params = {
                    'iterations': trial.suggest_int('iterations', 500, 1000),
                    'depth': trial.suggest_int('depth', 4, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                }
                model = CatBoostClassifier(**params, loss_function='MultiClass', random_seed=42, verbose=0)
                model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), early_stopping_rounds=50)
                return log_loss(y_val, model.predict_proba(X_val))

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=15)
            
            logger.info(f"Лучшие параметры: {study.best_params}")

            final_model = CatBoostClassifier(**study.best_params, loss_function='MultiClass', random_seed=42, verbose=100)
            final_model.fit(X, y, cat_features=cat_features)

            # Сохранение артефактов
            os.makedirs('./model', exist_ok=True)
            model_path = os.path.abspath('./model/best_model.bin')
            final_model.save_model(model_path)
            
            # ВАЖНО: Сохраняем список колонок (этого не хватало!)
            joblib.dump(X.columns.tolist(), './model/features.pkl')
            
            logger.info(f"Модель и признаки сохранены.")
            print(f"✅ Обучение завершено. Log Loss: {study.best_value:.4f}")

        except Exception as e:
            logger.error(f"Ошибка при обучении: {e}")
            raise

    def predict(self, dataset: str):
        """Метод предсказания"""
        try:
            logger.info(f"Запуск предсказания для: {dataset}")
            
            if not os.path.exists('./model/best_model.bin'):
                raise FileNotFoundError("Файл модели ./model/best_model.bin не найден!")

            model = CatBoostClassifier()
            model.load_model('./model/best_model.bin')
            feature_cols = joblib.load('./model/features.pkl')
            
            test_df = pd.read_csv(dataset)
            ids = test_df['id']
            
            processed_test = self._preprocess(test_df, is_train=False)
            X_test = processed_test.reindex(columns=feature_cols, fill_value=0)
            
            predictions = model.predict_proba(X_test)
            
            submission = pd.DataFrame(predictions, columns=[f'Status_{c}' for c in model.classes_])
            submission.insert(0, 'id', ids)
            
            submission.to_csv('./data/results.csv', index=False)
            
            logger.info("Предсказания сохранены в ./data/results.csv")
            print("✅ Файл результатов создан.")

        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            raise

if __name__ == '__main__':
    fire.Fire(My_Classifier_Model)