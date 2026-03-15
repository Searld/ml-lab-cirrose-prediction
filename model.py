import os
import logging
import pandas as pd
import numpy as np
import joblib
import fire
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from clearml import Task

# Настройка логирования
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

    def _get_features(self, df, stats=None):
        """Метод для создания признаков (Feature Engineering)"""
        df = df.copy()
        
        # Заполнение пропусков
        if stats:
            for col in self.numeric_cols:
                df[col] = df[col].fillna(stats['numeric'][col])
            for col in self.categorical_cols:
                df[col] = df[col].fillna(stats['categorical'][col])
        
        # Feature Engineering как в твоем новом коде
        df["Age_years"] = df["Age"] / 365
        df["Bilirubin_Albumin"] = df["Bilirubin"] / (df["Albumin"] + 1e-5)
        for col in ["Bilirubin", "Alk_Phos", "SGOT"]:
            df[f"log_{col}"] = np.log1p(df[col])
            
        for col in self.categorical_cols:
            df[col] = df[col].astype(str)
            
        return df

    def train(self, dataset: str):
        task = Task.init(project_name='Cirrhosis_Prediction', task_name='KFold_CatBoost_Training')
        task.upload_artifact('train_dataset', artifact_object=dataset)
        
        try:
            train_df = pd.read_csv(dataset)
            
            # Считаем статистики для продакшена
            stats = {
                'numeric': train_df[self.numeric_cols].median().to_dict(),
                'categorical': {col: train_df[col].mode()[0] for col in self.categorical_cols}
            }
            
            X_full = self._get_features(train_df, stats)
            y = X_full["Status"]
            X = X_full.drop(["id", "Status"], axis=1)
            
            cat_features = [X.columns.get_loc(col) for col in self.categorical_cols]
            
            # Кросс-валидация
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            oof = np.zeros((len(X), 3))
            
            logger.info("Начало кросс-валидации (5 фолдов)...")
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = CatBoostClassifier(
                    iterations=2000, learning_rate=0.03, depth=6,
                    loss_function="MultiClass", eval_metric="MultiClass",
                    random_seed=42, early_stopping_rounds=200, verbose=0
                )
                
                model.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_val, y_val))
                oof[val_idx] = model.predict_proba(X_val)
                logger.info(f"Fold {fold+1} завершен.")

            cv_score = log_loss(y, oof)
            print(f"🔥 FINAL CV LOGLOSS: {cv_score:.4f}")
            
            # Обучаем финальную модель на всех данных
            final_model = CatBoostClassifier(
                iterations=2000, learning_rate=0.03, depth=6,
                loss_function="MultiClass", random_seed=42, verbose=100
            )
            final_model.fit(X, y, cat_features=cat_features)

            # Сохранение всего необходимого
            os.makedirs('./model', exist_ok=True)
            final_model.save_model('./model/best_model.bin')
            joblib.dump({'stats': stats, 'columns': X.columns.tolist(), 'classes': final_model.classes_.tolist()}, './model/metadata.pkl')
            
            task.get_logger().report_single_value(name='Final CV LogLoss', value=cv_score)
            print("✅ Модель обучена и сохранена.")

        except Exception as e:
            logger.error(f"Ошибка: {e}")
            raise

    def predict(self, dataset: str):
        try:
            # Загрузка метаданных
            meta = joblib.load('./model/metadata.pkl')
            model = CatBoostClassifier()
            model.load_model('./model/best_model.bin')
            
            test_df = pd.read_csv(dataset)
            processed_test = self._get_features(test_df, meta['stats'])
            X_test = processed_test.reindex(columns=meta['columns'], fill_value=0)
            
            preds = model.predict_proba(X_test)
            
            submission = pd.DataFrame(preds, columns=[f'Status_{c}' for c in meta['classes']])
            submission.insert(0, 'id', test_df['id'])
            
            submission.to_csv('./data/results.csv', index=False)
            print("✅ Предсказания сохранены в ./data/results.csv")

        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            raise

if __name__ == '__main__':
    fire.Fire(My_Classifier_Model)