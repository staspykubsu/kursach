from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder


@dataclass
class DataPreprocessor:
    """Класс для предобработки данных"""
    name: str
    preprocessing_type: str
    columns: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessing_type == "normalize":
            return self._normalize(df)
        elif self.preprocessing_type == "standardize":
            return self._standardize(df)
        elif self.preprocessing_type == "label_encode":
            return self._label_encode(df)
        elif self.preprocessing_type == "one_hot_encode":
            return self._one_hot_encode(df)
        elif self.preprocessing_type == "remove_outliers":
            return self._remove_outliers(df)
        elif self.preprocessing_type == "binning":
            return self._binning(df)
        else:
            return df
    
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Нормализация данных (Min-Max)"""
        result_df = df.copy()
        columns = self.columns if self.columns else df.select_dtypes(include=[np.number]).columns
        
        scaler = MinMaxScaler()
        for col in columns:
            if col in result_df.columns and pd.api.types.is_numeric_dtype(result_df[col]):
                result_df[col] = scaler.fit_transform(result_df[[col]])
        
        return result_df
    
    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Стандартизация данных (Z-score)"""
        result_df = df.copy()
        columns = self.columns if self.columns else df.select_dtypes(include=[np.number]).columns
        
        scaler = StandardScaler()
        for col in columns:
            if col in result_df.columns and pd.api.types.is_numeric_dtype(result_df[col]):
                result_df[col] = scaler.fit_transform(result_df[[col]])
        
        return result_df
    
    def _label_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label Encoding для категориальных переменных"""
        result_df = df.copy()
        columns = self.columns if self.columns else df.select_dtypes(include=['object', 'category']).columns
        
        for col in columns:
            if col in result_df.columns:
                le = LabelEncoder()
                result_df[col] = le.fit_transform(result_df[col].astype(str))
        
        return result_df
    
    def _one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-Hot Encoding для категориальных переменных"""
        columns = self.columns if self.columns else df.select_dtypes(include=['object', 'category']).columns
        
        result_df = df.copy()
        for col in columns:
            if col in result_df.columns:
                dummies = pd.get_dummies(result_df[col], prefix=col, drop_first=True)
                result_df = pd.concat([result_df, dummies], axis=1)
                result_df = result_df.drop(columns=[col])
        
        return result_df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление выбросов методом IQR"""
        result_df = df.copy()
        columns = self.columns if self.columns else df.select_dtypes(include=[np.number]).columns
        
        method = self.params.get('method', 'iqr')
        threshold = self.params.get('threshold', 1.5)
        
        if method == 'iqr':
            for col in columns:
                if col in result_df.columns and pd.api.types.is_numeric_dtype(result_df[col]):
                    Q1 = result_df[col].quantile(0.25)
                    Q3 = result_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    result_df = result_df[
                        (result_df[col] >= lower_bound) & 
                        (result_df[col] <= upper_bound)
                    ]
        elif method == 'zscore':
            z_threshold = self.params.get('z_threshold', 3)
            for col in columns:
                if col in result_df.columns and pd.api.types.is_numeric_dtype(result_df[col]):
                    z_scores = np.abs((result_df[col] - result_df[col].mean()) / result_df[col].std())
                    result_df = result_df[z_scores < z_threshold]
        
        return result_df
    
    def _binning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Биннинг числовых данных"""
        result_df = df.copy()
        columns = self.columns if self.columns else df.select_dtypes(include=[np.number]).columns
        n_bins = self.params.get('n_bins', 5)
        
        for col in columns:
            if col in result_df.columns and pd.api.types.is_numeric_dtype(result_df[col]):
                result_df[f'{col}_binned'] = pd.cut(
                    result_df[col], 
                    bins=n_bins, 
                    labels=[f'Bin_{i+1}' for i in range(n_bins)]
                )
        
        return result_df