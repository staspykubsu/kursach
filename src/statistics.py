from dataclasses import dataclass, field
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px


@dataclass
class StatisticalTest:
    """Базовый класс для статистических тестов"""
    name: str
    source_df_id: str
    test_type: str
    result: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, df: pd.DataFrame) -> Dict[str, Any]:
        raise NotImplementedError
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "test_type": self.test_type,
            "source_df_id": self.source_df_id,
            "result": self.result
        }


@dataclass
class CorrelationAnalysis(StatisticalTest):
    """Корреляционный анализ"""
    columns: List[str] = field(default_factory=list)
    method: str = "pearson"
    test_type: str = "correlation"
    
    def execute(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.columns = numeric_cols[:10]
        
        corr_matrix = df[self.columns].corr(method=self.method)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}'
        ))
        
        fig.update_layout(
            title=f'Корреляционная матрица ({self.method})',
            height=600
        )
        
        self.result = {
            "correlation_matrix": corr_matrix.to_dict(),
            "figure": fig,
            "high_correlations": self._find_high_correlations(corr_matrix)
        }
        
        return self.result
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    high_corr.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": round(corr_matrix.iloc[i, j], 3)
                    })
        return sorted(high_corr, key=lambda x: abs(x['correlation']), reverse=True)


@dataclass
class TTestAnalysis(StatisticalTest):
    """T-тест для сравнения двух групп"""
    column: str = ""
    group_column: str = ""
    group1: Any = None
    group2: Any = None
    test_type: str = "ttest"
    
    def execute(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.column or not self.group_column:
            return {"error": "Выберите колонки для анализа"}
        
        group1_data = df[df[self.group_column] == self.group1][self.column].dropna()
        group2_data = df[df[self.group_column] == self.group2][self.column].dropna()
        
        if len(group1_data) == 0 or len(group2_data) == 0:
            return {"error": "Одна из групп пуста"}
        
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=group1_data,
            name=str(self.group1),
            boxmean='sd'
        ))
        
        fig.add_trace(go.Box(
            y=group2_data,
            name=str(self.group2),
            boxmean='sd'
        ))
        
        fig.update_layout(
            title=f'T-тест: {self.column} по {self.group_column}<br>p-value: {p_value:.4f}',
            yaxis_title=self.column
        )
        
        self.result = {
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,
            "group1_stats": {
                "mean": round(group1_data.mean(), 3),
                "std": round(group1_data.std(), 3),
                "size": len(group1_data)
            },
            "group2_stats": {
                "mean": round(group2_data.mean(), 3),
                "std": round(group2_data.std(), 3),
                "size": len(group2_data)
            },
            "figure": fig
        }
        
        return self.result


@dataclass
class ChiSquaredTest(StatisticalTest):
    """Хи-квадрат тест для категориальных переменных"""
    column1: str = ""
    column2: str = ""
    test_type: str = "chi_squared"
    
    def execute(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.column1 or not self.column2:
            return {"error": "Выберите две категориальные колонки"}
        
        contingency_table = pd.crosstab(df[self.column1], df[self.column2])
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        fig = px.imshow(
            contingency_table,
            text_auto=True,
            aspect="auto",
            title=f'Таблица сопряженности: {self.column1} vs {self.column2}'
        )
        
        self.result = {
            "chi2_statistic": round(chi2, 4),
            "p_value": round(p_value, 4),
            "degrees_of_freedom": dof,
            "significant": p_value < 0.05,
            "contingency_table": contingency_table.to_dict(),
            "figure": fig
        }
        
        return self.result


@dataclass
class ANOVAAnalysis(StatisticalTest):
    """Однофакторный ANOVA"""
    column: str = ""
    group_column: str = ""
    test_type: str = "anova"
    
    def execute(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.column or not self.group_column:
            return {"error": "Выберите колонки для анализа"}
        
        groups = []
        group_names = []
        
        for name, group in df.groupby(self.group_column):
            groups.append(group[self.column].dropna().values)
            group_names.append(str(name))
        
        if len(groups) < 2:
            return {"error": "Недостаточно групп для анализа"}
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        fig = go.Figure()
        
        for name, group in zip(group_names, groups):
            fig.add_trace(go.Box(
                y=group,
                name=name,
                boxmean='sd'
            ))
        
        fig.update_layout(
            title=f'ANOVA: {self.column} по {self.group_column}<br>p-value: {p_value:.4f}',
            yaxis_title=self.column
        )
        
        self.result = {
            "f_statistic": round(f_stat, 4),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,
            "groups": {name: {
                "mean": round(group.mean(), 3),
                "std": round(group.std(), 3),
                "size": len(group)
            } for name, group in zip(group_names, groups)},
            "figure": fig
        }
        
        return self.result