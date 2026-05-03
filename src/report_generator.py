import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List
import json
from datetime import datetime


class ReportGenerator:
    """Генератор отчетов в HTML"""
    
    def __init__(self, app_state):
        self.app_state = app_state
    
    def generate_html_report(self) -> str:
        """Генерация HTML отчета"""
        html = self._get_html_header()
        
        # Основная информация
        html += self._get_dataset_info()
        
        # Операции с данными
        html += self._get_operations_info()
        
        # Статистический анализ
        html += self._get_statistical_analysis()
        
        # Графики
        html += self._get_charts()
        
        html += self._get_html_footer()
        
        return html
    
    def _get_html_header(self) -> str:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Отчет анализа данных - SEDES</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .info-box {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .warning {{ color: #e74c3c; font-weight: bold; }}
                .success {{ color: #27ae60; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>📊 Отчет анализа данных</h1>
            <p>Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
    
    def _get_html_footer(self) -> str:
        return """
        </body>
        </html>
        """
    
    def _get_dataset_info(self) -> str:
        html = "<h2>1. Информация о датасете</h2>"
        
        current_df_id = self.app_state.current_df_id
        if current_df_id and current_df_id in self.app_state.dataframes:
            df = self.app_state.dataframes[current_df_id]
            
            html += '<div class="info-box">'
            html += f"<p><strong>Размер:</strong> {df.shape[0]} строк × {df.shape[1]} колонок</p>"
            html += f"<p><strong>Колонки:</strong> {', '.join(df.columns.tolist())}</p>"
            html += "</div>"
            
            # Базовая статистика
            html += "<h3>Базовая статистика числовых колонок</h3>"
            html += df.describe().to_html(classes='table')
            
            # Пропущенные значения
            html += "<h3>Пропущенные значения</h3>"
            missing = pd.DataFrame({
                'Колонка': df.columns,
                'Пропущено': df.isna().sum(),
                'Процент': (df.isna().sum() / len(df) * 100).round(2)
            })
            html += missing.to_html(classes='table', index=False)
        
        return html
    
    def _get_operations_info(self) -> str:
        html = "<h2>2. Выполненные операции</h2>"
        
        if not self.app_state.operations:
            html += "<p>Операции не выполнялись</p>"
            return html
        
        html += "<ol>"
        for op in self.app_state.operations:
            html += f"<li><strong>{op.name}</strong> (тип: {op.operation_type})</li>"
        html += "</ol>"
        
        return html
    
    def _get_statistical_analysis(self) -> str:
        html = "<h2>3. Статистический анализ</h2>"
        
        current_df_id = self.app_state.current_df_id
        if current_df_id and current_df_id in self.app_state.dataframes:
            df = self.app_state.dataframes[current_df_id]
            
            # Корреляционный анализ
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                html += "<h3>Корреляционный анализ</h3>"
                corr_matrix = df[numeric_cols].corr()
                
                # Поиск сильных корреляций
                html += "<p><strong>Сильные корреляции (|r| > 0.7):</strong></p><ul>"
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) >= 0.7:
                            html += f"<li>{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: <span class='warning'>{corr_matrix.iloc[i, j]:.3f}</span></li>"
                html += "</ul>"
                
                html += corr_matrix.round(3).to_html(classes='table')
        
        return html
    
    def _get_charts(self) -> str:
        html = "<h2>4. Визуализации</h2>"
        
        if self.app_state.components:
            for i, comp in enumerate(self.app_state.components):
                if hasattr(comp, 'chart'):
                    html += f"<h3>График {i+1}</h3>"
                    html += f'<div id="chart_{i}"></div>'
        
        return html
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Генерация сводной статистики"""
        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=['number']).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
            "missing_values": int(df.isna().sum().sum()),
            "missing_percentage": round((df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100), 2)
        }
        return summary