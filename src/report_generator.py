import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import base64
from io import BytesIO
import markdown
import re


class ReportGenerator:
    """Генератор HTML/PDF отчетов"""
    
    def __init__(self, app_state):
        self.app_state = app_state
        self.timestamp = datetime.now()
        self.md = markdown.Markdown(extensions=['extra', 'codehilite', 'tables'])
        
    def generate_html_report(self) -> str:
        """Генерация полного HTML отчета"""
        html = self._build_html_header()
        html += self._build_summary()
        html += self._build_data_operations()
        html += self._build_components()
        html += self._build_footer()
        return html
    
    def _markdown_to_html(self, text: str) -> str:
        """Конвертация Markdown текста в HTML"""
        if not text:
            return ""
        
        # Очищаем текст от лишних пробелов
        text = text.strip()
        
        # Конвертируем Markdown в HTML
        html = self.md.convert(text)
        
        # Добавляем стили для markdown элементов
        html = html.replace('<table>', '<table class="markdown-table">')
        html = html.replace('<h3>', '<h3 style="color: #667eea; margin-top: 20px;">')
        html = html.replace('<h4>', '<h4 style="color: #764ba2; margin-top: 15px;">')
        html = html.replace('<ul>', '<ul style="margin: 10px 0; padding-left: 25px;">')
        html = html.replace('<li>', '<li style="margin: 5px 0;">')
        html = html.replace('<p>', '<p style="margin: 8px 0; line-height: 1.5;">')
        html = html.replace('<strong>', '<strong style="color: #333;">')
        html = html.replace('<code>', '<code style="background: #f0f0f0; padding: 2px 6px; border-radius: 3px;">')
        
        return html
    
    def _build_html_header(self) -> str:
        """Построение заголовка HTML"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Отчет анализа данных</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #f0f2f5;
                    color: #333;
                    line-height: 1.6;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 40px;
                    border-radius: 12px;
                    box-shadow: 0 2px 20px rgba(0,0,0,0.08);
                }}
                .header {{
                    text-align: center;
                    padding: 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 12px;
                    margin-bottom: 40px;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 700;
                }}
                .header .meta {{
                    margin-top: 15px;
                    opacity: 0.95;
                    font-size: 1.1em;
                }}
                .section {{
                    margin: 30px 0;
                }}
                .section-title {{
                    font-size: 1.8em;
                    color: #667eea;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 3px solid #667eea;
                }}
                .operation-card {{
                    background: #fafbfc;
                    padding: 25px;
                    margin: 20px 0;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    border-left: 4px solid #667eea;
                    transition: transform 0.2s;
                }}
                .operation-card:hover {{
                    transform: translateX(5px);
                }}
                .operation-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 15px;
                    border-bottom: 2px solid #e1e4e8;
                }}
                .operation-number {{
                    background: #667eea;
                    color: white;
                    width: 35px;
                    height: 35px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    margin-right: 15px;
                }}
                .operation-title {{
                    font-size: 1.3em;
                    font-weight: 600;
                    color: #24292e;
                }}
                .operation-type-badge {{
                    padding: 6px 16px;
                    border-radius: 20px;
                    font-size: 0.85em;
                    font-weight: 500;
                    color: white;
                }}
                .badge-load {{ background: #28a745; }}
                .badge-filter {{ background: #dc3545; }}
                .badge-aggregate {{ background: #ffc107; color: #333; }}
                .badge-clean {{ background: #17a2b8; }}
                .badge-save {{ background: #6f42c1; }}
                
                .detail-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 12px;
                    margin: 20px 0;
                }}
                .detail-item {{
                    background: white;
                    padding: 12px 16px;
                    border-radius: 8px;
                    border: 1px solid #e1e4e8;
                }}
                .detail-label {{
                    font-size: 0.8em;
                    color: #586069;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    margin-bottom: 4px;
                }}
                .detail-value {{
                    font-size: 1.05em;
                    color: #24292e;
                    font-weight: 500;
                    word-break: break-word;
                }}
                
                .dataframe-container {{
                    margin: 20px 0;
                    overflow-x: auto;
                    border: 1px solid #e1e4e8;
                    border-radius: 8px;
                }}
                .dataframe-container table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.9em;
                }}
                .dataframe-container th {{
                    background: #667eea;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }}
                .dataframe-container td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #f0f0f0;
                }}
                .dataframe-container tr:nth-child(even) {{
                    background: #f8f9fa;
                }}
                .dataframe-container tr:hover {{
                    background: #f0f0f0;
                }}
                
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background: white;
                    padding: 18px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                    border: 1px solid #e1e4e8;
                }}
                .stat-value {{
                    font-size: 2em;
                    font-weight: 700;
                    color: #667eea;
                }}
                .stat-label {{
                    color: #586069;
                    margin-top: 5px;
                    font-size: 0.9em;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                
                .chart-container {{
                    margin: 25px 0;
                    padding: 20px;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    border: 1px solid #e1e4e8;
                }}
                .chart-container .js-plotly-plot {{
                    margin: 0 auto;
                }}
                
                .text-component {{
                    background: white;
                    padding: 25px;
                    margin: 20px 0;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    border-left: 4px solid #764ba2;
                    line-height: 1.7;
                }}
                .text-component h1, .text-component h2, .text-component h3, .text-component h4 {{
                    color: #667eea;
                    margin: 15px 0 10px 0;
                }}
                .text-component h3 {{
                    font-size: 1.3em;
                    border-bottom: 2px solid #e1e4e8;
                    padding-bottom: 8px;
                }}
                .text-component ul, .text-component ol {{
                    margin: 10px 0;
                    padding-left: 30px;
                }}
                .text-component li {{
                    margin: 5px 0;
                }}
                .text-component strong {{
                    color: #24292e;
                }}
                .text-component code {{
                    background: #f6f8fa;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                    font-size: 0.9em;
                }}
                .text-component table {{
                    border-collapse: collapse;
                    margin: 15px 0;
                    width: 100%;
                }}
                .text-component table th {{
                    background: #f6f8fa;
                    padding: 10px;
                    border: 1px solid #e1e4e8;
                }}
                .text-component table td {{
                    padding: 8px;
                    border: 1px solid #e1e4e8;
                }}
                
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .summary-card {{
                    background: linear-gradient(135deg, #667eea15, #764ba215);
                    padding: 25px;
                    border-radius: 12px;
                    text-align: center;
                    border: 2px solid #667eea30;
                }}
                .summary-number {{
                    font-size: 3em;
                    font-weight: 700;
                    color: #667eea;
                }}
                .summary-label {{
                    font-size: 1.1em;
                    color: #586069;
                    margin-top: 10px;
                }}
                
                .footer {{
                    text-align: center;
                    padding: 30px;
                    margin-top: 50px;
                    color: #586069;
                    border-top: 2px solid #e1e4e8;
                }}
                
                @media print {{
                    body {{
                        background: white;
                        padding: 0;
                    }}
                    .container {{
                        box-shadow: none;
                        padding: 20px;
                    }}
                    .operation-card {{
                        break-inside: avoid;
                    }}
                    .chart-container {{
                        break-inside: avoid;
                    }}
                }}
                
                @media (max-width: 768px) {{
                    .container {{
                        padding: 15px;
                    }}
                    .operation-header {{
                        flex-direction: column;
                        align-items: flex-start;
                    }}
                    .detail-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Отчет анализа данных</h1>
                    <div class="meta">
                        <p>Сгенерировано: {self.timestamp.strftime("%d.%m.%Y %H:%M:%S")}</p>
                        <p>Операций с данными: {len(self.app_state.operations)} | Компонентов визуализации: {len(self.app_state.components)}</p>
                    </div>
                </div>
        """
    
    def _build_summary(self) -> str:
        """Построение сводки отчета"""
        html = '<div class="section">'
        html += '<h2 class="section-title">📋 Общая сводка</h2>'
        html += '<div class="summary-grid">'
        
        html += f'''
            <div class="summary-card">
                <div class="summary-number">{len(self.app_state.operations)}</div>
                <div class="summary-label">Операций с данными</div>
            </div>
            <div class="summary-card">
                <div class="summary-number">{len(self.app_state.components)}</div>
                <div class="summary-label">Компонентов анализа</div>
            </div>
            <div class="summary-card">
                <div class="summary-number">{len(self.app_state.dataframes)}</div>
                <div class="summary-label">Датафреймов</div>
            </div>
        '''
        
        html += '</div></div>'
        return html
    
    def _build_data_operations(self) -> str:
        """Построение раздела операций с данными"""
        if not self.app_state.operations:
            return ''
        
        html = '<div class="section">'
        html += '<h2 class="section-title">🔧 Операции с данными</h2>'
        
        for i, op in enumerate(self.app_state.operations, 1):
            html += self._render_operation(op, i)
        
        html += '</div>'
        return html
    
    def _render_operation(self, operation, number: int) -> str:
        """Отрисовка отдельной операции"""
        op_type = getattr(operation, 'operation_type', 'unknown')
        
        type_labels = {
            'load_csv': '📂 Загрузка CSV',
            'filter': '🔍 Фильтрация',
            'aggregate': '🔄 Агрегация',
            'data_clean': '🧹 Очистка данных',
            'save_csv': '💾 Сохранение CSV'
        }
        
        badge_class = {
            'load_csv': 'badge-load',
            'filter': 'badge-filter',
            'aggregate': 'badge-aggregate',
            'data_clean': 'badge-clean',
            'save_csv': 'badge-save'
        }
        
        html = f'''
        <div class="operation-card">
            <div class="operation-header">
                <div style="display: flex; align-items: center;">
                    <div class="operation-number">{number}</div>
                    <div class="operation-title">{operation.name}</div>
                </div>
                <div class="operation-type-badge {badge_class.get(op_type, '')}">
                    {type_labels.get(op_type, op_type)}
                </div>
            </div>
        '''
        
        # Детали операции
        html += '<div class="detail-grid">'
        
        if op_type == 'load_csv':
            html += self._detail_card('📁 Файл', getattr(operation, 'file_path', 'N/A'))
            html += self._detail_card('📝 Разделитель', repr(getattr(operation, 'sep', ',')))
            
        elif op_type == 'filter':
            html += self._detail_card('📊 Колонка', getattr(operation, 'column', 'N/A'))
            html += self._detail_card('🔍 Тип фильтра', getattr(operation, 'filter_type', 'N/A'))
            filter_value = getattr(operation, 'filter_value', 'N/A')
            if isinstance(filter_value, list):
                filter_value = ', '.join(str(v) for v in filter_value)
            html += self._detail_card('💎 Значение', str(filter_value))
            
        elif op_type == 'aggregate':
            group_by = getattr(operation, 'group_by', [])
            html += self._detail_card('📦 Группировка', ', '.join(group_by))
            agg_funcs = getattr(operation, 'agg_func', {})
            if agg_funcs:
                agg_str = '<br>'.join(f'{col}: {func}' for col, func in agg_funcs.items())
                html += self._detail_card('📐 Агрегации', agg_str)
                
        elif op_type == 'data_clean':
            clean_type = getattr(operation, 'clean_type', 'N/A')
            clean_labels = {
                'dropna': 'Удаление пропусков',
                'fillna': 'Заполнение пропусков',
                'drop_duplicates': 'Удаление дубликатов',
                'replace': 'Замена значений',
                'rename': 'Переименование колонок'
            }
            html += self._detail_card('🧹 Метод', clean_labels.get(clean_type, clean_type))
            columns = getattr(operation, 'columns', [])
            if columns:
                html += self._detail_card('📋 Колонки', ', '.join(columns))
        
        # Исходный датафрейм
        source_id = getattr(operation, 'source_df_id', None)
        if source_id:
            source_name = self.app_state.dataframe_names.get(source_id, 'Неизвестный')
            html += self._detail_card('📂 Исходные данные', source_name)
        
        html += '</div>'
        
        # Превью датафрейма
        df = self.app_state.dataframes.get(operation.id)
        if df is not None:
            html += '<div style="margin-top: 20px;">'
            html += '<h4 style="color: #667eea; margin-bottom: 10px;">📋 Результат (первые 10 строк):</h4>'
            html += '<div class="dataframe-container">'
            html += df.head(10).to_html(classes='dataframe', border=0)
            html += '</div>'
            
            # Статистика
            html += '<div class="stats-grid">'
            html += f'<div class="stat-card"><div class="stat-value">{df.shape[0]:,}</div><div class="stat-label">Строк</div></div>'
            html += f'<div class="stat-card"><div class="stat-value">{df.shape[1]}</div><div class="stat-label">Колонок</div></div>'
            html += f'<div class="stat-card"><div class="stat-value">{df.isna().sum().sum():,}</div><div class="stat-label">Пропусков</div></div>'
            html += f'<div class="stat-card"><div class="stat-value">{df.memory_usage(deep=True).sum() / 1024:.0f}</div><div class="stat-label">КБ памяти</div></div>'
            html += '</div>'
            html += '</div>'
        
        html += '</div>'
        return html
    
    def _detail_card(self, label: str, value: str) -> str:
        """Создание карточки с деталями"""
        return f'''
        <div class="detail-item">
            <div class="detail-label">{label}</div>
            <div class="detail-value">{value}</div>
        </div>
        '''
    
    def _build_components(self) -> str:
        """Построение раздела компонентов"""
        if not self.app_state.components:
            return ''
        
        html = '<div class="section">'
        html += '<h2 class="section-title">📈 Визуализации и анализ</h2>'
        
        for i, comp in enumerate(self.app_state.components, 1):
            html += self._render_component(comp, i)
        
        html += '</div>'
        return html
    
    def _render_component(self, component, number: int) -> str:
        """Отрисовка компонента"""
        comp_name = getattr(component, 'name', f'Компонент {number}')
        
        html = f'<div class="operation-card">'
        html += f'<h3 style="color: #667eea; margin-bottom: 20px;">{number}. {comp_name}</h3>'
        
        # Текстовый компонент
        if hasattr(component, 'text') and component.text:
            html += '<div class="text-component">'
            # Конвертируем Markdown в HTML
            text_html = self._markdown_to_html(component.text)
            html += text_html
            html += '</div>'
        
        # Графический компонент
        elif hasattr(component, 'chart') and component.chart is not None:
            html += '<div class="chart-container">'
            try:
                chart_html = pio.to_html(
                    component.chart, 
                    include_plotlyjs='cdn', 
                    full_html=False,
                    config={'responsive': True}
                )
                html += chart_html
            except Exception as e:
                html += f'<p style="color: red;">Ошибка отображения графика: {e}</p>'
            html += '</div>'
            
            source_df_id = getattr(component, 'source_df_id', None)
            if source_df_id:
                source_name = self.app_state.dataframe_names.get(source_df_id, 'Неизвестный')
                html += f'<p style="color: #586069; font-style: italic; margin-top: 10px;">📂 Источник данных: {source_name}</p>'
        
        html += '</div>'
        return html
    
    def _build_footer(self) -> str:
        """Построение подвала отчета"""
        return f"""
                <div class="footer">
                    <p>📊 Отчет сгенерирован автоматически системой анализа данных</p>
                    <p>{self.timestamp.strftime("%d.%m.%Y %H:%M:%S")}</p>
                    <p style="margin-top: 10px; font-size: 0.9em;">Содержит {len(self.app_state.operations)} операций и {len(self.app_state.components)} компонентов анализа</p>
                </div>
            </div>
        </body>
        </html>
        """