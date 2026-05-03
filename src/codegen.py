import streamlit as st
import nbformat
import pandas as pd
from df_operations import FilterOperation, AggregateOperation, LoadCsvOperation, DataCleanOperation

def generate_notebook_cells():
    """Генерация ячеек Jupyter notebook на основе текущих операций"""
    cells = []
    
    # Добавление ячейки с импортами
    imports = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
    """
    cells.append(nbformat.v4.new_code_cell(imports))
    
    # Получение операций по типам
    load_ops = [op for op in st.session_state.app_state.operations if isinstance(op, LoadCsvOperation)]
    filter_ops = [op for op in st.session_state.app_state.operations if isinstance(op, FilterOperation)]
    clean_ops = [op for op in st.session_state.app_state.operations if isinstance(op, DataCleanOperation)]
    agg_ops = [op for op in st.session_state.app_state.operations if isinstance(op, AggregateOperation)]
    
    # Добавление операций загрузки
    if load_ops:
        cells.append(nbformat.v4.new_markdown_cell("## Загрузка данных"))
        
        for op in load_ops:
            code = f"# Загрузка набора данных: {op.name}\n"
            code += f"df_{op.id[:8]} = pd.read_csv('{op.file_path}', sep='{op.sep}')\n"
            code += f"print(f'Размер загруженного датафрейма: {{df_{op.id[:8]}.shape}}')"
            cells.append(nbformat.v4.new_code_cell(code))
    
    # Добавление операций фильтрации
    if filter_ops:
        cells.append(nbformat.v4.new_markdown_cell("## Фильтрация данных"))
        
        for op in filter_ops:
            source_df = f"df_{op.source_df_id[:8]}"
            result_df = f"df_{op.id[:8]}"
            
            code = f"# Применение фильтра: {op.name}\n"
            
            # Обработка случая с несколькими фильтрами
            if hasattr(op, 'filters') and op.filters:
                code += f"{result_df} = {source_df}.copy()\n"
                
                for i, filter_item in enumerate(op.filters):
                    col = filter_item.get('column')
                    filter_type = filter_item.get('filter_type')
                    filter_value = filter_item.get('filter_value')
                    
                    if filter_type == "equals":
                        code += f"{result_df} = {result_df}[{result_df}['{col}'] == {repr(filter_value)}]\n"
                    elif filter_type == "not_equals":
                        code += f"{result_df} = {result_df}[{result_df}['{col}'] != {repr(filter_value)}]\n"
                    elif filter_type == "contains":
                        code += f"{result_df} = {result_df}[{result_df}['{col}'].astype(str).str.contains({repr(str(filter_value))})]\n"
                    elif filter_type == "greater_than":
                        code += f"{result_df} = {result_df}[{result_df}['{col}'] > {repr(filter_value)}]\n"
                    elif filter_type == "less_than":
                        code += f"{result_df} = {result_df}[{result_df}['{col}'] < {repr(filter_value)}]\n"
            else:
                # Обработка случая с одним фильтром
                if op.filter_type == "equals":
                    code += f"{result_df} = {source_df}[{source_df}['{op.column}'] == {repr(op.filter_value)}]\n"
                elif op.filter_type == "not_equals":
                    code += f"{result_df} = {source_df}[{source_df}['{op.column}'] != {repr(op.filter_value)}]\n"
                elif op.filter_type == "contains":
                    code += f"{result_df} = {source_df}[{source_df}['{op.column}'].astype(str).str.contains({repr(str(op.filter_value))})]\n"
                elif op.filter_type == "greater_than":
                    code += f"{result_df} = {source_df}[{source_df}['{op.column}'] > {repr(op.filter_value)}]\n"
                elif op.filter_type == "less_than":
                    code += f"{result_df} = {source_df}[{source_df}['{op.column}'] < {repr(op.filter_value)}]\n"
            
            code += f"print(f'Размер отфильтрованного датафрейма: {{{{result_df}}}}.shape')"
            cells.append(nbformat.v4.new_code_cell(code))
    
    # Добавление операций очистки данных
    if clean_ops:
        cells.append(nbformat.v4.new_markdown_cell("## Очистка данных"))
        
        for op in clean_ops:
            source_df = f"df_{op.source_df_id[:8]}"
            result_df = f"df_{op.id[:8]}"
            
            code = f"# Применение очистки данных: {op.name}\n"
            code += f"{result_df} = {source_df}.copy()\n"
            
            if hasattr(op, 'drop_columns') and op.drop_columns:
                code += f"{result_df} = {result_df}.drop(columns={repr(op.drop_columns)})\n"
            
            if hasattr(op, 'drop_na') and op.drop_na:
                code += f"{result_df} = {result_df}.dropna()\n"
                
            if hasattr(op, 'rename_columns') and op.rename_columns:
                renames = {old: new for old, new in op.rename_columns.items() if old != new}
                if renames:
                    code += f"{result_df} = {result_df}.rename(columns={repr(renames)})\n"
            
            code += f"print(f'Размер очищенного датафрейма: {{{{result_df}}}}.shape')"
            cells.append(nbformat.v4.new_code_cell(code))
    
    # Добавление операций агрегации
    if agg_ops:
        cells.append(nbformat.v4.new_markdown_cell("## Агрегация данных"))
        
        for op in agg_ops:
            source_df = f"df_{op.source_df_id[:8]}"
            result_df = f"df_{op.id[:8]}"
            
            code = f"# Применение агрегации: {op.name}\n"
            
            code += f"{result_df} = {source_df}.groupby('{op.group_by}').agg({{\n"
            if hasattr(op, 'aggregations'):
                for col, func in op.aggregations.items():
                    code += f"    '{col}': '{func}',\n"
            elif hasattr(op, 'agg_func'):
                for col, func in op.agg_func.items():
                    code += f"    '{col}': '{func}',\n"
            code += "}).reset_index()\n"
            
            code += f"print(f'Размер агрегированного датафрейма: {{{{result_df}}}}.shape')"
            cells.append(nbformat.v4.new_code_cell(code))
    
    # Добавление раздела визуализации
    cells.append(nbformat.v4.new_markdown_cell("## Визуализации"))
    # Получение текущего датафрейма
    current_df_id = st.session_state.app_state.current_df_id
    if current_df_id:
        df_var = f"df_{current_df_id[:8]}"
        
        # Добавление базовых визуализаций
        code = f"# Базовые визуализации для {df_var}\n\n"
        
        # Гистограммы для числовых колонок
        code += "# Гистограммы для числовых колонок\n"
        code += f"numeric_cols = {df_var}.select_dtypes(include=['number']).columns\n"
        code += "for col in numeric_cols[:5]:  # Ограничение первыми 5 числовыми колонками\n"
        code += f"    fig = px.histogram({df_var}, x=col, title=f'Распределение {{{{col}}}}')\n"
        code += "    fig.show()\n\n"
        
        # Столбчатые диаграммы для категориальных колонок
        code += "# Столбчатые диаграммы для категориальных колонок\n"
        code += f"cat_cols = {df_var}.select_dtypes(include=['object', 'category']).columns\n"
        code += "for col in cat_cols[:5]:  # Ограничение первыми 5 категориальными колонками\n"
        code += f"    value_counts = {df_var}[col].value_counts().reset_index()\n"
        code += f"    value_counts.columns = [col, 'count']\n"
        code += f"    fig = px.bar(value_counts, x=col, y='count', title=f'Количество {{{{col}}}}')\n"
        code += "    fig.show()\n"
        
        cells.append(nbformat.v4.new_code_cell(code))
    
    return cells