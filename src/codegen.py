import streamlit as st
import nbformat
import pandas as pd
from df_operations import FilterOperation, AggregateOperation, LoadCsvOperation, SaveCsvOperation, DataCleanOperation


def generate_notebook_cells():
    """Генерация полного Jupyter notebook на основе текущих операций и компонентов"""
    cells = []
    
    # ============================================
    # ЯЧЕЙКА 1: Заголовок и описание
    # ============================================
    title = "# Автоматически сгенерированный анализ данных"
    description = f"Дата создания: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
    cells.append(nbformat.v4.new_markdown_cell(f"{title}\n\n{description}"))
    
    # ============================================
    # ЯЧЕЙКА 2: Импорты
    # ============================================
    imports = """# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Настройка отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

print("✅ Библиотеки загружены")
"""
    cells.append(nbformat.v4.new_code_cell(imports))
    
    # ============================================
    # ЯЧЕЙКА 3: Загрузка данных
    # ============================================
    load_ops = [op for op in st.session_state.app_state.operations if isinstance(op, LoadCsvOperation)]
    
    if load_ops:
        cells.append(nbformat.v4.new_markdown_cell("## 1. Загрузка данных"))
        
        for i, op in enumerate(load_ops):
            df_var = f"df_{op.id[:8]}" if hasattr(op, 'id') else f"df_{i}"
            
            code = f"""# Загрузка: {op.name}
{df_var} = pd.read_csv(
    '{op.file_path}',
    sep='{op.sep}',
    encoding='utf-8'
)

# Информация о загруженных данных
print(f"📂 Файл: {op.file_path}")
print(f"📏 Размер: {{{df_var}.shape[0]}} строк × {{{df_var}.shape[1]}} колонок")
print(f"📊 Колонки: {{', '.join({df_var}.columns.tolist())}}")
print(f"\\nПервые 5 строк:")
display({df_var}.head())
print(f"\\nТипы данных:")
display({df_var}.dtypes.to_frame('Тип'))
"""
            cells.append(nbformat.v4.new_code_cell(code))
    
    # ============================================
    # ЯЧЕЙКА 4: Анализ качества данных
    # ============================================
    if load_ops:
        cells.append(nbformat.v4.new_markdown_cell("## 2. Анализ качества данных"))
        
        df_var = f"df_{load_ops[0].id[:8]}" if hasattr(load_ops[0], 'id') else "df_0"
        
        code = f"""# Анализ качества данных
print("=" * 50)
print("АНАЛИЗ КАЧЕСТВА ДАННЫХ")
print("=" * 50)

# Пропущенные значения
missing = {df_var}.isna().sum()
missing_pct = (missing / len({df_var}) * 100).round(2)
missing_df = pd.DataFrame({{
    'Пропущено': missing,
    'Процент': missing_pct
}})
missing_df = missing_df[missing_df['Пропущено'] > 0].sort_values('Пропущено', ascending=False)

if len(missing_df) > 0:
    print(f"\\n❌ Найдены пропуски в {{len(missing_df)}} колонках:")
    display(missing_df)
else:
    print("\\n✅ Пропущенных значений нет")

# Дубликаты
duplicates = {df_var}.duplicated().sum()
if duplicates > 0:
    print(f"\\n📋 Найдено {{duplicates}} дубликатов строк ({{duplicates/len({df_var})*100:.2f}}%)")
else:
    print("\\n✅ Дубликатов нет")

# Выбросы в числовых колонках
numeric_cols = {df_var}.select_dtypes(include=['number']).columns
print(f"\\n📊 Числовых колонок: {{len(numeric_cols)}}")

for col in numeric_cols:
    Q1 = {df_var}[col].quantile(0.25)
    Q3 = {df_var}[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = {df_var}[({df_var}[col] < Q1 - 1.5*IQR) | ({df_var}[col] > Q3 + 1.5*IQR)]
    if len(outliers) > 0:
        print(f"  📦 {{col}}: {{len(outliers)}} выбросов ({{len(outliers)/len({df_var})*100:.1f}}%)")

# Общая оценка
total_cells = {df_var}.shape[0] * {df_var}.shape[1]
missing_cells = {df_var}.isna().sum().sum()
quality_score = ((total_cells - missing_cells) / total_cells) * 100
print(f"\\n📊 Оценка качества данных: {{quality_score:.1f}}%")
"""
        cells.append(nbformat.v4.new_code_cell(code))
    
    # ============================================
    # ЯЧЕЙКА 5: Статистический анализ
    # ============================================
    if load_ops:
        cells.append(nbformat.v4.new_markdown_cell("## 3. Статистический анализ"))
        
        df_var = f"df_{load_ops[0].id[:8]}" if hasattr(load_ops[0], 'id') else "df_0"
        
        code = f"""# Статистический анализ
print("=" * 50)
print("СТАТИСТИЧЕСКИЙ АНАЛИЗ")
print("=" * 50)

# Описательная статистика для числовых колонок
numeric_cols = {df_var}.select_dtypes(include=['number']).columns
if len(numeric_cols) > 0:
    print("\\n📈 Описательная статистика числовых колонок:")
    display({df_var}[numeric_cols].describe())
    
    # Асимметрия и эксцесс
    stats_df = pd.DataFrame({{
        'Асимметрия': {df_var}[numeric_cols].skew(),
        'Эксцесс': {df_var}[numeric_cols].kurtosis()
    }})
    print("\\n📊 Асимметрия и эксцесс:")
    display(stats_df)

# Категориальные колонки
cat_cols = {df_var}.select_dtypes(include=['object', 'category']).columns
for col in cat_cols[:10]:
    value_counts = {df_var}[col].value_counts()
    print(f"\\n📊 {{col}}: {{len(value_counts)}} уникальных значений")
    display(value_counts.head(10).to_frame('Количество'))

# Корреляционный анализ
if len(numeric_cols) >= 2:
    print("\\n🔥 Корреляционная матрица:")
    corr_matrix = {df_var}[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Матрица корреляций",
        color_continuous_scale='RdBu_r'
    )
    fig.show()
    
    # Сильные корреляции
    print("\\nСильные корреляции (|r| > 0.5):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.5:
                direction = "↑" if val > 0 else "↓"
                print(f"  {{corr_matrix.columns[i]}} {{direction}} {{corr_matrix.columns[j]}}: {{val:.3f}}")
"""
        cells.append(nbformat.v4.new_code_cell(code))
    
    # ============================================
    # ЯЧЕЙКА 6: Операции фильтрации
    # ============================================
    filter_ops = [op for op in st.session_state.app_state.operations if isinstance(op, FilterOperation)]
    
    if filter_ops:
        cells.append(nbformat.v4.new_markdown_cell("## 4. Фильтрация данных"))
        
        for op in filter_ops:
            source_df = get_df_var(op, load_ops)
            result_df = f"df_{op.id[:8]}"
            
            code = f"# Фильтрация: {op.name}\n"
            
            # Один фильтр
            if hasattr(op, 'column'):
                col = op.column
                filter_type = op.filter_type
                filter_value = op.filter_value
                
                code += f"{result_df} = {source_df}.copy()\n"
                
                if filter_type == "equals":
                    code += f"{result_df} = {result_df}[{result_df}['{col}'] == {repr(filter_value)}]\n"
                elif filter_type == "not_equals":
                    code += f"{result_df} = {result_df}[{result_df}['{col}'] != {repr(filter_value)}]\n"
                elif filter_type == "greater_than":
                    code += f"{result_df} = {result_df}[{result_df}['{col}'] > {filter_value}]\n"
                elif filter_type == "less_than":
                    code += f"{result_df} = {result_df}[{result_df}['{col}'] < {filter_value}]\n"
                elif filter_type == "between":
                    if isinstance(filter_value, list) and len(filter_value) == 2:
                        code += f"{result_df} = {result_df}[({result_df}['{col}'] >= {filter_value[0]}) & ({result_df}['{col}'] <= {filter_value[1]})]\n"
                elif filter_type == "contains":
                    code += f"{result_df} = {result_df}[{result_df}['{col}'].astype(str).str.contains({repr(str(filter_value))})]\n"
                elif filter_type == "startswith":
                    code += f"{result_df} = {result_df}[{result_df}['{col}'].astype(str).str.startswith({repr(str(filter_value))})]\n"
                elif filter_type == "endswith":
                    code += f"{result_df} = {result_df}[{result_df}['{col}'].astype(str).str.endswith({repr(str(filter_value))})]\n"
                elif filter_type == "isin":
                    if isinstance(filter_value, list):
                        code += f"{result_df} = {result_df}[{result_df}['{col}'].isin({filter_value})]\n"
                elif filter_type == "notna":
                    code += f"{result_df} = {result_df}[{result_df}['{col}'].notna()]\n"
                elif filter_type == "isna":
                    code += f"{result_df} = {result_df}[{result_df}['{col}'].isna()]\n"
            
            # Несколько фильтров
            elif hasattr(op, 'filters') and op.filters:
                code += f"{result_df} = {source_df}.copy()\n"
                for f in op.filters:
                    col = f.get('column')
                    ft = f.get('filter_type')
                    fv = f.get('filter_value')
                    
                    if ft == "equals":
                        code += f"{result_df} = {result_df}[{result_df}['{col}'] == {repr(fv)}]\n"
                    elif ft == "not_equals":
                        code += f"{result_df} = {result_df}[{result_df}['{col}'] != {repr(fv)}]\n"
                    elif ft == "greater_than":
                        code += f"{result_df} = {result_df}[{result_df}['{col}'] > {fv}]\n"
                    elif ft == "less_than":
                        code += f"{result_df} = {result_df}[{result_df}['{col}'] < {fv}]\n"
                    elif ft == "contains":
                        code += f"{result_df} = {result_df}[{result_df}['{col}'].astype(str).str.contains({repr(str(fv))})]\n"
                    elif ft == "isin":
                        if isinstance(fv, list):
                            code += f"{result_df} = {result_df}[{result_df}['{col}'].isin({fv})]\n"
            
            code += f"""print(f"🔍 Фильтр: {op.name}")
print(f"📏 Размер до: {{{source_df}.shape[0]}} строк")
print(f"📏 Размер после: {{{result_df}.shape[0]}} строк")
print(f"📉 Удалено: {{{source_df}.shape[0] - {result_df}.shape[0]}} строк ({{({source_df}.shape[0] - {result_df}.shape[0])/{source_df}.shape[0]*100:.1f}}%)\\n")
"""
            cells.append(nbformat.v4.new_code_cell(code))
    
    # ============================================
    # ЯЧЕЙКА 7: Операции очистки данных
    # ============================================
    clean_ops = [op for op in st.session_state.app_state.operations if isinstance(op, DataCleanOperation)]
    
    if clean_ops:
        cells.append(nbformat.v4.new_markdown_cell("## 5. Очистка данных"))
        
        for op in clean_ops:
            source_df = get_df_var(op, load_ops)
            result_df = f"df_{op.id[:8]}"
            
            code = f"# Очистка данных: {op.name}\n"
            code += f"{result_df} = {source_df}.copy()\n"
            
            if hasattr(op, 'clean_type'):
                clean_type = op.clean_type
                columns = getattr(op, 'columns', [])
                
                if clean_type == "dropna":
                    if columns:
                        code += f"{result_df} = {result_df}.dropna(subset={columns})\n"
                    else:
                        code += f"{result_df} = {result_df}.dropna()\n"
                
                elif clean_type == "fillna":
                    fill_value = getattr(op, 'fill_value', 'mean')
                    if columns:
                        for col in columns:
                            if fill_value == "mean":
                                code += f"{result_df}['{col}'] = {result_df}['{col}'].fillna({result_df}['{col}'].mean())\n"
                            elif fill_value == "median":
                                code += f"{result_df}['{col}'] = {result_df}['{col}'].fillna({result_df}['{col}'].median())\n"
                            elif fill_value == "mode":
                                code += f"{result_df}['{col}'] = {result_df}['{col}'].fillna({result_df}['{col}'].mode()[0])\n"
                            elif fill_value == "ffill":
                                code += f"{result_df}['{col}'] = {result_df}['{col}'].fillna(method='ffill')\n"
                            elif fill_value == "bfill":
                                code += f"{result_df}['{col}'] = {result_df}['{col}'].fillna(method='bfill')\n"
                            else:
                                code += f"{result_df}['{col}'] = {result_df}['{col}'].fillna({repr(fill_value)})\n"
                    else:
                        if fill_value == "mean":
                            code += f"{result_df} = {result_df}.fillna({result_df}.mean())\n"
                        elif fill_value == "median":
                            code += f"{result_df} = {result_df}.fillna({result_df}.median())\n"
                        else:
                            code += f"{result_df} = {result_df}.fillna({repr(fill_value)})\n"
                
                elif clean_type == "drop_duplicates":
                    if columns:
                        code += f"{result_df} = {result_df}.drop_duplicates(subset={columns})\n"
                    else:
                        code += f"{result_df} = {result_df}.drop_duplicates()\n"
                
                elif clean_type == "replace":
                    replace_values = getattr(op, 'replace_values', {})
                    if replace_values:
                        code += f"{result_df} = {result_df}.replace({replace_values})\n"
                
                elif clean_type == "rename":
                    new_names = getattr(op, 'new_column_names', {})
                    if new_names:
                        code += f"{result_df} = {result_df}.rename(columns={new_names})\n"
            
            code += f"""print(f"🧹 Очистка: {op.name}")
print(f"📏 Размер: {{{result_df}.shape[0]}} строк × {{{result_df}.shape[1]}} колонок")
print(f"❌ Осталось пропусков: {{{result_df}.isna().sum().sum()}}\\n")
"""
            cells.append(nbformat.v4.new_code_cell(code))
    
    # ============================================
    # ЯЧЕЙКА 8: Операции агрегации
    # ============================================
    agg_ops = [op for op in st.session_state.app_state.operations if isinstance(op, AggregateOperation)]
    
    if agg_ops:
        cells.append(nbformat.v4.new_markdown_cell("## 6. Агрегация данных"))
        
        for op in agg_ops:
            source_df = get_df_var(op, load_ops)
            result_df = f"df_{op.id[:8]}"
            
            code = f"# Агрегация: {op.name}\n"
            
            if hasattr(op, 'group_by') and hasattr(op, 'agg_func'):
                group_cols = op.group_by if isinstance(op.group_by, list) else [op.group_by]
                agg_funcs = op.agg_func
                
                if isinstance(agg_funcs, dict):
                    code += f"{result_df} = {source_df}.groupby({group_cols}).agg(\n"
                    for col, func in agg_funcs.items():
                        code += f"    {col}=('{col}', '{func}'),\n"
                    code += ").reset_index()\n"
                else:
                    code += f"{result_df} = {source_df}.groupby({group_cols}).agg({agg_funcs}).reset_index()\n"
            
            code += f"""print(f"🔄 Агрегация: {op.name}")
print(f"📊 Группировка по: {group_cols}")
print(f"📏 Размер: {{{result_df}.shape[0]}} строк × {{{result_df}.shape[1]}} колонок")
display({result_df}.head())\n
"""
            cells.append(nbformat.v4.new_code_cell(code))
    
    # ============================================
    # ЯЧЕЙКА 9: Визуализации из компонентов
    # ============================================
    chart_components = [
        comp for comp in st.session_state.app_state.components 
        if hasattr(comp, 'chart')
    ]
    
    if chart_components:
        cells.append(nbformat.v4.new_markdown_cell("## 7. Визуализации"))
        
        for i, comp in enumerate(chart_components):
            chart_name = getattr(comp, 'name', f'График {i+1}')
            source_df_id = getattr(comp, 'source_df_id', None)
            chart_type = getattr(comp, 'chart_type', None)
            chart_params = getattr(comp, 'chart_params', {})
            
            code = f"# Визуализация: {chart_name}\n"
            
            if source_df_id:
                df_var = get_df_var_by_id(source_df_id, load_ops)
                code += f"# Данные: {df_var}\n"
            
            # Генерируем код в зависимости от типа графика
            if chart_type == "Линейный график":
                code += f"""fig = px.line(
    {df_var},
    x='{chart_params.get('x_column', '')}',
    y='{chart_params.get('y_column', '')}',
    title='{chart_params.get('title', chart_name)}'
)
"""
                if chart_params.get('hue_column'):
                    code += f"fig = px.line({df_var}, x='{chart_params['x_column']}', y='{chart_params['y_column']}', color='{chart_params['hue_column']}', title='{chart_params.get('title', '')}')\n"
            
            elif chart_type == "Столбчатая диаграмма":
                code += f"""fig = px.bar(
    {df_var},
    x='{chart_params.get('x_column', '')}',
    y='{chart_params.get('y_column', '')}',
    title='{chart_params.get('title', chart_name)}'
)
"""
            
            elif chart_type == "Гистограмма":
                code += f"""fig = px.histogram(
    {df_var},
    x='{chart_params.get('x_column', '')}',
    nbins={chart_params.get('bins', 10)},
    title='{chart_params.get('title', chart_name)}'
)
"""
                if chart_params.get('color'):
                    code += f"fig = px.histogram({df_var}, x='{chart_params['x_column']}', color='{chart_params['color']}', nbins={chart_params.get('bins', 10)}, title='{chart_params.get('title', '')}')\n"
            
            elif chart_type == "Диаграмма рассеяния":
                code += f"""fig = px.scatter(
    {df_var},
    x='{chart_params.get('x_column', '')}',
    y='{chart_params.get('y_column', '')}',
    title='{chart_params.get('title', chart_name)}'
)
"""
                if chart_params.get('hue_column'):
                    code += f"fig = px.scatter({df_var}, x='{chart_params['x_column']}', y='{chart_params['y_column']}', color='{chart_params['hue_column']}', title='{chart_params.get('title', '')}')\n"
            
            elif chart_type == "Круговая диаграмма":
                code += f"""fig = px.pie(
    {df_var},
    names='{chart_params.get('x_column', '')}',
    values='{chart_params.get('values_column', '')}',
    title='{chart_params.get('title', chart_name)}'
)
"""
            
            elif chart_type == "Ящик с усами":
                code += f"""fig = px.box(
    {df_var},
    x='{chart_params.get('x_column', '')}',
    y='{chart_params.get('y_column', '')}',
    title='{chart_params.get('title', chart_name)}'
)
"""
            
            elif chart_type == "Тепловая карта":
                code += f"""# Тепловая карта
fig = px.imshow(
    {df_var}.select_dtypes(include=['number']).corr(),
    text_auto=True,
    title='{chart_params.get('title', chart_name)}',
    color_continuous_scale='RdBu_r'
)
"""
            
            else:
                # Общий случай
                code += f"# График: {chart_name}\n"
                code += f"# Используйте plotly.express или plotly.graph_objects для создания графика\n"
                code += f"fig = go.Figure()  # Замените на нужный тип графика\n"
            
            code += """
fig.update_layout(
    template='plotly_white',
    width=800,
    height=500
)
fig.show()
"""
            cells.append(nbformat.v4.new_code_cell(code))
    
    # ============================================
    # ЯЧЕЙКА 10: Сохранение результатов
    # ============================================
    save_ops = [op for op in st.session_state.app_state.operations if isinstance(op, SaveCsvOperation)]
    
    if save_ops:
        cells.append(nbformat.v4.new_markdown_cell("## 8. Сохранение результатов"))
        
        for op in save_ops:
            source_df = get_df_var(op, load_ops)
            
            code = f"""# Сохранение: {op.name}
{source_df}.to_csv(
    '{op.file_path}',
    sep='{op.sep}',
    index=False,
    encoding='utf-8'
)
print(f"💾 Данные сохранены в: {op.file_path}")
print(f"📏 Размер: {{{source_df}.shape[0]}} строк × {{{source_df}.shape[1]}} колонок")
"""
            cells.append(nbformat.v4.new_code_cell(code))
    
    # ============================================
    # ЯЧЕЙКА 11: Итоговый датафрейм
    # ============================================
    cells.append(nbformat.v4.new_markdown_cell("## 9. Итоговый результат"))
    
    current_df_id = st.session_state.app_state.current_df_id
    if current_df_id:
        df_var = get_df_var_by_id(current_df_id, load_ops)
        
        code = f"""# Итоговый датафрейм после всех операций
final_df = {df_var}.copy()

print("=" * 50)
print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
print("=" * 50)
print(f"📏 Итоговый размер: {{final_df.shape[0]}} строк × {{final_df.shape[1]}} колонок")
print(f"📊 Колонки: {{', '.join(final_df.columns.tolist())}}")
print(f"\\nПервые 10 строк:")
display(final_df.head(10))
print(f"\\nОписательная статистика:")
display(final_df.describe(include='all'))
print(f"\\nПропущенные значения: {{final_df.isna().sum().sum()}}")

# Финальная визуализация - сводка по всем числовым колонкам
numeric_cols = final_df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 0:
    fig = make_subplots(
        rows=min(3, len(numeric_cols)),
        cols=1,
        subplot_titles=[f'Распределение: {{col}}' for col in numeric_cols[:3]]
    )
    
    for i, col in enumerate(numeric_cols[:3], 1):
        fig.add_trace(
            go.Histogram(x=final_df[col], name=col),
            row=i, col=1
        )
    
    fig.update_layout(height=300*min(3, len(numeric_cols)), title_text="Итоговые распределения")
    fig.show()

print("\\n✅ Анализ завершен!")
"""
        cells.append(nbformat.v4.new_code_cell(code))
    
    return cells


def get_df_var(operation, load_ops):
    """Получить имя переменной датафрейма для операции"""
    if hasattr(operation, 'source_df_id') and operation.source_df_id:
        # Проверяем, не является ли source_df_id результатом другой операции
        all_ops = st.session_state.app_state.operations
        for op in all_ops:
            if hasattr(op, 'id') and op.id == operation.source_df_id:
                return f"df_{op.id[:8]}"
        return f"df_{operation.source_df_id[:8]}"
    
    # По умолчанию - первый загруженный датафрейм
    if load_ops and hasattr(load_ops[0], 'id'):
        return f"df_{load_ops[0].id[:8]}"
    return "df"


def get_df_var_by_id(df_id, load_ops):
    """Получить имя переменной датафрейма по ID"""
    # Ищем операцию с таким ID
    all_ops = st.session_state.app_state.operations
    for op in all_ops:
        if hasattr(op, 'id') and op.id == df_id:
            return f"df_{op.id[:8]}"
    
    # Если не нашли, используем первый загруженный
    if load_ops and hasattr(load_ops[0], 'id'):
        return f"df_{load_ops[0].id[:8]}"
    return "df"