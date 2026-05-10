"""
Модуль с шаблонами анализа данных.
Простые и эффективные шаблоны для быстрого анализа.
"""

import streamlit as st
import pandas as pd
import numpy as np
from components import TextComponent, ChartComponent
from charts import (
    LineChart, BarChart, Histogram, ScatterChart, 
    PieChart, BoxPlot, Heatmap, AreaChart
)
from datetime import datetime


def get_numeric_columns(df):
    """Получить список числовых колонок"""
    return df.select_dtypes(include=['number']).columns.tolist()


def get_categorical_columns(df):
    """Получить список категориальных колонок"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def get_date_columns(df):
    """Получить список колонок с датами"""
    date_cols = []
    for col in df.columns:
        if 'date' in col.lower() or 'дата' in col.lower() or 'time' in col.lower():
            date_cols.append(col)
    if not date_cols:
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        date_cols.extend(datetime_cols)
    return date_cols


def find_column_by_keywords(df, keywords):
    """Найти колонку по ключевым словам"""
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in keywords):
            return col
    return None


def calculate_quality_score(df):
    """Рассчитать оценку качества данных"""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    if total_cells == 0:
        return 100
    
    score = 100
    score -= (missing_cells / total_cells) * 40
    score -= (duplicate_rows / len(df)) * 20 if len(df) > 0 else 0
    
    return max(0, min(100, score))


def check_data_quality(df):
    """Проверить качество данных и вернуть список проблем"""
    issues = []
    
    # Пропуски
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            pct = count / len(df) * 100
            if pct > 50:
                issues.append(f"🔴 {col}: {count} пропусков ({pct:.0f}%)")
            elif pct > 10:
                issues.append(f"🟡 {col}: {count} пропусков ({pct:.0f}%)")
    
    # Дубликаты
    dupes = df.duplicated().sum()
    if dupes > 0:
        issues.append(f"📋 Найдено {dupes} дубликатов строк")
    
    # Константные колонки
    for col in df.columns:
        if df[col].nunique() <= 1:
            issues.append(f"⚠️ Колонка '{col}' имеет только одно значение")
    
    # Выбросы в числовых колонках
    for col in get_numeric_columns(df):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        if len(outliers) > len(df) * 0.1:
            issues.append(f"📊 {col}: {len(outliers)} выбросов ({len(outliers)/len(df)*100:.0f}%)")
    
    return issues


def template_quick_overview(df, app_state):
    """Шаблон: Быстрый обзор данных"""
    components = []
    
    # 1. Карточка с основной информацией
    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)
    date_cols = get_date_columns(df)
    
    text = f"""### 📊 Обзор данных
    
| Показатель | Значение |
|------------|----------|
| Размер | {df.shape[0]:,} строк × {df.shape[1]} колонок |
| Числовых колонок | {len(num_cols)} |
| Категориальных колонок | {len(cat_cols)} |
| Пропусков | {df.isna().sum().sum():,} |
| Дубликатов | {df.duplicated().sum():,} |
"""
    if date_cols:
        text += f"| Колонки с датами | {', '.join(date_cols[:3])} |\n"
    
    text += f"\n**Колонки:** {', '.join(df.columns.tolist())}"
    
    comp = TextComponent(text=text)
    comp.name = "📊 Обзор данных"
    components.append(comp)
    
    # 2. Статистика числовых колонок
    if num_cols:
        text = "### 📈 Числовая статистика\n\n"
        stats = df[num_cols].describe()
        
        for col in num_cols[:5]:
            text += f"**{col}:** "
            text += f"среднее={stats[col]['mean']:.2f}, "
            text += f"медиана={df[col].median():.2f}, "
            text += f"мин={stats[col]['min']:.2f}, "
            text += f"макс={stats[col]['max']:.2f}\n"
        
        comp = TextComponent(text=text)
        comp.name = "📈 Числовая статистика"
        components.append(comp)
    
    # 3. Гистограммы для числовых колонок
    for col in num_cols[:4]:
        try:
            chart = Histogram(
                df=df,
                x_column=col,
                title=f"Распределение: {col}",
                bins=30
            )
            comp = ChartComponent(chart=chart.plot())
            comp.name = f"📊 {col}"
            components.append(comp)
        except:
            pass
    
    # 4. Топ категорий для категориальных колонок
    for col in cat_cols[:2]:
        try:
            value_counts = df[col].value_counts().head(10)
            if len(value_counts) > 0:
                chart = BarChart(
                    df=value_counts.reset_index(),
                    x_column=col,
                    y_column='count',
                    title=f"Топ-10: {col}"
                )
                comp = ChartComponent(chart=chart.plot())
                comp.name = f"📊 {col}"
                components.append(comp)
        except:
            pass
    
    # 5. Корреляции если есть 2+ числовых колонки
    if len(num_cols) >= 2:
        try:
            corr = df[num_cols[:min(8, len(num_cols))]].corr()
            chart = Heatmap(
                df=corr,
                title="Корреляции числовых переменных"
            )
            comp = ChartComponent(chart=chart.plot())
            comp.name = "🔥 Корреляции"
            components.append(comp)
        except:
            pass
    
    # 6. Проблемы с качеством
    issues = check_data_quality(df)
    if issues:
        text = "### ⚠️ Обнаруженные проблемы\n\n"
        for issue in issues:
            text += f"- {issue}\n"
        comp = TextComponent(text=text)
        comp.name = "⚠️ Проблемы"
        components.append(comp)
    
    return components


def template_sales_analysis(df, app_state):
    """Шаблон: Анализ продаж"""
    components = []
    
    # Ищем подходящие колонки
    amount_col = find_column_by_keywords(df, ['сумма', 'amount', 'выручка', 'revenue', 'продажи', 'sales', 'цена', 'price', 'total', 'итого'])
    date_col = find_column_by_keywords(df, ['дата', 'date', 'месяц', 'month', 'год', 'year', 'период', 'period'])
    product_col = find_column_by_keywords(df, ['товар', 'product', 'название', 'name', 'продукт', 'item'])
    qty_col = find_column_by_keywords(df, ['количество', 'qty', 'quantity', 'штук', 'кол-во', 'count'])
    
    # Если не нашли по ключевым словам, берем первую подходящую
    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)
    
    if not amount_col and num_cols:
        amount_col = num_cols[0]
    if not product_col and cat_cols:
        product_col = cat_cols[0]
    if not date_col:
        date_cols = get_date_columns(df)
        if date_cols:
            date_col = date_cols[0]
    
    if not amount_col:
        comp = TextComponent(text="### ⚠️ Не найдена колонка с суммами\n\nДобавьте колонку с суммами продаж.")
        comp.name = "⚠️ Ошибка"
        components.append(comp)
        return components
    
    # 1. Общая сумма
    total = df[amount_col].sum()
    avg = df[amount_col].mean()
    
    text = f"""### 💰 Сводка по продажам
    
- **Общая сумма:** {total:,.2f}
- **Средняя сумма:** {avg:,.2f}
- **Всего записей:** {len(df):,}
"""
    comp = TextComponent(text=text)
    comp.name = "💰 Сводка"
    components.append(comp)
    
    # 2. Гистограмма сумм
    try:
        chart = Histogram(
            df=df,
            x_column=amount_col,
            title=f"Распределение сумм",
            bins=30
        )
        comp = ChartComponent(chart=chart.plot())
        comp.name = "📊 Распределение сумм"
        components.append(comp)
    except:
        pass
    
    # 3. Топ продуктов
    if product_col:
        try:
            top = df.groupby(product_col)[amount_col].sum().sort_values(ascending=False).head(10)
            text = f"### 🏆 Топ-10 по {product_col}\n\n"
            for i, (name, val) in enumerate(top.items(), 1):
                text += f"{i}. **{name}**: {val:,.2f}\n"
            
            comp = TextComponent(text=text)
            comp.name = "🏆 Топ-10"
            components.append(comp)
            
            # График
            top_df = top.reset_index()
            chart = BarChart(
                df=top_df,
                x_column=product_col,
                y_column=amount_col,
                title=f"Топ-10 по {product_col}"
            )
            comp = ChartComponent(chart=chart.plot())
            comp.name = "📊 Топ-10"
            components.append(comp)
        except:
            pass
    
    # 4. Динамика по датам
    if date_col:
        try:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            df_temp = df_temp.sort_values(date_col)
            
            # По дням
            daily = df_temp.groupby(df_temp[date_col].dt.date)[amount_col].sum().reset_index()
            daily.columns = ['date', amount_col]
            
            chart = LineChart(
                df=daily,
                x_column='date',
                y_column=amount_col,
                title="Динамика по дням"
            )
            comp = ChartComponent(chart=chart.plot())
            comp.name = "📈 Динамика"
            components.append(comp)
        except:
            pass
    
    # 5. Круговая диаграмма топ-5
    if product_col:
        try:
            top5 = df.groupby(product_col)[amount_col].sum().sort_values(ascending=False).head(5)
            top5_df = top5.reset_index()
            
            chart = PieChart(
                df=top5_df,
                x_column=product_col,
                values_column=amount_col,
                title="Топ-5 продуктов"
            )
            comp = ChartComponent(chart=chart.plot())
            comp.name = "🥧 Топ-5"
            components.append(comp)
        except:
            pass
    
    # 6. BoxPlot для выбросов
    try:
        chart = BoxPlot(
            df=df,
            y_column=amount_col,
            title=f"Выбросы по суммам"
        )
        comp = ChartComponent(chart=chart.plot())
        comp.name = "📦 Выбросы"
        components.append(comp)
    except:
        pass
    
    return components


def template_customer_analysis(df, app_state):
    """Шаблон: Анализ клиентов"""
    components = []
    
    cat_cols = get_categorical_columns(df)
    num_cols = get_numeric_columns(df)
    
    if not cat_cols and not num_cols:
        comp = TextComponent(text="### ⚠️ Нет данных для анализа\n\nДобавьте категориальные или числовые колонки.")
        comp.name = "⚠️ Ошибка"
        components.append(comp)
        return components
    
    # 1. Распределение категорий
    for col in cat_cols[:3]:
        try:
            value_counts = df[col].value_counts().head(10)
            if len(value_counts) > 1:
                chart = BarChart(
                    df=value_counts.reset_index(),
                    x_column=col,
                    y_column='count',
                    title=f"Распределение: {col}"
                )
                comp = ChartComponent(chart=chart.plot())
                comp.name = f"📊 {col}"
                components.append(comp)
        except:
            pass
    
    # 2. Числовые метрики
    for col in num_cols[:3]:
        try:
            stats = df[col].describe()
            text = f"### 📈 {col}\n\n"
            text += f"- Среднее: {stats['mean']:,.2f}\n"
            text += f"- Медиана: {stats['50%']:,.2f}\n"
            text += f"- Мин: {stats['min']:,.2f}\n"
            text += f"- Макс: {stats['max']:,.2f}\n"
            
            comp = TextComponent(text=text)
            comp.name = f"📈 {col}"
            components.append(comp)
            
            # Гистограмма
            chart = Histogram(
                df=df,
                x_column=col,
                title=f"Распределение: {col}",
                bins=25
            )
            comp = ChartComponent(chart=chart.plot())
            comp.name = f"📊 {col}"
            components.append(comp)
        except:
            pass
    
    # 3. Группировка по категории
    if cat_cols and num_cols:
        for cat_col in cat_cols[:1]:
            for num_col in num_cols[:1]:
                try:
                    if df[cat_col].nunique() <= 15:
                        grouped = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
                        
                        chart = BarChart(
                            df=grouped.reset_index(),
                            x_column=cat_col,
                            y_column=num_col,
                            title=f"Средний {num_col} по {cat_col}"
                        )
                        comp = ChartComponent(chart=chart.plot())
                        comp.name = f"📊 Группировка"
                        components.append(comp)
                except:
                    pass
                break
            break
    
    # 4. Корреляции
    if len(num_cols) >= 2:
        try:
            corr = df[num_cols[:6]].corr()
            chart = Heatmap(
                df=corr,
                title="Корреляции метрик"
            )
            comp = ChartComponent(chart=chart.plot())
            comp.name = "🔥 Корреляции"
            components.append(comp)
        except:
            pass
    
    return components


def template_time_series(df, app_state):
    """Шаблон: Анализ временных рядов"""
    components = []
    
    # Поиск даты
    date_col = find_column_by_keywords(df, ['дата', 'date', 'время', 'time', 'месяц', 'month', 'период'])
    if not date_col:
        date_cols = get_date_columns(df)
        if date_cols:
            date_col = date_cols[0]
    
    if not date_col:
        # Пробуем найти колонку, которую можно преобразовать в дату
        for col in df.select_dtypes(include=['object']).columns:
            try:
                sample = df[col].dropna().head(5)
                if len(sample) > 0:
                    pd.to_datetime(sample)
                    date_col = col
                    break
            except:
                pass
    
    if not date_col:
        comp = TextComponent(text="### ⚠️ Не найдена колонка с датами\n\nДобавьте колонку с датами для анализа временных рядов.")
        comp.name = "⚠️ Ошибка"
        components.append(comp)
        return components
    
    try:
        df_temp = df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col])
        df_temp = df_temp.sort_values(date_col)
    except:
        comp = TextComponent(text=f"### ⚠️ Не удалось преобразовать '{date_col}' в дату")
        comp.name = "⚠️ Ошибка"
        components.append(comp)
        return components
    
    num_cols = get_numeric_columns(df_temp)
    
    # 1. Информация
    min_date = df_temp[date_col].min()
    max_date = df_temp[date_col].max()
    
    text = f"""### 📅 Временной ряд
    
**Период:** {min_date.strftime('%Y-%m-%d')} — {max_date.strftime('%Y-%m-%d')}
**Точек:** {len(df_temp):,}
"""
    comp = TextComponent(text=text)
    comp.name = "📅 Информация"
    components.append(comp)
    
    # 2. Распределение по месяцам
    try:
        df_temp['month'] = df_temp[date_col].dt.to_period('M').astype(str)
        month_counts = df_temp.groupby('month').size().reset_index(name='count')
        
        chart = BarChart(
            df=month_counts,
            x_column='month',
            y_column='count',
            title="Количество записей по месяцам"
        )
        comp = ChartComponent(chart=chart.plot())
        comp.name = "📅 По месяцам"
        components.append(comp)
    except:
        pass
    
    # 3. Динамика числовых показателей
    for col in num_cols[:2]:
        try:
            daily = df_temp.groupby(df_temp[date_col].dt.date)[col].sum().reset_index()
            daily.columns = ['date', col]
            
            chart = LineChart(
                df=daily,
                x_column='date',
                y_column=col,
                title=f"Динамика: {col}"
            )
            comp = ChartComponent(chart=chart.plot())
            comp.name = f"📈 {col}"
            components.append(comp)
        except:
            pass
    
    # 4. По дням недели
    try:
        df_temp['weekday'] = df_temp[date_col].dt.day_name()
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        if num_cols:
            col = num_cols[0]
            weekday_avg = df_temp.groupby('weekday')[col].mean()
            weekday_avg = weekday_avg[[w for w in order if w in weekday_avg.index]]
            
            if len(weekday_avg) > 1:
                chart = BarChart(
                    df=weekday_avg.reset_index(),
                    x_column='weekday',
                    y_column=col,
                    title=f"Среднее по дням недели: {col}"
                )
                comp = ChartComponent(chart=chart.plot())
                comp.name = "📅 Дни недели"
                components.append(comp)
    except:
        pass
    
    return components


def template_correlation_analysis(df, app_state):
    """Шаблон: Корреляционный анализ"""
    components = []
    
    num_cols = get_numeric_columns(df)
    
    if len(num_cols) < 2:
        comp = TextComponent(text="### ⚠️ Нужно минимум 2 числовые колонки\n\nДобавьте больше числовых данных.")
        comp.name = "⚠️ Ошибка"
        components.append(comp)
        return components
    
    # 1. Матрица корреляций
    try:
        corr = df[num_cols].corr()
        chart = Heatmap(
            df=corr,
            title="Матрица корреляций"
        )
        comp = ChartComponent(chart=chart.plot())
        comp.name = "🔥 Корреляции"
        components.append(comp)
    except:
        pass
    
    # 2. Сильные корреляции
    try:
        strong = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.5:
                    strong.append({
                        'col1': corr.columns[i],
                        'col2': corr.columns[j],
                        'val': val
                    })
        
        if strong:
            text = "### 🔗 Обнаруженные связи\n\n"
            for item in sorted(strong, key=lambda x: abs(x['val']), reverse=True):
                direction = "↑" if item['val'] > 0 else "↓"
                text += f"- {item['col1']} {direction} {item['col2']}: {item['val']:.2f}\n"
            
            comp = TextComponent(text=text)
            comp.name = "🔗 Связи"
            components.append(comp)
        else:
            comp = TextComponent(text="### ℹ️ Нет сильных корреляций\n\nВсе |r| < 0.5")
            comp.name = "ℹ️ Корреляции"
            components.append(comp)
    except:
        pass
    
    # 3. Диаграммы рассеяния для сильных связей
    if strong:
        for item in strong[:3]:
            try:
                chart = ScatterChart(
                    df=df,
                    x_column=item['col1'],
                    y_column=item['col2'],
                    title=f"{item['col1']} vs {item['col2']} (r={item['val']:.2f})"
                )
                comp = ChartComponent(chart=chart.plot())
                comp.name = f"🎯 {item['col1']} vs {item['col2']}"
                components.append(comp)
            except:
                pass
    
    return components


def template_data_quality(df, app_state):
    """Шаблон: Проверка качества данных"""
    components = []
    
    score = calculate_quality_score(df)
    issues = check_data_quality(df)
    
    # 1. Оценка
    emoji = "🟢" if score >= 80 else ("🟡" if score >= 50 else "🔴")
    
    text = f"""### {emoji} Качество данных: {score:.0f}%
    
- 📏 Размер: {df.shape[0]:,} × {df.shape[1]} колонок
- ❌ Пропусков: {df.isna().sum().sum():,}
- 📋 Дубликатов: {df.duplicated().sum():,}
"""
    comp = TextComponent(text=text)
    comp.name = "📊 Оценка качества"
    components.append(comp)
    
    # 2. Проблемы
    if issues:
        text = "### ⚠️ Проблемы\n\n"
        for issue in issues:
            text += f"- {issue}\n"
        comp = TextComponent(text=text)
        comp.name = "⚠️ Проблемы"
        components.append(comp)
    else:
        comp = TextComponent(text="### ✅ Данные в хорошем состоянии\n\nЯвных проблем не обнаружено.")
        comp.name = "✅ Качество"
        components.append(comp)
    
    # 3. Пропуски по колонкам
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) > 0:
        text = "### ❌ Пропуски по колонкам\n\n"
        for col, count in missing.items():
            pct = count / len(df) * 100
            text += f"- {col}: {count} ({pct:.1f}%)\n"
        
        comp = TextComponent(text=text)
        comp.name = "❌ Пропуски"
        components.append(comp)
    
    # 4. Боксплоты для колонок с выбросами
    for col in get_numeric_columns(df)[:3]:
        try:
            chart = BoxPlot(
                df=df,
                y_column=col,
                title=f"Разброс: {col}"
            )
            comp = ChartComponent(chart=chart.plot())
            comp.name = f"📦 {col}"
            components.append(comp)
        except:
            pass
    
    # 5. Таблица уникальности
    text = "### 🔑 Уникальность\n\n"
    for col in df.columns:
        unique = df[col].nunique()
        if unique == len(df):
            text += f"🔑 {col}: уникально (возможный ID)\n"
        elif unique == 1:
            text += f"⚠️ {col}: одно значение\n"
    
    if text != "### 🔑 Уникальность\n\n":
        comp = TextComponent(text=text)
        comp.name = "🔑 Уникальность"
        components.append(comp)
    
    return components


# Словарь шаблонов
TEMPLATES = {
    "quick_overview": {
        "name": "Быстрый обзор",
        "description": "Общая статистика, распределения, корреляции, проблемы",
        "icon": "🔍",
        "function": template_quick_overview,
        "requirements": {
            "min_rows": 1,
            "message": "Можно применять к любым данным"
        }
    },
    "sales_analysis": {
        "name": "Анализ продаж",
        "description": "Суммы, топ-продукты, динамика, ABC-анализ",
        "icon": "💰",
        "function": template_sales_analysis,
        "requirements": {
            "min_rows": 5,
            "needs_numeric": True,
            "needs_keywords": ["сумма", "amount", "цена", "price", "total"],
            "message": "Нужна колонка с суммами/ценами"
        }
    },
    "customer_analysis": {
        "name": "Анализ клиентов",
        "description": "Категории, метрики, группировка, сегменты",
        "icon": "👥",
        "function": template_customer_analysis,
        "requirements": {
            "min_rows": 10,
            "needs_categorical_or_numeric": True,
            "message": "Нужны категории или числовые метрики"
        }
    },
    "time_series": {
        "name": "Временные ряды",
        "description": "Тренды, сезонность, динамика по периодам",
        "icon": "📅",
        "function": template_time_series,
        "requirements": {
            "min_rows": 10,
            "needs_date": True,
            "message": "Нужна колонка с датами"
        }
    },
    "correlation_analysis": {
        "name": "Корреляции",
        "description": "Матрица корреляций, сильные связи, диаграммы рассеяния",
        "icon": "🔗",
        "function": template_correlation_analysis,
        "requirements": {
            "min_rows": 10,
            "min_numeric": 2,
            "message": "Нужно минимум 2 числовые колонки"
        }
    },
    "data_quality": {
        "name": "Качество данных",
        "description": "Пропуски, выбросы, дубликаты, оценка качества",
        "icon": "✅",
        "function": template_data_quality,
        "requirements": {
            "min_rows": 1,
            "message": "Можно применять к любым данным"
        }
    }
}


def check_template_requirements(df, template_key):
    """Проверить, подходят ли данные для шаблона"""
    template = TEMPLATES.get(template_key)
    if not template:
        return False, ["Шаблон не найден"]
    
    reqs = template["requirements"]
    warnings = []
    
    # Минимальное количество строк
    if len(df) < reqs.get("min_rows", 1):
        warnings.append(f"Слишком мало строк: {len(df)} < {reqs['min_rows']}")
    
    # Нужны числовые колонки
    if reqs.get("needs_numeric") or reqs.get("min_numeric"):
        num_cols = get_numeric_columns(df)
        min_num = reqs.get("min_numeric", 1)
        if len(num_cols) < min_num:
            warnings.append(f"Недостаточно числовых колонок: {len(num_cols)} < {min_num}")
    
    # Нужны ключевые слова
    if reqs.get("needs_keywords"):
        found = False
        for col in df.columns:
            if any(kw in col.lower() for kw in reqs["needs_keywords"]):
                found = True
                break
        if not found:
            warnings.append("Не найдены подходящие колонки (суммы, цены и т.д.)")
    
    # Нужны даты
    if reqs.get("needs_date"):
        date_found = False
        for col in df.columns:
            if 'date' in col.lower() or 'дата' in col.lower():
                date_found = True
                break
        if not date_found:
            warnings.append("Не найдена колонка с датами")
    
    # Нужны категории или числовые
    if reqs.get("needs_categorical_or_numeric"):
        cat_cols = get_categorical_columns(df)
        num_cols = get_numeric_columns(df)
        if not cat_cols and not num_cols:
            warnings.append("Нет категориальных или числовых колонок")
    
    return len(warnings) == 0, warnings


def apply_template(template_key, df, app_state):
    """Применить шаблон анализа"""
    template = TEMPLATES.get(template_key)
    if not template:
        return None, ["Шаблон не найден"]
    
    # Проверяем требования
    can_run, warnings = check_template_requirements(df, template_key)
    
    if not can_run:
        return None, warnings
    
    # Выполняем шаблон
    try:
        components = template["function"](df, app_state)
        return components, warnings
    except Exception as e:
        return None, [f"Ошибка выполнения: {str(e)}"]