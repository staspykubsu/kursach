from re import T
import streamlit as st
import pandas as pd
import os
import json
import nbformat
from uuid import uuid4
from state import AppState
from components import TextComponent, ChartComponent, DataInfoComponent
from charts import (
    LineChart,
    BarChart,
    Histogram,
    ScatterChart,
    PieChart,
    BoxPlot,
    ViolinPlot,
    Heatmap,
    AreaChart,
    FunnelChart,
)
from dfinfo import DataFrameInfo
from df_operations import FilterOperation, AggregateOperation, LoadCsvOperation, SaveCsvOperation, DataCleanOperation
from codegen import generate_notebook_cells
from statistics import CorrelationAnalysis, TTestAnalysis, ChiSquaredTest, ANOVAAnalysis
from preprocessing import DataPreprocessor
from report_generator import ReportGenerator
from datetime import datetime
from gigachat_assistant import GigaChatAssistant, get_dataframe_info

# Настройка страницы
st.set_page_config(layout="wide", page_title="Система исследования данных", page_icon="🚽")

# Инициализация состояния сессии
if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()

if "show_df_info" not in st.session_state:
    st.session_state.show_df_info = False

# Карта типов графиков
chart_types = {
    "Линейный график": LineChart,
    "Столбчатая диаграмма": BarChart,
    "Гистограмма": Histogram,
    "Диаграмма рассеяния": ScatterChart,
    "Круговая диаграмма": PieChart,
    "Ящик с усами": BoxPlot,
    "Скрипичная диаграмма": ViolinPlot,
    "Тепловая карта": Heatmap,
    "Диаграмма с областями": AreaChart,
    "Воронкообразная диаграмма": FunnelChart,
}


# Вспомогательные функции
@st.dialog("Сохранить состояние")
def save_state():
    st.write("Сохранение текущего состояния")
    
    if not st.session_state.app_state.operations:
        st.warning("Нет операций для сохранения.")
        return
    
    st.write(f"Будет сохранено операций: {len(st.session_state.app_state.operations)}")
    st.write(f"Компонентов: {len(st.session_state.app_state.components)}")
    
    if st.button("Сохранить"):
        try:
            os.makedirs("data", exist_ok=True)
            st.session_state.app_state.save_state("data/app_state.json")
            st.success("Состояние успешно сохранено!")
        except Exception as e:
            st.error(f"Ошибка при сохранении состояния: {e}")


@st.dialog("Загрузить состояние")
def load_state():
    st.write("Загрузка сохранённого состояния")
    
    if not os.path.exists("data/app_state.json"):
        st.warning("Файл состояния не найден. Сначала сохраните состояние.")
        return
    
    # Показываем информацию о файле
    file_size = os.path.getsize("data/app_state.json")
    st.write(f"Найден файл: data/app_state.json ({file_size} байт)")
    
    if st.button("Загрузить"):
        try:
            st.session_state.app_state.load_state("data/app_state.json")
            st.success("Состояние успешно загружено!")
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка при загрузке состояния: {e}")

@st.dialog("Сгенерировать ноутбук")
def generate_notebook():
    """Генерация Jupyter notebook из текущих операций"""
    st.write("Генерация Jupyter Notebook")

    notebook_path = st.text_input("Путь к ноутбуку", value="data_analysis.ipynb")

    if st.button("Создать ноутбук"):
        try:
            # Создание ячеек ноутбука на основе операций
            cells = generate_notebook_cells()

            if not cells:
                st.error("Нет операций для генерации ноутбука. Добавьте операции с данными.")
                return

            # Создание ноутбука
            notebook = nbformat.v4.new_notebook()
            notebook.cells = cells

            # Сохранение ноутбука
            os.makedirs(os.path.dirname(notebook_path) if os.path.dirname(notebook_path) else ".", exist_ok=True)
            with open(notebook_path, "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)

            st.success(f"Ноутбук сохранён: {notebook_path}")

            # Кнопка для скачивания
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook_content = f.read()

            st.download_button(
                label="Скачать ноутбук",
                data=notebook_content,
                file_name=os.path.basename(notebook_path),
                mime="application/x-ipynb+json",
            )
        except Exception as e:
            st.error(f"Ошибка при генерации ноутбука: {e}")
            import traceback
            st.code(traceback.format_exc())


# Диалог для добавления или редактирования текстового компонента
@st.dialog("Добавить текст")
def add_text(component_id=None, edit=False):
    dialog_title = "Редактировать текстовый компонент" if edit else "Добавить текстовый компонент"
    st.write(dialog_title)

    component = None
    component_text = ""

    if edit and component_id:
        component = next(
            (c for c in st.session_state.app_state.components if getattr(c, "id", id(c)) == component_id), None
        )
        if component:
            component_text = getattr(component, "text", "")

    text = st.text_area("Текстовое содержимое", value=component_text)

    if st.button("Отправить"):
        if edit and component:
            # Обновление свойств компонента
            component.text = text
            st.session_state.app_state.update_component(component_id, component)
            st.rerun()
        else:
            # Создание компонента
            component = TextComponent(text=text)
            st.session_state.app_state.add_component(component)
            st.rerun()


def edit_text(component_id):
    return add_text(component_id=component_id, edit=True)


# Диалог для добавления или редактирования компонента графика
@st.dialog("График", width="large")
def add_chart(component_id=None, edit=False):
    """
    Диалог для добавления или редактирования компонента графика в приложении.

    Аргументы:
        component_id (str, optional): ID компонента для редактирования. По умолчанию None.
        edit (bool, optional): Является ли это операцией редактирования. По умолчанию False.
    """
    dialog_title = "Редактировать компонент графика" if edit else "Добавить компонент графика"
    st.write(dialog_title)

    component = None
    source_df_id = st.session_state.app_state.current_df_id
    chart_type = "Линейный график"
    existing_params = None

    if edit and component_id:
        # Получение компонента для редактирования
        component = get_component_by_id(component_id)
        if not component or not hasattr(component, "chart"):
            st.error("Компонент графика не найден")
            return

        source_df_id = getattr(component, "source_df_id", st.session_state.app_state.current_df_id)

        # Получение типа графика
        chart_type = getattr(component, "chart_type", "Линейный график")
        if not chart_type:
            chart_type = get_chart_type_from_component(component)

        # Получение существующих параметров или извлечение из графика
        existing_params = getattr(component, "chart_params", None)
        if not existing_params:
            existing_params = extract_chart_parameters(component.chart)

    # Получение исходного датафрейма
    selected_df_id, df = select_df(source_df_id)
    if df is None:
        return

    # Разрешить пользователю изменить тип графика
    if edit:
        chart_type = st.selectbox(
            "Тип графика", list(chart_types.keys()), index=list(chart_types.keys()).index(chart_type)
        )
    else:
        chart_type = st.selectbox("Тип графика", list(chart_types.keys()))

    # Получение параметров графика в зависимости от типа
    chart_params = get_chart_params(chart_type, df, existing_params)

    if st.button("Отправить"):
        # Создание графика в зависимости от типа
        chart_fig = create_chart(chart_type, df, chart_params)

        if edit and component:
            # Обновление компонента
            component.chart = chart_fig
            setattr(component, "source_df_id", selected_df_id)

            # Обновление сохраненных параметров графика и типа
            setattr(component, "chart_type", chart_type)
            setattr(component, "chart_params", chart_params)

            # Обновление компонента в состоянии приложения
            st.session_state.app_state.update_component(component_id, component)
            st.rerun()
        else:
            # Создание компонента
            component = ChartComponent(chart=chart_fig)

            # Сохранение ID исходного датафрейма и метаданных графика для последующего редактирования
            setattr(component, "source_df_id", selected_df_id)
            setattr(component, "chart_type", chart_type)
            setattr(component, "chart_params", chart_params)

            # Добавление компонента в состояние приложения
            st.session_state.app_state.add_component(component)
            st.rerun()


# Псевдоним для edit_chart, который вызывает add_chart с edit=True
def edit_chart(component_id):
    """
    Диалог для редактирования существующего компонента графика.

    Аргументы:
        component_id (str): ID компонента для редактирования
    """
    return add_chart(component_id=component_id, edit=True)


def select_df(source_df_id=None):
    """Вспомогательная функция для получения исходного датафрейма для графика"""
    # Получение доступных датафреймов
    df_names = st.session_state.app_state.get_dataframe_names()
    if not df_names:
        st.warning("Нет доступных данных. Пожалуйста, сначала загрузите данные.")
        return None, None

    # Выбор исходного датафрейма
    df_options = list(df_names.items())

    if source_df_id and source_df_id in [df_id for df_id, _ in df_options]:
        index = [df_id for df_id, _ in df_options].index(source_df_id)
    else:
        index = 0

    selected_df_id = st.selectbox(
        "Выберите датафрейм",
        options=[df_id for df_id, _ in df_options],
        index=index,
        format_func=lambda x: df_names.get(x, "Неизвестный"),
    )

    df = st.session_state.app_state.get_dataframe_by_id(selected_df_id)
    if df is None:
        st.warning("Выбранный датафрейм недоступен.")
        return None, None
    else:
        draw_sample_data(df)

    return selected_df_id, df


def draw_sample_data(df):
    with st.expander("Пример данных"):
        # Получение количества строк для выборки (минимум 5 или общее количество строк)
        sample_size = min(5, len(df))
        st.dataframe(df.sample(sample_size) if sample_size > 0 else df.head(0))


def get_chart_params(chart_type, df, existing_params=None):
    """
    Получение параметров графика в зависимости от типа графика.

    Аргументы:
        chart_type (str): Тип создаваемого графика
        df (pd.DataFrame): Датафрейм для использования в графике
        existing_params (dict, optional): Существующие параметры для редактирования. По умолчанию None.

    Возвращает:
        dict: Параметры для графика
    """
    columns = df.columns.tolist()
    existing_params = existing_params or {}

    # Общие параметры для всех графиков
    chart_params = {
        "title": st.text_input("Заголовок графика", value=existing_params.get("title", "")),
    }

    # Добавление x_column для всех типов графиков, кроме круговой диаграммы
    if chart_type != "Круговая диаграмма":
        default_x = existing_params.get("x_column", "")
        chart_params["x_column"] = st.selectbox(
            "Колонка X",
            options=columns,
            index=columns.index(default_x) if default_x in columns else 0,
        )

    # Добавление y_column для графиков, которым это необходимо
    if chart_type not in ["Гистограмма", "Круговая диаграмма"]:
        default_y = existing_params.get("y_column", "")
        chart_params["y_column"] = st.selectbox(
            "Колонка Y",
            options=columns,
            index=columns.index(default_y) if default_y in columns else 0,
        )

    # Добавление специфичных для графика параметров
    if chart_type in ["Линейный график", "Диаграмма рассеяния"]:
        default_hue = existing_params.get("hue_column", "")
        chart_params["hue_column"] = st.selectbox(
            "Цвет по",
            options=[""] + columns,
            index=([""] + columns).index(default_hue) if default_hue in [""] + columns else 0,
        )

    if chart_type == "Диаграмма рассеяния":
        default_size = existing_params.get("size_column", "")
        chart_params["size_column"] = st.selectbox(
            "Размер по",
            options=[""] + columns,
            index=([""] + columns).index(default_size) if default_size in [""] + columns else 0,
        )

    if chart_type in ["Столбчатая диаграмма", "Круговая диаграмма"]:
        default_group = existing_params.get("group_by", "")
        chart_params["group_by"] = st.selectbox(
            "Группировать по",
            options=[""] + columns,
            index=([""] + columns).index(default_group) if default_group in [""] + columns else 0,
        )

        # Добавление values_column для круговой диаграммы
        if chart_type == "Круговая диаграмма":
            default_values = existing_params.get("values_column", "")
            chart_params["values_column"] = st.selectbox(
                "Значения (опционально)",
                options=[""] + columns,
                index=([""] + columns).index(default_values) if default_values in [""] + columns else 0,
            )

    if chart_type == "Гистограмма":
        chart_params["bins"] = st.slider(
            "Количество интервалов",
            min_value=5,
            max_value=100,
            value=existing_params.get("bins", 10),
            step=5,
        )
        default_color = existing_params.get("color", "")
        chart_params["color"] = st.selectbox(
            "Цвет по",
            options=[""] + columns,
            index=([""] + columns).index(default_color) if default_color in [""] + columns else 0,
        )

    if chart_type in ["Ящик с усами", "Скрипичная диаграмма"]:
        default_color = existing_params.get("color", "")
        chart_params["color"] = st.selectbox(
            "Цвет по",
            options=[""] + columns,
            index=([""] + columns).index(default_color) if default_color in [""] + columns else 0,
        )

    # Добавление дополнительных параметров
    chart_params["extra_params"] = extra_params(existing_params)

    return chart_params


def extra_params(existing_params=None):
    """
    Добавление дополнительных параметров к графикам.

    Аргументы:
        existing_params (dict, optional): Существующие параметры для редактирования. По умолчанию None.

    Возвращает:
        dict: Дополнительные параметры для графика
    """
    # Инициализация состояния сессии для дополнительных параметров, если не существует
    if "extra_params_state" not in st.session_state:
        st.session_state.extra_params_state = existing_params.get("extra_params", {}) if existing_params else {}

    # Использование состояния сессии для хранения параметров
    params = st.session_state.extra_params_state

    with st.expander("Дополнительные параметры (Продвинутые, используйте с осторожностью)"):
        # Отображение существующих параметров
        params_to_delete = []
        params_to_update = {}

        for key, value in params.items():
            col1, col2, col3 = st.columns([2, 4, 1])
            with col1:
                new_key = st.text_input("Параметр", value=key, key=f"param_key_{key}")
            with col2:
                new_value = st.text_input("Значение", value=value, key=f"param_value_{key}")
            with col3:
                if st.button("🗑️", key=f"delete_param_{key}"):
                    params_to_delete.append(key)

            # Проверка, изменился ли ключ или значение
            if new_key != key or new_value != value:
                params_to_update[key] = (new_key, new_value)

        # Обработка обновлений параметров
        for old_key, (new_key, new_value) in params_to_update.items():
            if old_key in params:
                del params[old_key]
            params[new_key] = new_value

        # Обработка удалений параметров
        for key in params_to_delete:
            if key in params:
                del params[key]

        # Кнопка добавления нового параметра
        if st.button("➕ Добавить параметр"):
            # Генерация уникального временного ключа
            temp_key = f"new_param_{len(params)}"
            params[temp_key] = ""

    return params


def create_chart(chart_type, df, params):
    """
    Создание графика в зависимости от типа графика и параметров.

    Аргументы:
        chart_type (str): Тип создаваемого графика
        df (pd.DataFrame): Датафрейм для использования в графике
        params (dict): Параметры для графика

    Возвращает:
        plotly.graph_objects.Figure: Созданный график
    """
    chart_class = chart_types[chart_type]

    # Создание базовых параметров графика
    chart_params = {
        "df": df,
    }

    # Добавление только параметров, которые имеют значения
    if params.get("title"):
        chart_params["title"] = params["title"]

    # Добавление x_column для графиков, которым это необходимо (все, кроме круговой диаграммы)
    if chart_type != "Круговая диаграмма" and params.get("x_column"):
        chart_params["x_column"] = params["x_column"]

    # Добавление y_column для графиков, которым это необходимо
    if chart_type not in ["Гистограмма", "Круговая диаграмма"] and params.get("y_column"):
        chart_params["y_column"] = params["y_column"]

    # Добавление hue_column только для графиков, которые его поддерживают
    if chart_type in ["Линейный график", "Диаграмма рассеяния"] and params.get("hue_column"):
        chart_params["hue_column"] = params["hue_column"]

    # Добавление size_column для диаграмм рассеяния
    if chart_type == "Диаграмма рассеяния" and params.get("size_column"):
        chart_params["size_column"] = params["size_column"]

    # Добавление group_by для графиков, которым это необходимо
    if chart_type in ["Столбчатая диаграмма", "Круговая диаграмма"] and params.get("group_by"):
        chart_params["group_by"] = params["group_by"]

    # Добавление values_column для круговой диаграммы
    if chart_type == "Круговая диаграмма" and params.get("values_column"):
        chart_params["values_column"] = params["values_column"]

    # Добавление специфичных для гистограммы параметров
    if chart_type == "Гистограмма":
        if "bins" in params:
            chart_params["bins"] = params["bins"]
        if params.get("color"):
            chart_params["color"] = params["color"]

    # Добавление параметра color для графиков, которые его поддерживают
    if chart_type in ["Ящик с усами", "Скрипичная диаграмма", "Диаграмма с областями", "Воронкообразная диаграмма"] and params.get("color"):
        chart_params["color"] = params["color"]

    # Добавление дополнительных параметров
    if params.get("extra_params"):
        chart_params["params"] = params["extra_params"]

    # Создание графика
    chart = chart_class(**chart_params)
    return chart.plot()


def get_component_by_id(component_id):
    """Вспомогательная функция для получения компонента по его ID"""
    return next((c for c in st.session_state.app_state.components if getattr(c, "id", id(c)) == component_id), None)


def get_chart_type_from_component(component):
    """Вспомогательная функция для определения типа графика из компонента"""
    chart_type = "Линейный график"  # По умолчанию
    for name, chart_class in chart_types.items():
        if component.chart.__class__.__name__ == chart_class.__name__:
            chart_type = name
            break
    return chart_type


def extract_chart_parameters(chart):
    """Вспомогательная функция для извлечения параметров из существующего графика"""
    params = {}

    # Для фигур Plotly нам нужно извлечь из данных фигуры и макета
    try:
        # Извлечение заголовка из макета
        if hasattr(chart, "layout") and hasattr(chart.layout, "title"):
            if hasattr(chart.layout.title, "text"):
                params["title"] = chart.layout.title.text
            else:
                params["title"] = str(chart.layout.title)

        # Извлечение колонок x и y из данных фигуры
        if hasattr(chart, "data") and len(chart.data) > 0:
            # Попытка получить x_column из первой трассировки
            if hasattr(chart.data[0], "x") and isinstance(chart.data[0].x, pd.Series):
                params["x_column"] = chart.data[0].x.name

            # Попытка получить y_column из первой трассировки
            if hasattr(chart.data[0], "y") and isinstance(chart.data[0].y, pd.Series):
                params["y_column"] = chart.data[0].y.name

            # Для гистограмм данные x находятся в 'x', а не в 'y'
            if chart.data[0].type == "histogram" and "y_column" not in params and "x_column" in params:
                params["y_column"] = params["x_column"]

            # Попытка извлечь информацию о цвете/оттенке
            if hasattr(chart.data[0], "marker") and hasattr(chart.data[0].marker, "color"):
                if isinstance(chart.data[0].marker.color, pd.Series):
                    params["hue_column"] = chart.data[0].marker.color.name
                elif isinstance(chart.data[0].marker.color, str):
                    params["color"] = chart.data[0].marker.color

            # Попытка извлечь информацию о размере для диаграмм рассеяния
            if chart.data[0].type == "scatter" and hasattr(chart.data[0].marker, "size"):
                if isinstance(chart.data[0].marker.size, pd.Series):
                    params["size_column"] = chart.data[0].marker.size.name

            # Для гистограмм попытка извлечь интервалы
            if chart.data[0].type == "histogram" and hasattr(chart.data[0], "nbinsx"):
                params["bins"] = chart.data[0].nbinsx
    except Exception as e:
        st.warning(f"Не удалось извлечь все параметры из графика: {e}")

    # Установка значений по умолчанию для отсутствующих параметров
    params.setdefault("name", "")
    params.setdefault("title", "")
    params.setdefault("x_column", "")
    params.setdefault("y_column", "")
    params.setdefault("hue_column", "")
    params.setdefault("size_column", "")
    params.setdefault("group_by", "")
    params.setdefault("bins", 10)
    params.setdefault("color", "")

    return params


# Диалог для загрузки CSV
@st.dialog("Загрузить CSV")
def load_csv():
    st.write("Загрузить CSV файл")

    name = st.text_input("Название операции", value=f"Загрузка CSV {uuid4().hex[:4]}")

    # Получение пути к файлу и разделителя
    file_path, separator = get_csv_file_parameters()

    if st.button("Отправить"):
        if not file_path:
            st.error("Пожалуйста, укажите путь к файлу или загрузите файл")
            return

        # Создание операции
        operation = LoadCsvOperation(name=name, id=uuid4().hex, file_path=file_path, sep=separator)
        st.session_state.app_state.add_operation(operation)
        st.rerun()


def get_csv_file_parameters():
    """Вспомогательная функция для получения параметров CSV файла"""
    file_path = st.text_input("Путь к файлу")

    # Загрузка файла как альтернатива
    uploaded_file = st.file_uploader("Или загрузите файл", type=["csv"])

    # Получение разделителя
    separator = get_csv_separator()

    # Обработка загрузки файла
    if uploaded_file is not None:
        # Сохранение загруженного файла
        save_path = os.path.join("data", uploaded_file.name)
        os.makedirs("data", exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_path = save_path

    return file_path, separator


def get_csv_separator():
    """Вспомогательная функция для получения разделителя CSV"""
    # Выбор разделителя
    separator_options = {"Запятая (,)": ",", "Точка с запятой (;)": ";", "Табуляция (\t)": "\t", "Вертикальная черта (|)": "|", "Пробел ( )": " "}
    separator_choice = st.selectbox("Разделитель CSV", options=list(separator_options.keys()), index=0)
    separator = separator_options[separator_choice]

    # Опция пользовательского разделителя
    use_custom_separator = st.checkbox("Использовать пользовательский разделитель")
    if use_custom_separator:
        custom_separator = st.text_input("Пользовательский разделитель")
        if custom_separator:
            separator = custom_separator

    return separator


# Диалог для добавления фильтра
@st.dialog("Добавить фильтр")
def add_filter(operation_id=None, edit=False):
    dialog_title = "Редактировать операцию фильтрации" if edit else "Добавить фильтр"
    st.write(dialog_title)

    # Инициализация переменных
    operation = None
    source_df_id = None
    operation_name = f"Фильтр {uuid4().hex[:4]}"
    filter_column = None
    filter_type = None
    filter_value = None

    # Если редактирование, получить операцию и её значения
    if edit and operation_id:
        operation = next((op for op in st.session_state.app_state.operations if op.id == operation_id), None)
        if not operation or not hasattr(operation, "column") or operation.operation_type != "filter":
            st.error("Операция фильтрации не найдена")
            return

        source_df_id = operation.source_df_id
        operation_name = operation.name
        filter_column = operation.column
        filter_type = operation.filter_type
        filter_value = operation.filter_value

    # Получение исходного датафрейма
    source_df_id, df = get_source_df("Выберите исходный датафрейм" if not edit else "Выберите исходный датафрейм", source_df_id)
    if df is None:
        return

    name = st.text_input("Название операции", value=operation_name)

    # Получение параметров фильтра
    if edit and filter_column and filter_type:
        # Для редактирования нам нужно предварительно выбрать колонку и тип фильтра
        columns = df.columns.tolist()

        # Установка индекса колонки по умолчанию
        column_index = 0
        if filter_column in columns:
            column_index = columns.index(filter_column)

        column = st.selectbox("Выберите колонку", options=columns, index=column_index)

        # Установка индекса типа фильтра по умолчанию
        filter_types = get_filter_types(df[column].dtype)
        filter_type_index = 0
        if filter_type in filter_types:
            filter_type_index = filter_types.index(filter_type)

        filter_type = st.selectbox("Тип фильтра", options=filter_types, index=filter_type_index)

        # Установка значения фильтра
        if filter_type == "isin":
            # Для 'isin' нам нужно обработать список значений
            if isinstance(filter_value, list):
                filter_value_str = ", ".join(str(val) for val in filter_value)
            else:
                filter_value_str = str(filter_value)
            filter_value_input = st.text_input(
                "Значения фильтра (через запятую для нескольких значений)", value=filter_value_str
            )
        else:
            # Для других типов фильтра это одно значение
            filter_value_input = st.text_input("Значение фильтра", value=str(filter_value))

        filter_params = {"column": column, "filter_type": filter_type, "filter_value": filter_value_input}
    else:
        # Для добавления новых операций используем существующую вспомогательную функцию
        filter_params = get_filter_parameters(df)

    if st.button("Отправить"):
        # Преобразование значения фильтра в зависимости от типа данных колонки
        filter_value = convert_filter_value(
            df, filter_params["column"], filter_params["filter_type"], filter_params["filter_value"]
        )

        if edit and operation:
            # Обновление операции
            operation.name = name
            operation.column = filter_params["column"]
            operation.filter_type = filter_params["filter_type"]
            operation.filter_value = filter_value
            operation.source_df_id = source_df_id

            # Обновление операции в состоянии приложения
            st.session_state.app_state.update_operation(operation_id, operation)
        else:
            # Создание операции
            operation = FilterOperation(
                name=name,
                id=uuid4().hex,
                column=filter_params["column"],
                filter_type=filter_params["filter_type"],
                filter_value=filter_value,
                source_df_id=source_df_id,
            )
            st.session_state.app_state.add_operation(operation)
        st.rerun()


def edit_filter_operation(operation_id):
    """Диалог для редактирования операции фильтрации"""
    return add_filter(operation_id=operation_id, edit=True)


def get_filter_parameters(df):
    """Вспомогательная функция для получения параметров фильтра"""
    columns = df.columns.tolist()
    column = st.selectbox("Колонка", options=columns)

    filter_types = get_filter_types(df[column].dtype)
    filter_type = st.selectbox("Тип фильтра", options=filter_types)

    # Различный ввод в зависимости от типа фильтра
    if filter_type == "between":
        col1, col2 = st.columns(2)
        with col1:
            min_value = st.text_input("Минимальное значение")
        with col2:
            max_value = st.text_input("Максимальное значение")
        filter_value = f"{min_value},{max_value}"
    elif filter_type == "isin":
        filter_value = st.text_input("Значения фильтра (через запятую для нескольких значений)")
    else:
        filter_value = st.text_input("Значение фильтра")

    return {"column": column, "filter_type": filter_type, "filter_value": filter_value}


def get_filter_types(dtype):
    """Вспомогательная функция для получения подходящих типов фильтра в зависимости от типа данных колонки"""
    # Базовые типы фильтра для всех типов данных
    filter_types = ["equals", "not_equals", "isin"]

    # Добавление специфичных для числовых данных типов фильтра
    if pd.api.types.is_numeric_dtype(dtype):
        filter_types.extend(["greater_than", "less_than", "between"])

    # Добавление специфичных для строковых данных типов фильтра
    if pd.api.types.is_string_dtype(dtype):
        filter_types.extend(["contains", "startswith", "endswith"])

    # Добавление специфичных для данных даты/времени типов фильтра
    if pd.api.types.is_datetime64_dtype(dtype):
        filter_types.extend(["before", "after", "between"])

    return filter_types


def convert_filter_value(df, column, filter_type, filter_value):
    """Вспомогательная функция для преобразования значения фильтра в зависимости от типа данных колонки"""
    try:
        # Обработка пустых значений
        if not filter_value:
            if filter_type == "isin":
                return []
            return None

        # Обработка типов фильтра на основе списка
        if filter_type == "isin":
            values = [val.strip() for val in filter_value.split(",")]
            # Преобразование в соответствующий тип в зависимости от колонки
            if pd.api.types.is_numeric_dtype(df[column].dtype):
                return [float(val) if "." in val else int(val) for val in values if val]
            elif pd.api.types.is_datetime64_dtype(df[column].dtype):
                return [pd.to_datetime(val) for val in values if val]
            else:
                return values

        # Обработка типа фильтра between
        if filter_type == "between":
            if "," in filter_value:
                min_val, max_val = filter_value.split(",", 1)
                min_val = min_val.strip()
                max_val = max_val.strip()

                # Преобразование в соответствующий тип
                if pd.api.types.is_numeric_dtype(df[column].dtype):
                    return [
                        float(min_val) if "." in min_val else int(min_val),
                        float(max_val) if "." in max_val else int(max_val),
                    ]
                elif pd.api.types.is_datetime64_dtype(df[column].dtype):
                    return [pd.to_datetime(min_val), pd.to_datetime(max_val)]
                else:
                    return [min_val, max_val]
            else:
                # Если нет запятой, вернуть как список из одного значения
                return [filter_value, filter_value]

        # Обработка типов фильтра с одним значением
        if pd.api.types.is_numeric_dtype(df[column].dtype):
            # Преобразование в float или int в зависимости от значения
            if "." in filter_value:
                return float(filter_value)
            else:
                return int(filter_value)
        elif pd.api.types.is_datetime64_dtype(df[column].dtype):
            return pd.to_datetime(filter_value)
        else:
            # Для строковых колонок вернуть как есть
            return filter_value
    except (ValueError, TypeError):
        # Если преобразование не удалось, вернуть как строку
        return filter_value


# Диалог для добавления агрегации
@st.dialog("Добавить агрегацию")
def add_aggregation(operation_id=None, edit=False):
    dialog_title = "Редактировать операцию агрегации" if edit else "Добавить агрегацию"
    st.write(dialog_title)

    # Инициализация переменных
    operation = None
    source_df_id = None
    operation_name = f"Агрегация {uuid4().hex[:4]}"
    group_by = []
    agg_func = {}

    # Если редактирование, получить операцию и её значения
    if edit and operation_id:
        operation = next((op for op in st.session_state.app_state.operations if op.id == operation_id), None)
        if not operation or not hasattr(operation, "group_by") or operation.operation_type != "aggregate":
            st.error("Операция агрегации не найдена")
            return

        source_df_id = operation.source_df_id
        operation_name = operation.name
        group_by = operation.group_by
        agg_func = operation.agg_func

    # Получение исходного датафрейма
    source_df_id, df = get_source_df("Выберите исходный датафрейм" if not edit else "Выберите исходный датафрейм", source_df_id)
    if df is None:
        return

    name = st.text_input("Название операции", value=operation_name)

    # Получение параметров агрегации
    if edit and operation:
        # Для редактирования нам нужно предварительно выбрать колонки группировки и функции агрегации
        columns = df.columns.tolist()

        # Колонки группировки со значениями по умолчанию из операции
        group_by = st.multiselect("Колонки для группировки", options=columns, default=group_by)

        # Функции агрегации для каждой числовой колонки
        st.subheader("Функции агрегации")
        agg_func_new = {}

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in group_by]
        if not numeric_columns:
            st.warning("Нет числовых колонок, доступных для агрегации")
        else:
            for col in numeric_columns:
                agg_options = ["none", "mean", "sum", "min", "max", "count", "median"]
                # Установка индекса по умолчанию на основе существующей операции
                default_index = 0
                if col in agg_func:
                    try:
                        default_index = agg_options.index(agg_func[col])
                    except ValueError:
                        default_index = 0

                selected_agg = st.selectbox(
                    f"Агрегация для {col}", options=agg_options, index=default_index, key=f"edit_agg_{col}"
                )

                # Добавление только функций агрегации, отличных от 'none', в словарь
                if selected_agg != "none":
                    agg_func_new[col] = selected_agg

        agg_params = {"group_by": group_by, "agg_func": agg_func_new}
    else:
        # Для добавления новых операций используем существующую вспомогательную функцию
        agg_params = get_aggregation_parameters(df)

    if st.button("Отправить"):
        if not agg_params["group_by"]:
            st.error("Пожалуйста, выберите хотя бы одну колонку для группировки")
            return

        if not agg_params["agg_func"]:
            st.error("Пожалуйста, выберите хотя бы одну функцию агрегации")
            return

        if edit and operation:
            # Обновление операции
            operation.name = name
            operation.group_by = agg_params["group_by"]
            operation.agg_func = agg_params["agg_func"]
            operation.source_df_id = source_df_id

            # Обновление операции в состоянии приложения
            st.session_state.app_state.update_operation(operation_id, operation)
        else:
            # Создание операции
            operation = AggregateOperation(
                name=name,
                id=uuid4().hex,
                group_by=agg_params["group_by"],
                agg_func=agg_params["agg_func"],
                source_df_id=source_df_id,
            )
            st.session_state.app_state.add_operation(operation)
        st.rerun()


def edit_aggregation_operation(operation_id):
    """Диалог для редактирования операции агрегации"""
    return add_aggregation(operation_id=operation_id, edit=True)


def get_aggregation_parameters(df):
    """
    Получение параметров агрегации для операций с датафреймом.

    Аргументы:
        df (pd.DataFrame): Датафрейм для агрегации

    Возвращает:
        dict: Параметры агрегации, включая колонки группировки и функции агрегации
    """
    columns = df.columns.tolist()
    group_by = st.multiselect("Колонки для группировки", options=columns)

    # Функции агрегации для каждой числовой колонки
    st.subheader("Функции агрегации")
    agg_func = {}

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in group_by]
    if not numeric_columns:
        st.warning("Нет числовых колонок, доступных для агрегации")
    else:
        for col in numeric_columns:
            agg_options = ["none", "mean", "sum", "min", "max", "count", "median"]
            selected_agg = st.selectbox(f"Агрегация для {col}", options=agg_options, index=0, key=f"agg_{col}")

            # Добавление только функций агрегации, отличных от 'none', в словарь
            if selected_agg != "none":
                agg_func[col] = selected_agg

    # Фильтрация любых значений 'none', которые могли проскользнуть
    return {"group_by": group_by, "agg_func": {k: v for k, v in agg_func.items() if v != "none"}}


# Диалог для добавления или редактирования очистки данных
@st.dialog("Добавить очистку данных")
def add_data_cleaning(operation_id=None, edit=False):
    dialog_title = "Редактировать операцию очистки данных" if edit else "Добавить очистку данных"
    st.write(dialog_title)

    # Инициализация переменных
    operation = None
    source_df_id = None
    operation_name = f"Очистка данных {uuid4().hex[:4]}"
    clean_type = None
    selected_columns = []
    params = {"fill_value": None, "replace_values": {}, "new_column_names": {}}

    # Если редактирование, получить операцию и её значения
    if edit and operation_id:
        operation = next((op for op in st.session_state.app_state.operations if op.id == operation_id), None)
        if not operation or not hasattr(operation, "clean_type"):
            st.error("Операция очистки данных не найдена")
            return

        source_df_id = operation.source_df_id
        operation_name = operation.name
        clean_type = operation.clean_type
        selected_columns = operation.columns if hasattr(operation, "columns") else []

        # Установка параметров в зависимости от типа очистки
        if operation.clean_type == "fillna" and hasattr(operation, "fill_value"):
            params["fill_value"] = operation.fill_value
        elif operation.clean_type == "replace" and hasattr(operation, "replace_values"):
            params["replace_values"] = operation.replace_values
        elif operation.clean_type == "rename" and hasattr(operation, "new_column_names"):
            params["new_column_names"] = operation.new_column_names

    # Получение исходного датафрейма
    source_df_id, df = get_source_df("Выберите исходный датафрейм" if not edit else "Выберите исходный датафрейм", source_df_id)
    if df is None:
        return

    # Установка названия операции
    name = st.text_input("Название операции", value=operation_name)

    # Выбор типа очистки
    if edit and clean_type:
        clean_types = ["dropna", "fillna", "drop_duplicates", "replace", "rename"]
        clean_type_index = clean_types.index(clean_type) if clean_type in clean_types else 0
        clean_type = st.selectbox(
            "Тип очистки" if not edit else "Метод очистки",
            options=clean_types,
            index=clean_type_index,
            format_func=lambda x: {
                "dropna": "Удалить пропущенные значения",
                "fillna": "Заполнить пропущенные значения",
                "drop_duplicates": "Удалить дубликаты",
                "replace": "Заменить значения",
                "rename": "Переименовать колонки",
            }.get(x, x),
        )
    else:
        clean_type = select_clean_type()

    # Получение колонок из датафрейма
    columns = df.columns.tolist()

    # Получение колонок для применения очистки
    if edit and selected_columns:
        # Для редактирования нам нужно обработать выбранные колонки по-другому
        if clean_type in ["dropna", "fillna", "drop_duplicates", "replace"]:
            selected_columns = st.multiselect(
                "Выберите колонки" if clean_type != "rename" else "Выберите колонки для переименования", options=columns
            )
        else:
            selected_columns = []
    else:
        selected_columns = select_clean_cols(clean_type, columns)

    # Получение дополнительных параметров в зависимости от типа
    if edit:
        if clean_type == "fillna":
            fill_options = ["", "mean", "median", "mode", "ffill", "bfill"]
            fill_value = params["fill_value"]
            fill_index = fill_options.index(fill_value) if fill_value in fill_options else 0
            params["fill_value"] = st.selectbox("Значение заполнения", options=fill_options, index=fill_index)
            if params["fill_value"] == "":
                params["fill_value"] = st.text_input(
                    "Пользовательское значение заполнения", value=str(fill_value) if fill_value not in fill_options else ""
                )
        elif clean_type == "replace":
            st.write("Заменить значения")
            replace_values = {}

            # Отображение существующих значений замены
            for i, (old_val, new_val) in enumerate(params["replace_values"].items()):
                col1, col2, col3 = st.columns([3, 3, 1])
                with col1:
                    old_value = st.text_input(f"Старое значение {i+1}", value=str(old_val))
                with col2:
                    new_value = st.text_input(f"Новое значение {i+1}", value=str(new_val))
                with col3:
                    if st.button("", key=f"del_replace_{i}"):
                        continue  # Пропустить эту пару в цикле

                replace_values[convert_val_to_type(old_value)] = convert_val_to_type(new_value)

            # Добавление нового значения замены
            if st.button("Добавить значение замены"):
                col1, col2 = st.columns(2)
                with col1:
                    old_value = st.text_input("Старое значение", key="new_old_val")
                with col2:
                    new_value = st.text_input("Новое значение", key="new_new_val")

                if old_value:
                    replace_values[convert_val_to_type(old_value)] = convert_val_to_type(new_value)

            params["replace_values"] = replace_values
        elif clean_type == "rename":
            st.write("Переименовать колонки")
            new_column_names = {}

            # Отображение существующих переименований колонок
            for i, (old_name, new_name) in enumerate(params["new_column_names"].items()):
                if old_name in df.columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text(f"Исходное: {old_name}")
                    with col2:
                        new_column_name = st.text_input(f"Новое имя для {old_name}", value=new_name, key=f"rename_{i}")
                        new_column_names[old_name] = new_column_name

            # Добавление опций для колонок, которые ещё не переименованы
            for col in df.columns:
                if col not in new_column_names:
                    rename_col = st.checkbox(f"Переименовать {col}", key=f"check_{col}")
                    if rename_col:
                        new_name = st.text_input(f"Новое имя для {col}", key=f"new_name_{col}")
                        if new_name:
                            new_column_names[col] = new_name

            params["new_column_names"] = new_column_names
    else:
        # Для добавления новых операций используем существующие вспомогательные функции
        cleaning_params = get_cleaning_params(clean_type, df, columns)
        params["fill_value"] = cleaning_params.get("fill_value")
        params["replace_values"] = cleaning_params.get("replace_values", {})
        params["new_column_names"] = cleaning_params.get("new_column_names", {})

    if st.button("Отправить"):
        if edit and operation:
            # Обновление операции
            operation.name = name
            operation.clean_type = clean_type
            operation.columns = selected_columns
            operation.source_df_id = source_df_id

            # Обновление специфичных параметров в зависимости от типа очистки
            if clean_type == "fillna":
                operation.fill_value = convert_val_to_type(params["fill_value"])
            elif clean_type == "replace":
                operation.replace_values = params["replace_values"]
            elif clean_type == "rename":
                operation.new_column_names = params["new_column_names"]

            # Обновление операции в состоянии приложения
            st.session_state.app_state.update_operation(operation_id, operation)
            st.rerun()
        else:
            # Создание операции
            operation = DataCleanOperation(
                name=name,
                id=uuid4().hex,
                clean_type=clean_type,
                columns=selected_columns,
                fill_value=params["fill_value"],
                replace_values=params["replace_values"],
                new_column_names=params["new_column_names"],
                source_df_id=source_df_id,
            )
            st.session_state.app_state.add_operation(operation)
            st.rerun()


# Псевдоним для edit_data_cleaning_operation, который вызывает add_data_cleaning с edit=True
def edit_data_cleaning_operation(operation_id):
    """Диалог для редактирования операции очистки данных"""
    return add_data_cleaning(operation_id=operation_id, edit=True)


def get_source_df(label="Выберите датафрейм", source_df_id=None):
    """Вспомогательная функция для получения исходного датафрейма для операций"""
    # Получение доступных датафреймов
    df_names = st.session_state.app_state.get_dataframe_names()
    if not df_names:
        st.warning("Нет доступных данных. Пожалуйста, сначала загрузите данные.")
        return None, None

    # Выбор исходного датафрейма
    df_options = list(df_names.items())

    # Установка индекса по умолчанию на основе source_df_id, если предоставлен
    index = 0
    if source_df_id and source_df_id in [df_id for df_id, _ in df_options]:
        index = [df_id for df_id, _ in df_options].index(source_df_id)

    selected_df_id = st.selectbox(
        label,
        options=[df_id for df_id, _ in df_options],
        index=index,
        format_func=lambda x: df_names.get(x, "Неизвестный"),
    )

    df = st.session_state.app_state.get_dataframe_by_id(selected_df_id)
    if df is None:
        st.warning("Выбранный датафрейм недоступен.")
        return None, None

    return selected_df_id, df


def select_clean_type():
    """Вспомогательная функция для выбора типа очистки"""
    clean_types = ["dropna", "fillna", "drop_duplicates", "replace", "rename"]
    return st.selectbox(
        "Метод очистки",
        options=clean_types,
        format_func=lambda x: {
            "dropna": "Удалить пропущенные значения",
            "fillna": "Заполнить пропущенные значения",
            "drop_duplicates": "Удалить дубликаты",
            "replace": "Заменить значения",
            "rename": "Переименовать колонки",
        }.get(x, x),
    )


def select_clean_cols(clean_type, columns):
    """Вспомогательная функция для выбора колонок для операции очистки"""
    if clean_type in ["dropna", "fillna", "drop_duplicates", "replace"]:
        return st.multiselect(
            "Выберите колонки" if clean_type != "rename" else "Выберите колонки для переименования", options=columns
        )
    else:
        return []


def get_cleaning_params(clean_type, df, columns):
    """Вспомогательная функция для получения дополнительных параметров в зависимости от типа очистки"""
    params = {"fill_value": None, "replace_values": {}, "new_column_names": {}}

    if clean_type == "fillna":
        params["fill_value"] = get_fillna_params()
    elif clean_type == "replace":
        params["replace_values"] = get_replace_params()
    elif clean_type == "rename":
        params["new_column_names"] = get_rename_params(columns)

    return params


def get_fillna_params():
    """Вспомогательная функция для получения параметров для операции fillna"""
    fill_method = st.selectbox("Метод заполнения", options=["value", "mean", "median", "mode", "ffill", "bfill"])

    if fill_method == "value":
        fill_value = st.text_input("Значение заполнения")
        # Попытка преобразовать в соответствующий тип
        try:
            if fill_value.lower() in ["true", "false"]:
                fill_value = fill_value.lower() == "true"
            elif "." in fill_value and fill_value.replace(".", "", 1).isdigit():
                fill_value = float(fill_value)
            elif fill_value.isdigit():
                fill_value = int(fill_value)
        except ValueError:
            # Оставить как строку, если преобразование не удалось
            pass
    else:
        # Для других методов мы обработаем их в методе apply
        fill_value = fill_method

    return fill_value


def get_replace_params():
    """Вспомогательная функция для получения параметров для операции замены"""
    replace_values = {}

    st.write("Введите значения для замены (одна пара на строку):")
    st.write("Формат: старое_значение,новое_значение")
    replace_text = st.text_area("Значения замены")

    if replace_text:
        for line in replace_text.strip().split("\n"):
            if "," in line:
                old_val, new_val = line.split(",", 1)
                old_val = old_val.strip()
                new_val = new_val.strip()

                # Попытка преобразовать в соответствующие типы
                old_val = convert_val_to_type(old_val)
                new_val = convert_val_to_type(new_val)

                replace_values[old_val] = new_val

    return replace_values


def get_rename_params(columns):
    """Вспомогательная функция для получения параметров для операции переименования"""
    new_column_names = {}

    st.write("Выберите колонки для переименования:")

    for col in columns:
        new_name = st.text_input(f"Новое имя для '{col}'", value=col)
        if new_name != col:
            new_column_names[col] = new_name

    return new_column_names


def convert_val_to_type(val):
    """Вспомогательная функция для преобразования строкового значения в соответствующий тип"""
    try:
        if val.lower() in ["true", "false"]:
            return val.lower() == "true"
        elif "." in val and val.replace(".", "", 1).isdigit():
            return float(val)
        elif val.isdigit():
            return int(val)
        return val
    except (ValueError, AttributeError):
        # Оставить как есть, если преобразование не удалось или если это не строка
        return val


# Диалог для сохранения CSV
@st.dialog("Сохранить CSV")
def save_csv():
    st.write("Сохранить CSV файл")

    # Получение доступных датафреймов
    df_names = st.session_state.app_state.get_dataframe_names()
    if not df_names:
        st.warning("Нет доступных данных. Пожалуйста, сначала загрузите данные.")
        return

    # Выбор исходного датафрейма
    df_options = list(df_names.items())
    selected_df_id = st.selectbox(
        "Выберите датафрейм для сохранения",
        options=[df_id for df_id, _ in df_options],
        format_func=lambda x: df_names.get(x, "Неизвестный"),
    )

    df = st.session_state.app_state.get_dataframe_by_id(selected_df_id)
    if df is None:
        st.warning("Выбранный датафрейм недоступен.")
        return

    name = st.text_input("Название операции", value=f"Сохранение CSV {uuid4().hex[:4]}")

    # Получение пути к файлу и разделителя
    file_path = st.text_input("Путь к файлу", value=os.path.join("data", f"export_{uuid4().hex[:8]}.csv"))
    separator = get_csv_separator()

    if st.button("Отправить"):
        if not file_path:
            st.error("Пожалуйста, укажите путь к файлу")
            return

        # Создание директории, если она не существует
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Создание операции
        operation = SaveCsvOperation(
            name=name, id=uuid4().hex, file_path=file_path, sep=separator, source_df_id=selected_df_id
        )

        # Применение операции напрямую
        operation.apply(df)
        st.session_state.app_state.add_operation(operation)
        st.success(f"Файл сохранен по пути {file_path}")
        st.rerun()


# Диалог для отображения информации о датафрейме
@st.dialog("Информация о датафрейме")
def show_dataframe_info():
    # Получение исходного датафрейма
    selected_df_id, df = get_source_df("Выберите датафрейм")
    if df is None:
        return

    # Получение информации о датафрейме
    df_info = DataFrameInfo(df)
    info_dict = df_info.get_info()

    # Отображение компонентов информации
    display_basic_info(info_dict)
    display_column_info(info_dict)
    display_numeric_stats(info_dict)
    display_sample_data(df)


def display_basic_info(info_dict):
    """Вспомогательная функция для отображения основной информации о датафрейме"""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Строк", info_dict["shape"]["rows"])
    with col2:
        st.metric("Колонок", info_dict["shape"]["columns"])


def display_column_info(info_dict):
    """Вспомогательная функция для отображения информации о колонках"""
    st.subheader("Колонки")
    col_data = []
    for col in info_dict["columns"]:
        dtype = info_dict["dtypes"][col]
        missing = info_dict["missing_values"][col]
        col_data.append(
            {
                "Колонка": col,
                "Тип": dtype,
                "Пропущено": f"{missing['count']} ({missing['percentage']:.2f}%)",
            }
        )

    st.dataframe(pd.DataFrame(col_data))


def display_numeric_stats(info_dict):
    """Вспомогательная функция для отображения числовой статистики"""
    if info_dict["numeric_stats"]:
        st.subheader("Числовая статистика")

        # Создание датафрейма для каждой статистики
        stats_df = {}
        for col, stats in info_dict["numeric_stats"].items():
            for stat_name, stat_value in stats.items():
                if stat_name not in stats_df:
                    stats_df[stat_name] = {}
                stats_df[stat_name][col] = stat_value

        tabs = st.tabs(list(stats_df.keys()))
        for i, (stat_name, stat_data) in enumerate(stats_df.items()):
            with tabs[i]:
                st.dataframe(pd.Series(stat_data))


def display_sample_data(df):
    """Вспомогательная функция для отображения примера данных"""
    st.subheader("Пример данных")
    st.dataframe(df.head())


def display_eda_tab():
    """Отображение компонентов EDA в первой вкладке"""
    # Отображение компонентов
    st.header("Компоненты анализа")
    if st.session_state.app_state.components:
        for i, component in enumerate(st.session_state.app_state.components):
            component_id = getattr(component, "id", id(component))
            component_name = getattr(component, "name", f"Компонент {i+1}")

            with st.expander(f"{component_name}", expanded=True):
                # Отображение исходного датафрейма, если доступен
                source_df_id = getattr(component, "source_df_id", None)
                if source_df_id:
                    source_name = st.session_state.app_state.dataframe_names.get(source_df_id, "Неизвестный")
                    st.write(f"Исходный датафрейм: {source_name}")

                # Отображение компонента в зависимости от типа
                if hasattr(component, "text"):
                    st.markdown(component.text)
                elif hasattr(component, "chart"):
                    st.plotly_chart(component.chart, use_container_width=True)
                elif hasattr(component, "info_type"):
                    display_data_info(component.info_type, component.source_df_id)

                # Действия с компонентом
                col1, col2 = st.columns(2)
                with col1:
                    if hasattr(component, "text"):
                        if st.button("Редактировать", key=f"edit_comp_{i}"):
                            edit_text(component_id)
                    elif hasattr(component, "chart"):
                        if st.button("Редактировать", key=f"edit_comp_{i}"):
                            edit_chart(component_id)
                    elif hasattr(component, "info_type"):
                        if st.button("Редактировать", key=f"edit_comp_{i}"):
                            edit_data_info_component(component_id)

                with col2:
                    if st.button("Удалить", key=f"delete_comp_{i}"):
                        st.session_state.app_state.delete_component(component_id)
                        st.rerun()
    else:
        st.info("Компоненты еще не добавлены. Используйте боковую панель для добавления компонентов.")


def display_data_operations_tab():
    """Отображение операций с данными во второй вкладке"""
    # Отображение операций
    st.header("Операции с данными")
    if st.session_state.app_state.operations:
        for i, operation in enumerate(st.session_state.app_state.operations):
            with st.expander(f"{operation.name}", expanded=True):
                st.write(f"Тип: {operation.__class__.__name__}")

                # Отображение деталей, специфичных для операции
                if hasattr(operation, "file_path"):
                    st.write(f"Файл: {operation.file_path}")
                    st.write(f"Разделитель: {operation.sep}")
                elif hasattr(operation, "column"):
                    st.write(f"Колонка: {operation.column}")
                    st.write(f"Тип фильтра: {operation.filter_type}")
                    st.write(f"Значение фильтра: {operation.filter_value}")
                    if operation.source_df_id:
                        source_name = st.session_state.app_state.dataframe_names.get(operation.source_df_id, "Неизвестный")
                        st.write(f"Исходный датафрейм: {source_name}")
                elif hasattr(operation, "group_by"):
                    st.write(f"Группировка по: {', '.join(operation.group_by)}")
                    st.write(f"Агрегации: {operation.agg_func}")
                    if operation.source_df_id:
                        source_name = st.session_state.app_state.dataframe_names.get(operation.source_df_id, "Неизвестный")
                        st.write(f"Исходный датафрейм: {source_name}")
                elif hasattr(operation, "clean_type"):
                    st.write(f"Тип очистки: {operation.clean_type}")
                    if operation.columns:
                        st.write(f"Колонки: {', '.join(operation.columns)}")
                    if operation.source_df_id:
                        source_name = st.session_state.app_state.dataframe_names.get(operation.source_df_id, "Неизвестный")
                        st.write(f"Исходный датафрейм: {source_name}")
                    if operation.clean_type == "fillna":
                        st.write(f"Значение заполнения: {operation.fill_value}")
                    elif operation.clean_type == "replace":
                        st.write(f"Значения замены: {operation.replace_values}")
                    elif operation.clean_type == "rename":
                        st.write(f"Новые имена колонок: {operation.new_column_names}")

                # Действия с операцией
                col1, col2 = st.columns(2)
                with col1:
                    # Добавление кнопки редактирования в зависимости от типа операции
                    if hasattr(operation, "column") and operation.operation_type == "filter":
                        if st.button("Редактировать", key=f"edit_op_{i}"):
                            edit_filter_operation(operation.id)
                    elif hasattr(operation, "group_by") and operation.operation_type == "aggregate":
                        if st.button("Редактировать", key=f"edit_op_{i}"):
                            edit_aggregation_operation(operation.id)
                    elif hasattr(operation, "clean_type") and operation.operation_type == "data_clean":
                        if st.button("Редактировать", key=f"edit_op_{i}"):
                            edit_data_cleaning_operation(operation.id)

                with col2:
                    if st.button("Удалить", key=f"delete_op_{i}"):
                        st.session_state.app_state.delete_operation(operation.id)
                        st.rerun()
    else:
        st.info("Операции еще не добавлены. Используйте боковую панель для добавления операций.")


def display_data_tab():
    """Отображение информации о датафрейме и предпросмотра во вкладке данных"""
    # Отображение доступных датафреймов
    st.header("Доступные датафреймы")
    df_names = st.session_state.app_state.get_dataframe_names()
    if df_names:
        df_table = []
        for df_id, df_name in df_names.items():
            df = st.session_state.app_state.get_dataframe_by_id(df_id)
            if df is not None:
                df_table.append(
                    {
                        "Название": df_name,
                        "ID": df_id,
                        "Строк": df.shape[0],
                        "Колонок": df.shape[1],
                        "Текущий": "✓" if df_id == st.session_state.app_state.current_df_id else "",
                    }
                )

        st.dataframe(pd.DataFrame(df_table))
    else:
        st.info("Датафреймы пока недоступны. Загрузите CSV файл, чтобы начать.")

    # Предпросмотр датафрейма с выбором
    st.header("Предпросмотр датафрейма")
    if df_names:
        # Создание списка названий датафреймов для selectbox
        df_options = list(df_names.values())
        df_ids = list(df_names.keys())

        # Добавление selectbox для выбора датафрейма для предпросмотра
        selected_df_name = st.selectbox(
            "Выберите датафрейм для предпросмотра",
            options=df_options,
            index=df_ids.index(st.session_state.app_state.current_df_id)
            if st.session_state.app_state.current_df_id in df_ids
            else 0,
        )

        # Получение ID выбранного датафрейма
        selected_df_id = df_ids[df_options.index(selected_df_name)]

        # Отображение выбранного датафрейма
        df = st.session_state.app_state.get_dataframe_by_id(selected_df_id)
        if df is not None:
            st.dataframe(df)

            # Добавление базовой статистики для выбранного датафрейма
            with st.expander("Статистика датафрейма"):
                st.write("### Основная статистика")
                st.write(f"Размер: {df.shape[0]} строк × {df.shape[1]} колонок")
                st.write("### Статистика числовых колонок")
                st.dataframe(df.describe())

                st.write("### Типы колонок")
                dtypes_df = pd.DataFrame({"Тип данных": [str(dtype) for dtype in df.dtypes]}, index=df.columns)
                dtypes_df.index.name = "Колонка"
                st.dataframe(dtypes_df.reset_index())

                st.write("### Пропущенные значения")
                missing_df = pd.DataFrame(
                    {"Пропущенные значения": df.isna().sum(), "Процент": (df.isna().sum() / len(df) * 100).round(2)}
                )
                missing_df.index.name = "Колонка"
                st.dataframe(missing_df.reset_index())
        else:
            st.warning("Выбранный датафрейм недоступен.")
    else:
        st.info("Датафреймы пока недоступны. Загрузите CSV файл, чтобы начать.")


# Диалог для добавления компонента информации о данных
@st.dialog("Информация о данных")
def add_data_info(component_id=None, edit=False):
    """
    Диалог для добавления или редактирования компонента информации о данных в приложении.

    Аргументы:
        component_id (str, optional): ID компонента для редактирования. По умолчанию None.
        edit (bool, optional): Является ли это операцией редактирования. По умолчанию False.
    """
    dialog_title = "Редактировать компонент информации о данных" if edit else "Добавить компонент информации о данных"
    st.write(dialog_title)

    component = None
    source_df_id = st.session_state.app_state.current_df_id
    info_type = "preview"

    if edit and component_id:
        # Получение компонента для редактирования
        component = get_component_by_id(component_id)
        if not component or not isinstance(component, DataInfoComponent):
            st.error("Компонент информации о данных не найден")
            return

        source_df_id = getattr(component, "source_df_id", st.session_state.app_state.current_df_id)
        info_type = getattr(component, "info_type", "preview")

    # Получение исходного датафрейма
    selected_df_id, df = select_df(source_df_id)
    if df is None:
        return

    # Выбор типа информации для отображения
    info_types = {
        "preview": "Предпросмотр датафрейма",
        "shape": "Размер (строки и колонки)",
        "stats": "Статистика (describe)",
        "types": "Типы колонок",
        "missing": "Пропущенные значения",
        "all": "Вся информация",
    }

    info_type = st.selectbox(
        "Тип информации",
        options=list(info_types.keys()),
        format_func=lambda x: info_types[x],
        index=list(info_types.keys()).index(info_type) if info_type in info_types else 0,
    )

    if st.button("Отправить"):
        if edit and component:
            # Обновление компонента
            component.source_df_id = selected_df_id
            component.info_type = info_type

            # Обновление компонента в состоянии приложения
            st.session_state.app_state.update_component(component_id, component)
            st.rerun()
        else:
            # Создание компонента
            component = DataInfoComponent(
                source_df_id=selected_df_id, info_type=info_type, name=f"Информация о данных: {info_types[info_type]}"
            )

            # Добавление компонента в состояние приложения
            st.session_state.app_state.add_component(component)
            st.rerun()


def edit_data_info_component(component_id):
    """Диалог для редактирования компонента информации о данных"""
    return add_data_info(component_id=component_id, edit=True)


def display_data_info(info_type, source_df_id):
    """Вспомогательная функция для отображения информации о данных"""
    df = st.session_state.app_state.get_dataframe_by_id(source_df_id)
    if df is None:
        st.warning("Выбранный датафрейм недоступен.")
        return

    if info_type == "preview":
        st.dataframe(df.sample(10) if df.shape[0] > 10 else df.head())
    elif info_type == "shape":
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Строк", df.shape[0])
        with c2:
            st.metric("Колонок", df.shape[1])
    elif info_type == "stats":
        st.dataframe(df.describe())
    elif info_type == "types":
        dtypes_df = pd.DataFrame({"Тип данных": [str(dtype) for dtype in df.dtypes]}, index=df.columns)
        dtypes_df.index.name = "Колонка"
        st.dataframe(dtypes_df.reset_index())
    elif info_type == "missing":
        missing_df = pd.DataFrame(
            {"Пропущенные значения": df.isna().sum(), "Процент": (df.isna().sum() / len(df) * 100).round(2)}
        )
        missing_df.index.name = "Колонка"
        st.dataframe(missing_df.reset_index())
    elif info_type == "all":
        display_dataframe_info(df)


def display_dataframe_info(df):
    """Вспомогательная функция для отображения информации о датафрейме"""
    # Отображение основной информации
    st.subheader("Основная информация")
    st.write(f"Размер: {df.shape[0]} строк × {df.shape[1]} колонок")

    # Отображение типов колонок
    st.subheader("Типы колонок")
    dtypes_df = pd.DataFrame({"Тип данных": [str(dtype) for dtype in df.dtypes]}, index=df.columns)
    dtypes_df.index.name = "Колонка"
    st.dataframe(dtypes_df.reset_index())

    # Отображение пропущенных значений
    st.subheader("Пропущенные значения")
    missing_df = pd.DataFrame(
        {"Пропущенные значения": df.isna().sum(), "Процент": (df.isna().sum() / len(df) * 100).round(2)}
    )
    missing_df.index.name = "Колонка"
    st.dataframe(missing_df.reset_index())

    # Отображение числовой статистики
    st.subheader("Числовая статистика")
    st.dataframe(df.describe())

    # Отображение примера данных
    st.subheader("Пример данных")
    st.dataframe(df.sample(10) if df.shape[0] > 10 else df.head())


def display_state_management_tab():
    """Отображение управления состоянием в четвертой вкладке"""
    st.header("Управление состоянием")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Сохранить состояние")
        st.write("Сохраните текущее состояние приложения в файл для последующего использования.")
        
        if st.button("💾 Сохранить состояние", use_container_width=True):
            try:
                os.makedirs("data", exist_ok=True)
                st.session_state.app_state.save_state("data/app_state.json")
                st.success("Состояние успешно сохранено!")
                
                # Показываем информацию о сохраненном состоянии
                if os.path.exists("data/app_state.json"):
                    file_size = os.path.getsize("data/app_state.json")
                    st.info(f"Файл: data/app_state.json ({file_size} байт)")
            except Exception as e:
                st.error(f"Ошибка при сохранении: {e}")
    
    with col2:
        st.subheader("Загрузить состояние")
        st.write("Загрузите ранее сохраненное состояние приложения.")
        
        if os.path.exists("data/app_state.json"):
            file_size = os.path.getsize("data/app_state.json")
            file_time = datetime.fromtimestamp(os.path.getmtime("data/app_state.json"))
            st.info(f"Найдено сохранение: {file_time.strftime('%Y-%m-%d %H:%M:%S')} ({file_size} байт)")
        else:
            st.warning("Сохраненное состояние не найдено")
        
        if st.button("📂 Загрузить состояние", use_container_width=True):
            try:
                if os.path.exists("data/app_state.json"):
                    st.session_state.app_state.load_state("data/app_state.json")
                    st.success("Состояние успешно загружено!")
                    st.rerun()
                else:
                    st.warning("Файл состояния не найден. Сначала сохраните состояние.")
            except Exception as e:
                st.error(f"Ошибка при загрузке: {e}")
    
    st.divider()
    
    # Генерация ноутбука
    st.subheader("📓 Генерация Jupyter Notebook")
    st.write("Создайте Jupyter Notebook с кодом для воспроизведения всех выполненных операций.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        notebook_path = st.text_input("Путь к ноутбуку", value="data_analysis.ipynb")
    
    with col2:
        if st.button("Сгенерировать", use_container_width=True):
            try:
                cells = generate_notebook_cells()
                
                if not cells:
                    st.warning("Нет операций для генерации ноутбука.")
                else:
                    notebook = nbformat.v4.new_notebook()
                    notebook.cells = cells
                    
                    os.makedirs(os.path.dirname(notebook_path) if os.path.dirname(notebook_path) else ".", exist_ok=True)
                    with open(notebook_path, "w", encoding="utf-8") as f:
                        nbformat.write(notebook, f)
                    
                    st.success(f"Ноутбук сохранен: {notebook_path}")
                    
                    with open(notebook_path, "r", encoding="utf-8") as f:
                        notebook_content = f.read()
                    
                    st.download_button(
                        label="📥 Скачать ноутбук",
                        data=notebook_content,
                        file_name=os.path.basename(notebook_path),
                        mime="application/x-ipynb+json",
                    )
            except Exception as e:
                st.error(f"Ошибка: {e}")
    
    st.divider()
    
    # Информация о состоянии
    st.subheader("Текущее состояние")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Операций", len(st.session_state.app_state.operations))
    with col2:
        st.metric("Компонентов", len(st.session_state.app_state.components))
    with col3:
        st.metric("Датафреймов", len(st.session_state.app_state.dataframes))
    
    # Список операций
    if st.session_state.app_state.operations:
        with st.expander("Список операций"):
            for i, op in enumerate(st.session_state.app_state.operations):
                st.write(f"**{i+1}. {op.name}** ({op.operation_type})")
                if hasattr(op, 'source_df_id') and op.source_df_id:
                    source_name = st.session_state.app_state.dataframe_names.get(op.source_df_id, "Неизвестный")
                    st.write(f"   Исходный датафрейм: {source_name}")
    
    # Список датафреймов
    if st.session_state.app_state.dataframe_names:
        with st.expander("Доступные датафреймы"):
            df_info = []
            for df_id, df_name in st.session_state.app_state.dataframe_names.items():
                df = st.session_state.app_state.dataframes.get(df_id)
                if df is not None:
                    df_info.append({
                        "Название": df_name,
                        "Строк": df.shape[0],
                        "Колонок": df.shape[1],
                        "Текущий": "✓" if df_id == st.session_state.app_state.current_df_id else ""
                    })
            st.dataframe(pd.DataFrame(df_info), use_container_width=True)
    
    st.divider()
    
    # Сброс состояния
    st.subheader("Сброс состояния")
    st.warning("Это действие удалит все операции, компоненты и датафреймы.")
    
    if st.button("🗑 Сбросить все состояние", type="primary", use_container_width=True):
        st.session_state.app_state = AppState()
        st.session_state.show_df_info = False
        
        # Очистка дополнительных результатов
        for key in ['correlation_result', 'correlation_name', 
                     'ttest_result', 'ttest_name',
                     'chi_result', 'chi_name']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("Состояние сброшено!")
        st.rerun()

@st.dialog("Корреляционный анализ", width="large")
def add_correlation_analysis():
    st.write("Корреляционный анализ")
    
    selected_df_id, df = select_df(st.session_state.app_state.current_df_id)
    if df is None:
        return
    
    name = st.text_input("Название анализа", value=f"Корреляция {uuid4().hex[:4]}")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Недостаточно числовых колонок для анализа")
        return
    
    columns = st.multiselect("Колонки для анализа", options=numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
    
    method = st.selectbox("Метод корреляции", options=["pearson", "spearman", "kendall"])
    
    if st.button("Выполнить анализ"):
        if len(columns) < 2:
            st.error("Выберите минимум 2 колонки")
            return
        
        analysis = CorrelationAnalysis(
            name=name,
            source_df_id=selected_df_id,
            columns=columns,
            method=method
        )
        
        result = analysis.execute(df)
        
        # Сохраняем результат в сессию
        st.session_state.correlation_result = result
        st.session_state.correlation_name = name
        
        # Добавляем компонент с графиком в app_state
        chart_component = ChartComponent(chart=result["figure"])
        chart_component.name = f"📈 Корреляция: {name}"
        setattr(chart_component, "source_df_id", selected_df_id)
        st.session_state.app_state.add_component(chart_component)
        
        # Добавляем текстовый компонент с результатами
        text = f"### 📈 Корреляционный анализ: {name}\n\n"
        text += f"**Метод:** {method}\n\n"
        
        if result["high_correlations"]:
            text += "**Обнаружены сильные корреляции (|r| > 0.7):**\n\n"
            for corr in result["high_correlations"]:
                text += f"- {corr['var1']} ↔ {corr['var2']}: **{corr['correlation']}**\n"
        else:
            text += "*Сильных корреляций не обнаружено.*\n"
        
        text_component = TextComponent(text=text)
        text_component.name = f"📈 Результаты корреляции: {name}"
        st.session_state.app_state.add_component(text_component)
        
        st.rerun()


@st.dialog("T-тест")
def add_ttest():
    st.write("T-тест для сравнения двух групп")
    
    selected_df_id, df = select_df(st.session_state.app_state.current_df_id)
    if df is None:
        return
    
    name = st.text_input("Название анализа", value=f"T-тест {uuid4().hex[:4]}")
    
    columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not numeric_cols:
        st.warning("Нет числовых колонок для анализа")
        return
    
    if not cat_cols:
        st.warning("Нет категориальных колонок для группировки")
        return
    
    column = st.selectbox("Анализируемая колонка", options=numeric_cols)
    group_column = st.selectbox("Группирующая колонка", options=cat_cols)
    
    group1 = None
    group2 = None
    
    if group_column in df.columns:
        unique_values = df[group_column].dropna().unique()
        if len(unique_values) >= 2:
            group1 = st.selectbox("Группа 1", options=unique_values, index=0)
            group2 = st.selectbox("Группа 2", options=unique_values, index=min(1, len(unique_values)-1))
    
    if st.button("Выполнить тест"):
        if group1 is None or group2 is None:
            st.error("Выберите группы для сравнения")
            return
        
        if group1 == group2:
            st.error("Выберите разные группы для сравнения")
            return
        
        analysis = TTestAnalysis(
            name=name,
            source_df_id=selected_df_id,
            column=column,
            group_column=group_column,
            group1=group1,
            group2=group2
        )
        
        result = analysis.execute(df)
        
        if "error" in result:
            st.error(result["error"])
            return
        
        # Сохраняем результат в сессию
        st.session_state.ttest_result = result
        st.session_state.ttest_name = name
        
        # Добавляем компонент с графиком
        chart_component = ChartComponent(chart=result["figure"])
        chart_component.name = f"🔬 T-тест: {name}"
        setattr(chart_component, "source_df_id", selected_df_id)
        st.session_state.app_state.add_component(chart_component)
        
        # Добавляем текстовый компонент с результатами
        significance = "статистически значимо" if result["significant"] else "не значимо"
        text = f"### 🔬 T-тест: {name}\n\n"
        text += f"**Анализируемая колонка:** {column}\n"
        text += f"**Группирующая колонка:** {group_column}\n\n"
        text += f"**t-статистика:** {result['t_statistic']}\n"
        text += f"**p-value:** {result['p_value']}\n"
        text += f"**Результат:** Различие {significance} (p {'<' if result['significant'] else '>='} 0.05)\n\n"
        text += f"**Группа 1** ({group1}): среднее = {result['group1_stats']['mean']}, N = {result['group1_stats']['size']}\n"
        text += f"**Группа 2** ({group2}): среднее = {result['group2_stats']['mean']}, N = {result['group2_stats']['size']}\n"
        
        text_component = TextComponent(text=text)
        text_component.name = f"🔬 Результаты T-теста: {name}"
        st.session_state.app_state.add_component(text_component)
        
        st.rerun()


@st.dialog("Хи-квадрат тест")
def add_chi_squared():
    st.write("Хи-квадрат тест для категориальных переменных")
    
    selected_df_id, df = select_df(st.session_state.app_state.current_df_id)
    if df is None:
        return
    
    name = st.text_input("Название анализа", value=f"Хи-квадрат {uuid4().hex[:4]}")
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(cat_cols) < 2:
        st.warning("Необходимо минимум 2 категориальные колонки")
        return
    
    col1 = st.selectbox("Первая категориальная колонка", options=cat_cols)
    col2 = st.selectbox("Вторая категориальная колонка", options=[c for c in cat_cols if c != col1])
    
    if st.button("Выполнить тест"):
        if col1 == col2:
            st.error("Выберите разные колонки")
            return
        
        analysis = ChiSquaredTest(
            name=name,
            source_df_id=selected_df_id,
            column1=col1,
            column2=col2
        )
        
        result = analysis.execute(df)
        
        if "error" in result:
            st.error(result["error"])
            return
        
        # Сохраняем результат в сессию
        st.session_state.chi_result = result
        st.session_state.chi_name = name
        
        # Добавляем компонент с графиком
        chart_component = ChartComponent(chart=result["figure"])
        chart_component.name = f"🧪 Хи-квадрат: {name}"
        setattr(chart_component, "source_df_id", selected_df_id)
        st.session_state.app_state.add_component(chart_component)
        
        # Добавляем текстовый компонент с результатами
        significance = "существует значимая связь" if result["significant"] else "связь не значима"
        text = f"### 🧪 Хи-квадрат тест: {name}\n\n"
        text += f"**Переменная 1:** {col1}\n"
        text += f"**Переменная 2:** {col2}\n\n"
        text += f"**χ²:** {result['chi2_statistic']}\n"
        text += f"**p-value:** {result['p_value']}\n"
        text += f"**Степени свободы:** {result['degrees_of_freedom']}\n"
        text += f"**Результат:** {significance} (p {'<' if result['significant'] else '>='} 0.05)\n"
        
        text_component = TextComponent(text=text)
        text_component.name = f"🧪 Результаты Хи-квадрат: {name}"
        st.session_state.app_state.add_component(text_component)
        
        st.rerun()

@st.dialog("Предобработка данных")
def add_preprocessing():
    st.write("Предобработка данных")
    
    selected_df_id, df = select_df(st.session_state.app_state.current_df_id)
    if df is None:
        return
    
    name = st.text_input("Название операции", value=f"Предобработка {uuid4().hex[:4]}")
    
    preprocessing_type = st.selectbox("Тип предобработки", options=[
        "normalize",
        "standardize", 
        "label_encode",
        "one_hot_encode",
        "remove_outliers",
        "binning"
    ], format_func=lambda x: {
        "normalize": "Нормализация (Min-Max)",
        "standardize": "Стандартизация (Z-score)",
        "label_encode": "Label Encoding",
        "one_hot_encode": "One-Hot Encoding",
        "remove_outliers": "Удаление выбросов",
        "binning": "Биннинг"
    }.get(x, x))
    
    columns = []
    if preprocessing_type in ["normalize", "standardize", "binning"]:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        columns = st.multiselect("Колонки", options=numeric_cols)
    elif preprocessing_type in ["label_encode", "one_hot_encode"]:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        columns = st.multiselect("Колонки", options=cat_cols)
    elif preprocessing_type == "remove_outliers":
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        columns = st.multiselect("Колонки", options=numeric_cols)
    
    params = {}
    if preprocessing_type == "remove_outliers":
        params["method"] = st.selectbox("Метод", options=["iqr", "zscore"])
        if params["method"] == "iqr":
            params["threshold"] = st.slider("Порог IQR", 1.0, 3.0, 1.5, 0.1)
        else:
            params["z_threshold"] = st.slider("Z-score порог", 2.0, 5.0, 3.0, 0.5)
    elif preprocessing_type == "binning":
        params["n_bins"] = st.slider("Количество бинов", 2, 20, 5)
    
    if st.button("Применить"):
        if not columns and preprocessing_type not in ["remove_outliers"]:
            st.warning("Будут обработаны все подходящие колонки")
        
        preprocessor = DataPreprocessor(
            name=name,
            preprocessing_type=preprocessing_type,
            columns=columns,
            params=params
        )
        
        result_df = preprocessor.apply(df)
        
        # Сохраняем результат как новую операцию
        from df_operations import DataCleanOperation
        operation = DataCleanOperation(
            name=name,
            id=uuid4().hex,
            clean_type="preprocessing",
            source_df_id=selected_df_id
        )
        
        st.session_state.app_state.operations.append(operation)
        st.session_state.app_state.dataframes[operation.id] = result_df
        st.session_state.app_state.dataframe_names[operation.id] = name
        st.session_state.app_state.current_df_id = operation.id
        
        st.success(f"Предобработка применена! Новый датафрейм: {result_df.shape}")
        st.dataframe(result_df.head())
        
        st.rerun()


@st.dialog("Сгенерировать отчет")
def generate_report():
    st.write("Генерация HTML отчета")
    
    if not st.session_state.app_state.operations:
        st.warning("Нет данных для отчета. Выполните операции с данными.")
        return
    
    report_path = st.text_input("Путь к отчету", value="data_analysis_report.html")
    
    # Убрали выбор формата, оставили только HTML
    
    if st.button("Сгенерировать отчет"):
        try:
            with st.spinner("Генерация отчета..."):
                generator = ReportGenerator(st.session_state.app_state)
                html_content = generator.generate_html_report()
                
                # Сохраняем HTML
                os.makedirs(os.path.dirname(report_path) if os.path.dirname(report_path) else ".", exist_ok=True)
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                
                st.success(f"HTML отчет сохранен: {report_path}")
                
                # Кнопка скачивания
                st.download_button(
                    label="📥 Скачать HTML отчет",
                    data=html_content,
                    file_name=os.path.basename(report_path),
                    mime="text/html"
                )
                
                # Предпросмотр отчета
                with st.expander("Предпросмотр отчета"):
                    st.components.v1.html(html_content, height=600, scrolling=True)
                        
        except Exception as e:
            st.error(f"Ошибка при генерации отчета: {e}")
            import traceback
            st.code(traceback.format_exc())

def display_ai_tab():
    """Отображение вкладки AI-ассистента"""
    st.header("🤖 AI-ассистент (GigaChat)")
    
    # Инициализация ассистента
    if "gigachat_assistant" not in st.session_state:
        st.session_state.gigachat_assistant = None
    
    # Проверка наличия ключа
    has_key = False
    api_key = None
    try:
        api_key = st.secrets.get("gigachat", {}).get("api_key")
        has_key = bool(api_key)
    except:
        pass
    
    # Если ассистент не инициализирован
    if not st.session_state.gigachat_assistant or not st.session_state.gigachat_assistant.client:
        st.info("""
        ### 🔑 Необходим API ключ GigaChat
        
        **Как получить ключ:**
        1. Перейдите на [developers.sber.ru](https://developers.sber.ru/)
        2. Зарегистрируйтесь или войдите
        3. Создайте проект и получите API ключ
        
        **Способы подключения:**
        - Введите ключ ниже
        - Добавьте в файл `.streamlit/secrets.toml`:
        ```toml
        [gigachat]
        api_key = "ваш_ключ"
        ```
        """)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            api_key = st.text_input(
                "API ключ GigaChat",
                type="password",
                value=api_key if has_key else "",
                placeholder="Введите ваш API ключ..."
            )
        with col2:
            st.write("")
            st.write("")
            if st.button("🔌 Подключить", use_container_width=True, type="primary") and api_key:
                with st.spinner("Подключаюсь..."):
                    st.session_state.gigachat_assistant = GigaChatAssistant(api_key=api_key)
                    if st.session_state.gigachat_assistant.client:
                        st.success("✅ Успешно подключено!")
                        st.rerun()
                    else:
                        st.error("❌ Не удалось подключиться. Проверьте ключ.")
        
        # Если ключ был в secrets, пробуем подключиться автоматически
        if has_key and not st.session_state.gigachat_assistant:
            if st.button("🔄 Подключиться автоматически", use_container_width=True):
                with st.spinner("Подключаюсь..."):
                    st.session_state.gigachat_assistant = GigaChatAssistant(api_key=api_key)
                    if st.session_state.gigachat_assistant.client:
                        st.success("✅ Успешно подключено!")
                        st.rerun()
                    else:
                        st.error("❌ Не удалось подключиться")
        
        return
    
    # Ассистент подключен
    assistant = st.session_state.gigachat_assistant
    st.success("✅ AI-ассистент активен")
    
    # Выбор режима работы
    mode = st.radio(
        "Режим работы",
        options=["chat", "analysis", "charts", "explain"],
        format_func=lambda x: {
            "chat": "💬 Чат",
            "analysis": "🔍 Анализ данных",
            "charts": "📊 Подбор графиков",
            "explain": "📝 Объяснение операций"
        }.get(x, x),
        horizontal=True
    )
    
    st.divider()
    
    # Получение списка датафреймов
    df_names = st.session_state.app_state.get_dataframe_names()
    
    # РЕЖИМ: Чат с ассистентом
    if mode == "chat":
        st.subheader("💬 Чат с AI-ассистентом")
        
        # Инициализация истории
        if "ai_chat_history" not in st.session_state:
            st.session_state.ai_chat_history = []
        
        # Отображение истории сообщений
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.ai_chat_history:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(msg["content"])
        
        # Поле ввода
        if prompt := st.chat_input("Задайте вопрос о данных, анализе, визуализации..."):
            # Добавляем сообщение пользователя
            st.session_state.ai_chat_history.append({"role": "user", "content": prompt})
            
            # Собираем контекст из текущих данных
            df_info = {}
            if df_names:
                current_df = st.session_state.app_state.get_dataframe_by_id(
                    st.session_state.app_state.current_df_id
                )
                if current_df is not None:
                    df_info = get_dataframe_info(current_df)
            
            # Получаем ответ от ассистента
            with st.spinner("🤔 Думаю..."):
                response = assistant.generate_data_analysis(df_info, prompt)
            
            # Добавляем ответ
            st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
            st.rerun()
        
        # Кнопка очистки истории
        if st.session_state.ai_chat_history:
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("🗑 Очистить чат"):
                    st.session_state.ai_chat_history = []
                    st.rerun()
    
    # РЕЖИМ: Автоматический анализ
    elif mode == "analysis":
        st.subheader("🔍 Автоматический анализ данных")
        st.write("AI проанализирует данные и предложит рекомендации")
        
        if not df_names:
            st.warning("Нет доступных данных. Загрузите CSV файл.")
            return
        
        # Выбор датафрейма
        selected_df_id = st.selectbox(
            "Выберите датафрейм для анализа",
            options=list(df_names.keys()),
            format_func=lambda x: df_names[x],
            key="ai_analysis_df"
        )
        
        df = st.session_state.app_state.get_dataframe_by_id(selected_df_id)
        
        if df is not None:
            # Показываем краткую информацию
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Строк", df.shape[0])
            with col2:
                st.metric("Колонок", df.shape[1])
            with col3:
                missing = df.isna().sum().sum()
                st.metric("Пропусков", missing)
            
            # Кнопка анализа
            if st.button("🔍 Анализировать данные", use_container_width=True, type="primary"):
                with st.spinner("🔍 Анализирую данные... Это может занять некоторое время"):
                    analysis = assistant.analyze_dataframe(df)
                
                if analysis:
                    st.markdown("### 📊 Результаты анализа")
                    st.markdown(analysis)
                    
                    # Добавление в компоненты
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📌 Сохранить анализ как компонент", use_container_width=True):
                            from components import TextComponent
                            text_component = TextComponent(text=f"### 🤖 AI-анализ данных\n\n{analysis}")
                            text_component.name = f"🤖 AI-анализ: {df_names[selected_df_id]}"
                            st.session_state.app_state.add_component(text_component)
                            st.success("✅ Анализ добавлен в компоненты!")
                    with col2:
                        # Скачивание анализа
                        st.download_button(
                            label="📥 Скачать анализ",
                            data=f"# AI-анализ данных\n\n{analysis}",
                            file_name="ai_analysis.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
    
    # РЕЖИМ: Подбор графиков
    elif mode == "charts":
        st.subheader("📊 Рекомендации по визуализации")
        st.write("AI предложит подходящие типы графиков")
        
        if not df_names:
            st.warning("Нет доступных данных. Загрузите CSV файл.")
            return
        
        # Выбор датафрейма
        selected_df_id = st.selectbox(
            "Выберите датафрейм",
            options=list(df_names.keys()),
            format_func=lambda x: df_names[x],
            key="ai_charts_df"
        )
        
        df = st.session_state.app_state.get_dataframe_by_id(selected_df_id)
        
        if df is not None:
            # Выбор колонок
            st.write("Выберите интересующие колонки (опционально):")
            columns = st.multiselect(
                "Колонки для визуализации",
                options=df.columns.tolist(),
                help="Оставьте пустым для рекомендаций по всем данным"
            )
            
            # Кнопка получения рекомендаций
            if st.button("📊 Получить рекомендации", use_container_width=True, type="primary"):
                with st.spinner("🎨 Подбираю визуализации..."):
                    suggestions = assistant.suggest_charts(df, columns if columns else None)
                
                if suggestions:
                    st.markdown("### 📊 Рекомендованные графики")
                    st.markdown(suggestions)
                    
                    # Быстрые действия
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📌 Добавить в компоненты", use_container_width=True):
                            from components import TextComponent
                            text_component = TextComponent(text=f"### 📊 Рекомендации по графикам\n\n{suggestions}")
                            text_component.name = f"📊 Рекомендации: {df_names[selected_df_id]}"
                            st.session_state.app_state.add_component(text_component)
                            st.success("✅ Рекомендации добавлены!")
    
    # РЕЖИМ: Объяснение операций
    elif mode == "explain":
        st.subheader("📝 Объяснение операций с данными")
        st.write("AI объяснит, что делает выбранная операция")
        
        if not st.session_state.app_state.operations:
            st.warning("Нет выполненных операций. Выполните операции с данными.")
            return
        
        # Список операций
        operation_options = {
            op.id: op for op in st.session_state.app_state.operations
        }
        
        selected_op_id = st.selectbox(
            "Выберите операцию для объяснения",
            options=list(operation_options.keys()),
            format_func=lambda x: f"{operation_options[x].operation_type}: {operation_options[x].name}"
        )
        
        if selected_op_id:
            operation = operation_options[selected_op_id]
            
            # Показываем информацию об операции
            with st.expander("📋 Информация об операции"):
                st.write(f"Тип: {operation.operation_type}")
                st.write(f"Название: {operation.name}")
                
                if hasattr(operation, 'column'):
                    st.write(f"Колонка: {operation.column}")
                    st.write(f"Тип фильтра: {getattr(operation, 'filter_type', 'N/A')}")
                if hasattr(operation, 'group_by'):
                    st.write(f"Группировка: {', '.join(operation.group_by)}")
                    st.write(f"Агрегации: {getattr(operation, 'agg_func', {})}")
                if hasattr(operation, 'clean_type'):
                    st.write(f"Тип очистки: {operation.clean_type}")
                if hasattr(operation, 'columns') and operation.columns:
                    st.write(f"Колонки: {', '.join(operation.columns)}")
            
            # Кнопка объяснения
            if st.button("📝 Объяснить операцию", use_container_width=True, type="primary"):
                # Получаем датафрейм
                source_df_id = getattr(operation, 'source_df_id', None)
                df = st.session_state.app_state.get_dataframe_by_id(source_df_id) if source_df_id else None
                
                if df is not None:
                    df_info = get_dataframe_info(df)
                    
                    with st.spinner("📝 Объясняю операцию..."):
                        # Собираем параметры
                        params = {}
                        if hasattr(operation, 'column'):
                            params['column'] = operation.column
                            params['filter_type'] = getattr(operation, 'filter_type', '')
                            params['filter_value'] = getattr(operation, 'filter_value', '')
                        if hasattr(operation, 'group_by'):
                            params['group_by'] = operation.group_by
                            params['agg_func'] = getattr(operation, 'agg_func', {})
                        if hasattr(operation, 'clean_type'):
                            params['clean_type'] = operation.clean_type
                        if hasattr(operation, 'fill_value'):
                            params['fill_value'] = operation.fill_value
                        if hasattr(operation, 'replace_values'):
                            params['replace_values'] = operation.replace_values
                        
                        explanation = assistant.explain_operation(
                            df_info,
                            operation.operation_type,
                            params
                        )
                    
                    if explanation:
                        st.markdown("### 📝 Объяснение")
                        st.markdown(explanation)
                        
                        if st.button("📌 Добавить объяснение в компоненты"):
                            from components import TextComponent
                            text_component = TextComponent(
                                text=f"### 📝 Объяснение операции: {operation.name}\n\n{explanation}"
                            )
                            text_component.name = f"📝 Объяснение: {operation.name}"
                            st.session_state.app_state.add_component(text_component)
                            st.success("✅ Объяснение добавлено!")
                else:
                    st.warning("Не удалось получить исходный датафрейм для операции")
    
    # Отключение ассистента
    st.divider()
    with st.expander("⚙️ Настройки"):
        if st.button("🔌 Отключить ассистента"):
            st.session_state.gigachat_assistant = None
            if "ai_chat_history" in st.session_state:
                del st.session_state.ai_chat_history
            st.rerun()

def main():
    # Основной макет приложения
    st.title("Система исследования данных")

    # Боковая панель для операций
    with st.sidebar:
        st.header("Операции")

        # Раздел операций с данными
        st.subheader("Операции с данными")

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("📂", help="Загрузить CSV", key="load_csv_btn"):
                load_csv()
        with c2:
            if st.button("📊", help="Информация о датафрейме", key="df_info_btn"):
                show_dataframe_info()
        with c3:
            if st.button("💾", help="Сохранить CSV", key="save_csv_btn"):
                save_csv()

        # Раздел преобразований данных
        st.subheader("Преобразования данных")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if st.button("🔍", help="Добавить фильтр", key="add_filter_btn"):
                add_filter()
        with c2:
            if st.button("🔄", help="Добавить агрегацию", key="add_agg_btn"):
                add_aggregation()
        with c3:
            if st.button("🧹", help="Добавить очистку данных", key="add_clean_btn"):
                add_data_cleaning()
                
        with c4:
            if st.button("⚙️", help="Предобработка данных", key="preprocess_btn"):
                add_preprocessing()
                
        # Раздел компонентов EDA
        st.subheader("Компоненты EDA")

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("📝", help="Добавить текст", key="add_text_btn"):
                add_text()
        with c2:
            if st.button("📈", help="Добавить график", key="add_chart_btn"):
                add_chart()
        with c3:
            if st.button("📊", help="Добавить информацию о данных", key="add_data_info_btn"):
                add_data_info()

        # Раздел статистического анализа
        st.subheader("Статистический анализ")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            if st.button("📈", help="Корреляционный анализ", key="corr_btn"):
                add_correlation_analysis()
        with c2:
            if st.button("🔬", help="T-тест", key="ttest_btn"):
                add_ttest()
        with c3:
            if st.button("🧪", help="Хи-квадрат", key="chi_btn"):
                add_chi_squared()

        # Раздел управления состоянием
        st.subheader("Управление состоянием")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if st.button("💾", help="Сохранить состояние", key="save_state_btn"):
                save_state()
        with c2:
            if st.button("📂", help="Загрузить состояние", key="load_state_btn"):
                load_state()
        with c3:
            if st.button("📓", help="Сгенерировать ноутбук", key="gen_notebook_btn"):
                generate_notebook()

        with c4:
            if st.button("📄", help="Сгенерировать отчет", key="report_btn"):
                generate_report()
    
    # Основное содержимое
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Анализ", 
        "Операции с данными", 
        "Предпросмотр данных", 
        "Управление состоянием",
        "🤖 AI-ассистент"
    ])

    with tab1:
        display_eda_tab()

    with tab2:
        display_data_operations_tab()

    with tab3:
        display_data_tab()

    with tab4:
        display_state_management_tab()
    
    with tab5:
        display_ai_tab()


if __name__ == "__main__":
    main()