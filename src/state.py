import os
import json
import pandas as pd
import nbformat as nbf
import plotly.graph_objects as go
import plotly.io as pio
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field, asdict
from components import BaseComponent, TextComponent, ChartComponent, DataInfoComponent
from df_operations import BaseDfOperation, LoadCsvOperation, FilterOperation, AggregateOperation, DataCleanOperation

@dataclass
class AppState:
    """Класс для управления состоянием приложения"""
    components: List[BaseComponent] = field(default_factory=list)
    operations: List[BaseDfOperation] = field(default_factory=list)
    dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)
    current_df_id: Optional[str] = None
    dataframe_names: Dict[str, str] = field(default_factory=dict)
    original_dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def add_component(self, component: BaseComponent) -> None:
        """Добавление компонента в состояние приложения"""
        self.components.append(component)

    def update_component(self, component_id: str, component: BaseComponent) -> None:
        """Обновление компонента в состоянии приложения"""
        for i, comp in enumerate(self.components):
            if getattr(comp, "id", id(comp)) == component_id:
                self.components[i] = component
                break

    def delete_component(self, component_id: str) -> None:
        """Удаление компонента из состояния приложения"""
        self.components = [comp for comp in self.components if getattr(comp, "id", id(comp)) != component_id]

    def add_operation(self, operation: BaseDfOperation) -> None:
        """Добавление операции в состояние приложения и её выполнение"""
        self.operations.append(operation)

        # Выполнение операции
        if operation.operation_type == "load_csv":
            # Для операций загрузки создаем новый датафрейм
            df = operation.apply(None)
            self.dataframes[operation.id] = df
            self.original_dataframes[operation.id] = df.copy()
            self.dataframe_names[operation.id] = operation.name
            self.current_df_id = operation.id
        else:
            # Для других операций применяем к соответствующему датафрейму
            source_df_id = operation.source_df_id or self.current_df_id
            if source_df_id in self.dataframes:
                # Получаем исходный датафрейм для работы
                source_df = self.dataframes.get(source_df_id, pd.DataFrame())
                
                # Применяем текущую операцию к копии исходного датафрейма
                result_df = operation.apply(source_df.copy())
                
                # Сохраняем результат
                self.dataframes[operation.id] = result_df
                self.original_dataframes[operation.id] = source_df.copy()  # Сохраняем оригинал как ссылку
                self.dataframe_names[operation.id] = operation.name
                self.current_df_id = operation.id

    def delete_operation(self, operation_id: str) -> None:
        """Удаление операции из состояния приложения"""
        # Удаляем операцию
        self.operations = [op for op in self.operations if op.id != operation_id]

        # Удаляем датафрейм, если он существует
        if operation_id in self.dataframes:
            del self.dataframes[operation_id]

        if operation_id in self.original_dataframes:
            del self.original_dataframes[operation_id]

        if operation_id in self.dataframe_names:
            del self.dataframe_names[operation_id]

        # Обновляем current_df_id при необходимости
        if self.current_df_id == operation_id:
            if self.operations:
                # Устанавливаем текущим последнюю операцию
                self.current_df_id = self.operations[-1].id
            else:
                self.current_df_id = None

    def update_operation(self, operation_id: str, operation: BaseDfOperation) -> None:
        """Обновление операции в состоянии приложения и её повторное выполнение"""
        # Находим и обновляем операцию
        for i, op in enumerate(self.operations):
            if op.id == operation_id:
                self.operations[i] = operation
                break
        else:
            # Операция не найдена
            return

        # Повторно выполняем операцию
        if operation.operation_type == "load_csv":
            # Для операций загрузки создаем новый датафрейм
            df = operation.apply(None)
            self.dataframes[operation.id] = df
            self.original_dataframes[operation.id] = df.copy()
            self.dataframe_names[operation.id] = operation.name
        else:
            # Для других операций применяем к соответствующему датафрейму
            source_df_id = operation.source_df_id or self.current_df_id
            if source_df_id in self.dataframes:
                # Получаем исходный датафрейм для работы
                source_df = self.dataframes.get(source_df_id, pd.DataFrame())
                
                # Применяем текущую операцию к копии исходного датафрейма
                result_df = operation.apply(source_df.copy())
                
                # Сохраняем результат
                self.dataframes[operation.id] = result_df
                self.original_dataframes[operation.id] = source_df.copy()  # Сохраняем оригинал как ссылку
                self.dataframe_names[operation.id] = operation.name

        # Обновляем зависимые операции
        self._update_dependent_operations(operation_id)

    def _update_dependent_operations(self, parent_operation_id: str) -> None:
        """Обновление операций, которые зависят от данной операции"""
        # Находим операции, которые используют датафрейм этой операции как исходный
        dependent_ops = [op for op in self.operations if op.source_df_id == parent_operation_id]
        
        # Обновляем каждую зависимую операцию
        for op in dependent_ops:
            if parent_operation_id in self.dataframes:
                # Получаем исходный датафрейм
                source_df = self.dataframes.get(parent_operation_id, pd.DataFrame())
                
                # Применяем операцию
                result_df = op.apply(source_df.copy())
                
                # Обновляем результат
                self.dataframes[op.id] = result_df
                self.original_dataframes[op.id] = source_df.copy()  # Сохраняем оригинал как ссылку
                
                # Рекурсивно обновляем операции, которые зависят от этой
                self._update_dependent_operations(op.id)

    def get_current_dataframe(self) -> Optional[pd.DataFrame]:
        """Получение текущего датафрейма"""
        if self.current_df_id and self.current_df_id in self.dataframes:
            return self.dataframes[self.current_df_id]
        return None

    def get_dataframe_by_id(self, df_id: str) -> Optional[pd.DataFrame]:
        """Получение датафрейма по его ID"""
        if df_id in self.dataframes:
            return self.dataframes[df_id]
        return None

    def get_dataframe_names(self) -> Dict[str, str]:
        """Получение словаря ID датафреймов и их названий"""
        return self.dataframe_names

    def save_state(self, file_path: str) -> None:
        """Сохранение состояния приложения в JSON файл"""
        state_dict: Dict[str, Any] = {
            "components": [self._component_to_dict(comp) for comp in self.components],
            "operations": [op.to_dict() for op in self.operations],
            "current_df_id": self.current_df_id,
            "dataframe_names": self.dataframe_names,
        }

        with open(file_path, "w") as f:
            json.dump(state_dict, f, indent=2, default=self._json_serializer)

    def _json_serializer(self, obj):
        """Пользовательский сериализатор для JSON"""
        if isinstance(obj, pd.DataFrame):
            return None  # Пропускаем датафреймы
        elif isinstance(obj, (go.Figure,)):
            # Конвертируем фигуру Plotly в JSON-совместимый словарь
            return pio.to_json(obj)
        elif isinstance(obj, Enum):
            return obj.name
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _component_to_dict(self, component: BaseComponent) -> Dict[str, Any]:
        """Преобразование компонента в словарь со всеми дополнительными атрибутами"""
        base_dict = component.to_dict()
        
        # Добавляем все дополнительные атрибуты, которые есть у компонента
        extra_attrs = {}
        for attr_name in dir(component):
            if not attr_name.startswith('_') and attr_name not in base_dict:
                attr_value = getattr(component, attr_name)
                # Пропускаем методы
                if callable(attr_value):
                    continue
                
                try:
                    # Проверяем, можно ли сериализовать в JSON
                    if isinstance(attr_value, (go.Figure,)):
                        extra_attrs[attr_name] = pio.to_json(attr_value)
                    elif isinstance(attr_value, pd.DataFrame):
                        continue  # Пропускаем датафреймы
                    else:
                        json.dumps(attr_value, default=self._json_serializer)
                        extra_attrs[attr_name] = attr_value
                except (TypeError, OverflowError):
                    continue
        
        base_dict.update(extra_attrs)
        return base_dict

    def load_state(self, file_path: str) -> None:
        """Загрузка состояния приложения из JSON файла"""
        if not os.path.exists(file_path):
            return

        with open(file_path, "r") as f:
            state_dict: Dict[str, Any] = json.load(f)

        # Очистка текущего состояния
        self.components = []
        self.operations = []
        self.dataframes = {}
        self.original_dataframes = {}
        self.dataframe_names = {}

        # Загрузка компонентов
        for comp_dict in state_dict.get("components", []):
            component_type = comp_dict.get("component_type")
            
            if component_type == "CHART" or component_type == "chart":
                # Извлекаем только базовые атрибуты для создания компонента
                base_keys = {'component_type', 'id', 'name', 'chart'}
                extra_attrs = {}
                
                # Разделяем базовые и дополнительные атрибуты
                comp_data = {}
                for key, value in comp_dict.items():
                    if key in base_keys:
                        # Обрабатываем chart отдельно
                        if key == 'chart' and isinstance(value, str):
                            try:
                                value = pio.from_json(value)
                            except:
                                value = None
                        comp_data[key] = value
                    else:
                        # Сохраняем дополнительные атрибуты
                        if isinstance(value, str) and (value.startswith('{"data":') or value.startswith('{"layout":')):
                            try:
                                value = pio.from_json(value)
                            except:
                                pass
                        extra_attrs[key] = value
                
                # Создаем компонент только с базовыми атрибутами
                comp = ChartComponent.from_dict(comp_data)
                
                # Добавляем дополнительные атрибуты
                for key, value in extra_attrs.items():
                    setattr(comp, key, value)
                
                self.components.append(comp)
                
            elif component_type == "TEXT" or component_type == "text":
                # Для текстового компонента тоже разделяем атрибуты
                base_keys = {'component_type', 'id', 'name', 'text'}
                text_data = {k: v for k, v in comp_dict.items() if k in base_keys}
                extra_attrs = {k: v for k, v in comp_dict.items() if k not in base_keys}
                
                comp = TextComponent.from_dict(text_data)
                
                # Добавляем дополнительные атрибуты
                for key, value in extra_attrs.items():
                    setattr(comp, key, value)
                
                self.components.append(comp)
                
            elif component_type == "DATA" or component_type == "data":
                comp = DataInfoComponent(
                    source_df_id=comp_dict.get("source_df_id", ""),
                    info_type=comp_dict.get("info_type", "preview"),
                    id=comp_dict.get("id"),
                    name=comp_dict.get("name")
                )
                self.components.append(comp)

        # Загрузка операций
        for op_dict in state_dict.get("operations", []):
            op_type = op_dict.get("operation_type")
            if op_type == "load_csv":
                op = LoadCsvOperation.from_dict(op_dict)
            elif op_type == "filter":
                op = FilterOperation.from_dict(op_dict)
            elif op_type == "aggregate":
                op = AggregateOperation.from_dict(op_dict)
            elif op_type == "data_clean":
                op = DataCleanOperation.from_dict(op_dict)
            else:
                continue

            self.operations.append(op)

        # Установка текущего ID датафрейма
        self.current_df_id = state_dict.get("current_df_id")

        # Загрузка названий датафреймов
        self.dataframe_names = state_dict.get("dataframe_names", {})

        # Выполнение операций для восстановления датафреймов
        for op in self.operations:
            if op.operation_type == "load_csv":
                try:
                    df = op.apply(None)
                    self.dataframes[op.id] = df
                    self.original_dataframes[op.id] = df.copy()
                except Exception as e:
                    print(f"Error loading CSV operation {op.id}: {e}")
            else:
                source_df_id = op.source_df_id or self.current_df_id
                if source_df_id in self.original_dataframes:
                    source_df = self.original_dataframes[source_df_id]
                    try:
                        result_df = op.apply(source_df.copy())
                        self.dataframes[op.id] = result_df
                        self.original_dataframes[op.id] = source_df.copy()
                    except Exception as e:
                        print(f"Error applying operation {op.id}: {e}")

    def generate_notebook(self, file_path: str) -> None:
        """Генерация Jupyter notebook из состояния приложения"""
        nb = nbf.v4.new_notebook()
        cells = []

        # Добавление импортов
        cells.append(nbf.v4.new_markdown_cell("# Ноутбук анализа данных\n\nСгенерировано из SEDES"))
        cells.append(nbf.v4.new_code_cell("import pandas as pd\nimport plotly.express as px"))

        # Добавление операций
        for op in self.operations:
            if op.operation_type == "load_csv":
                cells.append(nbf.v4.new_markdown_cell(f"## Загрузка данных: {op.name}"))
                separator_repr = repr(op.sep)
                cells.append(
                    nbf.v4.new_code_cell(
                        f"df_{op.id} = pd.read_csv('{op.file_path}', sep={separator_repr})\ndf_{op.id}.head()"
                    )
                )
            elif op.operation_type == "filter":
                cells.append(nbf.v4.new_markdown_cell(f"## Фильтрация данных: {op.name}"))

                source_df_id = op.source_df_id or self.current_df_id
                filter_code = f"df_{op.id} = df_{source_df_id}.copy()\n"

                if op.filter_type == "equals":
                    filter_code += f"df_{op.id} = df_{op.id}[df_{op.id}['{op.column}'] == {repr(op.filter_value)}]"
                elif op.filter_type == "contains":
                    filter_code += f"df_{op.id} = df_{op.id}[df_{op.id}['{op.column}'].astype(str).str.contains({repr(str(op.filter_value))}, na=False)]"
                elif op.filter_type == "greater_than":
                    filter_code += f"df_{op.id} = df_{op.id}[df_{op.id}['{op.column}'] > {op.filter_value}]"
                elif op.filter_type == "less_than":
                    filter_code += f"df_{op.id} = df_{op.id}[df_{op.id}['{op.column}'] < {op.filter_value}]"
                elif op.filter_type == "between" and isinstance(op.filter_value, list) and len(op.filter_value) == 2:
                    filter_code += f"df_{op.id} = df_{op.id}[(df_{op.id}['{op.column}'] >= {op.filter_value[0]}) & (df_{op.id}['{op.column}'] <= {op.filter_value[1]})]"

                filter_code += f"\ndf_{op.id}.head()"
                cells.append(nbf.v4.new_code_cell(filter_code))
            elif op.operation_type == "aggregate":
                cells.append(nbf.v4.new_markdown_cell(f"## Агрегация данных: {op.name}"))

                source_df_id = op.source_df_id or self.current_df_id
                agg_code = f"df_{op.id} = df_{source_df_id}.copy()\n"

                group_cols = ", ".join([f"'{col}'" for col in op.group_by_columns])
                agg_dict = {col: func for col, func in op.agg_functions.items()}

                agg_code += f"df_{op.id} = df_{op.id}.groupby([{group_cols}]).agg({agg_dict}).reset_index()\n"
                agg_code += f"df_{op.id}.head()"
                cells.append(nbf.v4.new_code_cell(agg_code))
            elif op.operation_type == "data_clean":
                cells.append(nbf.v4.new_markdown_cell(f"## Очистка данных: {op.name}"))

                source_df_id = op.source_df_id or self.current_df_id
                clean_code = f"df_{op.id} = df_{source_df_id}.copy()\n"

                if op.clean_type == "dropna":
                    if op.columns:
                        cols_str = ", ".join([f"'{col}'" for col in op.columns])
                        clean_code += f"df_{op.id} = df_{op.id}.dropna(subset=[{cols_str}])"
                    else:
                        clean_code += f"df_{op.id} = df_{op.id}.dropna()"

                elif op.clean_type == "fillna":
                    if op.columns:
                        if op.fill_value in ["mean", "median", "mode", "ffill", "bfill"]:
                            # Специальные методы заполнения
                            for col in op.columns:
                                if op.fill_value == "mean":
                                    clean_code += f"df_{op.id}[{repr(col)}] = df_{op.id}[{repr(col)}].fillna(df_{op.id}[{repr(col)}].mean())\n"
                                elif op.fill_value == "median":
                                    clean_code += f"df_{op.id}[{repr(col)}] = df_{op.id}[{repr(col)}].fillna(df_{op.id}[{repr(col)}].median())\n"
                                elif op.fill_value == "mode":
                                    clean_code += f"mode_val = df_{op.id}[{repr(col)}].mode()\n"
                                    clean_code += f"if not mode_val.empty:\n"
                                    clean_code += (
                                        f"    df_{op.id}[{repr(col)}] = df_{op.id}[{repr(col)}].fillna(mode_val[0])\n"
                                    )
                                elif op.fill_value == "ffill":
                                    clean_code += f"df_{op.id}[{repr(col)}] = df_{op.id}[{repr(col)}].ffill()\n"
                                elif op.fill_value == "bfill":
                                    clean_code += f"df_{op.id}[{repr(col)}] = df_{op.id}[{repr(col)}].bfill()\n"
                        else:
                            # Заполнение конкретным значением
                            cols_str = ", ".join([f"'{col}'" for col in op.columns])
                            clean_code += (
                                f"df_{op.id}[[{cols_str}]] = df_{op.id}[[{cols_str}]].fillna({repr(op.fill_value)})"
                            )
                    else:
                        # Заполнение всех колонок
                        if op.fill_value in ["mean", "median", "mode"]:
                            if op.fill_value == "mean":
                                clean_code += "# Заполнение числовых колонок средним значением\n"
                                clean_code += f"for col in df_{op.id}.select_dtypes(include=['number']).columns:\n"
                                clean_code += f"    df_{op.id}[col] = df_{op.id}[col].fillna(df_{op.id}[col].mean())\n"
                            elif op.fill_value == "median":
                                clean_code += "# Заполнение числовых колонок медианой\n"
                                clean_code += f"for col in df_{op.id}.select_dtypes(include=['number']).columns:\n"
                                clean_code += (
                                    f"    df_{op.id}[col] = df_{op.id}[col].fillna(df_{op.id}[col].median())\n"
                                )
                            elif op.fill_value == "mode":
                                clean_code += "# Заполнение колонок модой\n"
                                clean_code += f"for col in df_{op.id}.columns:\n"
                                clean_code += f"    mode_val = df_{op.id}[col].mode()\n"
                                clean_code += f"    if not mode_val.empty:\n"
                                clean_code += f"        df_{op.id}[col] = df_{op.id}[col].fillna(mode_val[0])\n"
                        elif op.fill_value == "ffill":
                            clean_code += f"df_{op.id} = df_{op.id}.ffill()"
                        elif op.fill_value == "bfill":
                            clean_code += f"df_{op.id} = df_{op.id}.bfill()"
                        else:
                            clean_code += f"df_{op.id} = df_{op.id}.fillna({repr(op.fill_value)})"

                elif op.clean_type == "drop_duplicates":
                    if op.columns:
                        cols_str = ", ".join([f"'{col}'" for col in op.columns])
                        clean_code += f"df_{op.id} = df_{op.id}.drop_duplicates(subset=[{cols_str}])"
                    else:
                        clean_code += f"df_{op.id} = df_{op.id}.drop_duplicates()"

                elif op.clean_type == "replace":
                    if op.columns and op.replace_values:
                        for col in op.columns:
                            clean_code += f"df_{op.id}[{repr(col)}] = df_{op.id}[{repr(col)}].replace({repr(op.replace_values)})\n"
                    elif op.replace_values:
                        clean_code += f"df_{op.id} = df_{op.id}.replace({repr(op.replace_values)})"

                elif op.clean_type == "rename":
                    if op.new_column_names:
                        clean_code += f"df_{op.id} = df_{op.id}.rename(columns={repr(op.new_column_names)})"

                clean_code += f"\ndf_{op.id}.head()"
                cells.append(nbf.v4.new_code_cell(clean_code))

        # Добавление компонентов
        for i, comp in enumerate(self.components):
            if hasattr(comp, "text"):
                cells.append(nbf.v4.new_markdown_cell(comp.text))
            elif hasattr(comp, "chart"):
                cells.append(nbf.v4.new_markdown_cell(f"## График {i+1}"))

                # Получение ID исходного датафрейма
                source_df_id = getattr(comp, "source_df_id", self.current_df_id)
                if not source_df_id:
                    continue

                # Получение типа графика и параметров
                chart_type = comp.chart.__class__.__name__
                chart_code = ""

                if chart_type == "LineChart":
                    chart_code = f"px.line(df_{source_df_id}, x='{comp.chart.x_column}', y='{comp.chart.y_column}'"
                    if hasattr(comp.chart, "hue_column") and comp.chart.hue_column:
                        chart_code += f", color='{comp.chart.hue_column}'"
                    chart_code += f", title='{comp.chart.title}')"
                elif chart_type == "BarChart":
                    chart_code = f"px.bar(df_{source_df_id}, x='{comp.chart.x_column}', y='{comp.chart.y_column}'"
                    if hasattr(comp.chart, "group_by") and comp.chart.group_by:
                        chart_code += f", color='{comp.chart.group_by}'"
                    chart_code += f", title='{comp.chart.title}')"
                elif chart_type == "ScatterChart":
                    chart_code = f"px.scatter(df_{source_df_id}, x='{comp.chart.x_column}', y='{comp.chart.y_column}'"
                    if hasattr(comp.chart, "hue_column") and comp.chart.hue_column:
                        chart_code += f", color='{comp.chart.hue_column}'"
                    if hasattr(comp.chart, "size_column") and comp.chart.size_column:
                        chart_code += f", size='{comp.chart.size_column}'"
                    chart_code += f", title='{comp.chart.title}')"
                elif chart_type == "Histogram":
                    chart_code = f"px.histogram(df_{source_df_id}, x='{comp.chart.x_column}'"
                    if hasattr(comp.chart, "bins") and comp.chart.bins:
                        chart_code += f", nbins={comp.chart.bins}"
                    if hasattr(comp.chart, "color") and comp.chart.color:
                        chart_code += f", color='{comp.chart.color}'"
                    chart_code += f", title='{comp.chart.title}')"

                if chart_code:
                    cells.append(nbf.v4.new_code_cell(f"fig = {chart_code}\nfig.show()"))

        nb.cells = cells

        with open(file_path, "w") as f:
            nbf.write(nb, f)