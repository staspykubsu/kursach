import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Сохраняем для подсказок типов Figure
from typing import Dict, List, Any, Protocol
from dataclasses import dataclass, asdict, field
from uuid import uuid4


class Chart(Protocol):
    """Протокол для объектов графиков"""

    def plot(self, df: pd.DataFrame): ...

    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any], df: pd.DataFrame = None) -> "Chart":
        """
        Создает объект Chart из сериализованного словаря.

        Аргументы:
            data (Dict[str, Any]): Сериализованные данные графика.
            df (pd.DataFrame, optional): Датафрейм для использования с графиком. По умолчанию None.

        Возвращает:
            Chart: Созданный объект Chart.
        """
        ...


@dataclass
class BaseChart:
    df: pd.DataFrame
    id: str = uuid4().hex
    chart_type: str = ""
    # name: str = ""
    x_column: str = ""
    y_column: str = ""
    title: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        # Сериализуем только конфигурацию, без датафрейма
        data = asdict(self)
        data.pop("df", None)  # Удаляем датафрейм из сериализации
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], df: pd.DataFrame = None) -> "BaseChart":
        chart_map = {
            "line": LineChart,
            "bar": BarChart,
            "scatter": ScatterChart,
            "histogram": Histogram,
            "box": BoxPlot,
            "violin": ViolinPlot,
            "heatmap": Heatmap,
            "area": AreaChart,
            "funnel": FunnelChart,
        }
        chart_type = data.pop("chart_type", "")
        if df is not None:
            data["df"] = df
        if chart_type in chart_map:
            return chart_map[chart_type](**data)
        return cls(**data)

    @classmethod
    def _get_name(cls) -> str:
        return cls.__name__


@dataclass
class LineChart(BaseChart):
    """Линейный график"""
    chart_type: str = "line"
    hue_column: str = ""

    def plot(self) -> px.line:
        # Строим словарь параметров только с непустыми значениями
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Добавляем цвет, только если он задан
        if self.hue_column:
            params["color"] = self.hue_column

        # Добавляем любые дополнительные параметры
        params.update(self.params)

        fig = px.line(self.df, **params)
        return fig


@dataclass
class BarChart(BaseChart):
    """Столбчатая диаграмма"""
    chart_type: str = "bar"
    group_by: str = ""

    def plot(self) -> px.bar:
        # Группируем данные соответствующим образом
        if self.group_by:
            grouped = self.df.groupby([self.x_column, self.group_by])[self.y_column].mean().reset_index()
        else:
            grouped = self.df.groupby([self.x_column])[self.y_column].mean().reset_index()

        # Строим словарь параметров только с непустыми значениями
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Добавляем цвет, только если задана группировка
        if self.group_by:
            params["color"] = self.group_by

        # Добавляем любые дополнительные параметры
        params.update(self.params)

        fig = px.bar(grouped, **params)
        return fig


@dataclass
class Histogram(BaseChart):
    """Гистограмма"""
    chart_type: str = "histogram"
    color: str = ""
    bins: int = 10

    def plot(self) -> px.histogram:
        # Строим словарь параметров только с непустыми значениями
        params = {
            "x": self.x_column,
            "nbins": self.bins,
            "title": self.title if self.title else None,
        }

        # Добавляем цвет, только если он задан
        if self.color:
            params["color"] = self.color

        # Добавляем любые дополнительные параметры
        params.update(self.params)

        fig = px.histogram(self.df, **params)
        return fig


@dataclass
class ScatterChart(BaseChart):
    """Диаграмма рассеяния"""
    chart_type: str = "scatter"
    hue_column: str = ""
    size_column: str = ""

    def plot(self) -> px.scatter:
        # Строим словарь параметров только с непустыми значениями
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Добавляем цвет и размер, только если они заданы
        if self.hue_column:
            params["color"] = self.hue_column
        if self.size_column:
            params["size"] = self.size_column

        # Добавляем любые дополнительные параметры
        params.update(self.params)

        fig = px.scatter(self.df, **params)
        return fig


@dataclass
class PieChart(BaseChart):
    """Круговая диаграмма"""
    chart_type: str = "pie"
    group_by: str = ""
    values_column: str = ""

    def plot(self) -> px.pie:
        # Строим словарь параметров только с непустыми значениями
        params = {
            "names": self.group_by,
            "title": self.title if self.title else None,
        }
        
        # Добавляем колонку значений, если указана
        if self.values_column:
            params["values"] = self.values_column

        # Добавляем любые дополнительные параметры
        params.update(self.params)

        fig = px.pie(self.df, **params)
        return fig


@dataclass
class BoxPlot(BaseChart):
    """Ящик с усами"""
    chart_type: str = "boxplot"
    color: str = ""

    def plot(self) -> px.box:
        # Строим словарь параметров только с непустыми значениями
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Добавляем цвет, только если он задан
        if self.color:
            params["color"] = self.color

        # Добавляем любые дополнительные параметры
        params.update(self.params)

        fig = px.box(self.df, **params)
        return fig


@dataclass
class ViolinPlot(BaseChart):
    """Скрипичная диаграмма"""
    chart_type: str = "violin"
    color: str = ""

    def plot(self) -> px.violin:
        # Строим словарь параметров только с непустыми значениями
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Добавляем цвет, только если он задан
        if self.color:
            params["color"] = self.color

        # Добавляем любые дополнительные параметры
        params.update(self.params)

        fig = px.violin(self.df, **params)
        return fig


@dataclass
class Heatmap(BaseChart):
    """Тепловая карта"""
    chart_type: str = "heatmap"
    z_column: str = ""

    def plot(self) -> px.density_heatmap:
        # Строим словарь параметров только с непустыми значениями
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Добавляем z, только если он задан
        if self.z_column:
            params["z"] = self.z_column

        # Добавляем любые дополнительные параметры
        params.update(self.params)

        fig = px.density_heatmap(self.df, **params)
        return fig


@dataclass
class AreaChart(BaseChart):
    """Диаграмма с областями"""
    chart_type: str = "area"
    color: str = ""

    def plot(self) -> px.area:
        # Строим словарь параметров только с непустыми значениями
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Добавляем цвет, только если он задан
        if self.color:
            params["color"] = self.color

        # Добавляем любые дополнительные параметры
        params.update(self.params)

        fig = px.area(self.df, **params)
        return fig


@dataclass
class SunburstChart(BaseChart):
    """Солнечная диаграмма"""
    chart_type: str = "sunburst"
    color: str = ""

    def plot(self) -> px.sunburst:
        # Строим словарь параметров только с непустыми значениями
        params = {
            "path": [self.x_column, self.y_column],
            "title": self.title if self.title else None,
        }

        # Добавляем цвет, только если он задан
        if self.color:
            params["color"] = self.color

        # Добавляем любые дополнительные параметры
        params.update(self.params)

        fig = px.sunburst(self.df, **params)
        return fig


@dataclass
class FunnelChart(BaseChart):
    """Воронкообразная диаграмма"""
    chart_type: str = "funnel"
    color: str = ""

    def plot(self) -> px.funnel:
        # Строим словарь параметров только с непустыми значениями
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Добавляем цвет, только если он задан
        if self.color:
            params["color"] = self.color

        # Добавляем любые дополнительные параметры
        params.update(self.params)

        fig = px.funnel(self.df, **params)
        return fig


# Карта типов графиков с русскими названиями
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