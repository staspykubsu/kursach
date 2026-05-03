from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class BaseDfOperation:
    """Базовый класс для операций с датафреймом"""
    name: str
    id: str
    operation_type: str
    source_df_id: str = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование операции в словарь для сериализации"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseDfOperation":
        """Создание операции из словаря"""
        return cls(**data)


@dataclass
class FilterOperation(BaseDfOperation):
    """Операция фильтрации данных"""
    column: str = ""
    filter_type: str = ""  # "equals", "contains", "greater_than" и т.д.
    filter_value: Any = None
    operation_type: str = "filter"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применение фильтра к датафрейму"""
        if not self.column or not self.filter_type:
            return df

        # Определение функций фильтрации для каждого типа фильтра
        case_map = {
            "equals": lambda x: x == self.filter_value,
            "not_equals": lambda x: x != self.filter_value,
            "greater_than": lambda x: x > self.filter_value,
            "less_than": lambda x: x < self.filter_value,
            "between": lambda x: (x >= self.filter_value[0]) & (x <= self.filter_value[1]),
            "isin": lambda x: x in self.filter_value if isinstance(self.filter_value, (list, tuple)) else False,
        }

        # Добавление специфичных для строк типов фильтрации
        string_filters = {
            "contains": lambda x: str(self.filter_value) in str(x),
            "startswith": lambda x: str(x).startswith(str(self.filter_value)),
            "endswith": lambda x: str(x).endswith(str(self.filter_value)),
        }

        # Объединение всех типов фильтрации
        case_map.update(string_filters)

        if self.filter_type not in case_map:
            return df

        try:
            # Применение фильтра
            return df[df[self.column].apply(case_map[self.filter_type])]
        except Exception as e:
            # Резервный вариант при ошибках
            print(f"Ошибка при применении фильтра: {e}")
            return df


@dataclass
class AggregateOperation(BaseDfOperation):
    """Операция агрегации данных"""
    group_by: List[str] = field(default_factory=list)
    agg_func: Dict[str, str] = field(default_factory=dict)  # колонка: функция_агрегации
    operation_type: str = "aggregate"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применение агрегации к датафрейму"""
        if not self.group_by or not self.agg_func:
            return df

        return df.groupby(self.group_by).agg(self.agg_func).reset_index()


@dataclass
class DataCleanOperation(BaseDfOperation):
    """Операция очистки данных"""
    clean_type: str = ""  # "dropna", "fillna", "drop_duplicates", "replace", "rename"
    columns: List[str] = field(default_factory=list)
    fill_value: Any = None
    replace_values: Dict[Any, Any] = field(default_factory=dict)  # старое_значение: новое_значение
    new_column_names: Dict[str, str] = field(default_factory=dict)  # старое_имя: новое_имя
    operation_type: str = "data_clean"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применение очистки данных к датафрейму"""
        if not self.clean_type:
            return df

        result_df = df.copy()

        clean_methods = {
            "dropna": self._dropna,
            "fillna": self._fillna,
            "drop_duplicates": self._drop_duplicates,
            "replace": self._replace,
            "rename": self._rename,
        }

        if self.clean_type in clean_methods:
            return clean_methods[self.clean_type](result_df)

        return result_df

    def _dropna(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление строк с пропущенными значениями"""
        if self.columns:
            return df.dropna(subset=self.columns)
        else:
            return df.dropna()

    def _fillna(self, df: pd.DataFrame) -> pd.DataFrame:
        """Заполнение пропущенных значений"""
        result_df = df.copy()

        if self.columns:
            return self._fillna_columns(result_df, self.columns)
        else:
            return self._fillna_all(result_df)

    def _fillna_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Применение fillna к определенным колонкам"""
        result_df = df.copy()

        for col in columns:
            if col in result_df.columns:
                result_df[col] = self._fill_column_values(result_df, col)

        return result_df

    def _fillna_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применение fillna ко всем колонкам"""
        result_df = df.copy()

        if self.fill_value == "mean":
            for col in result_df.select_dtypes(include=["number"]).columns:
                result_df[col] = result_df[col].fillna(result_df[col].mean())

        elif self.fill_value == "median":
            for col in result_df.select_dtypes(include=["number"]).columns:
                result_df[col] = result_df[col].fillna(result_df[col].median())

        elif self.fill_value == "mode":
            for col in result_df.columns:
                mode_value = result_df[col].mode()
                if not mode_value.empty:
                    result_df[col] = result_df[col].fillna(mode_value[0])

        elif self.fill_value == "ffill":
            result_df = result_df.ffill()

        elif self.fill_value == "bfill":
            result_df = result_df.bfill()
        else:
            result_df = result_df.fillna(self.fill_value)

        return result_df

    def _fill_column_values(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Применение метода заполнения к определенной колонке"""
        if self.fill_value == "mean":
            if pd.api.types.is_numeric_dtype(df[column]):
                return df[column].fillna(df[column].mean())

        elif self.fill_value == "median":
            if pd.api.types.is_numeric_dtype(df[column]):
                return df[column].fillna(df[column].median())

        elif self.fill_value == "mode":
            mode_value = df[column].mode()
            if not mode_value.empty:
                return df[column].fillna(mode_value[0])

        elif self.fill_value == "ffill":
            return df[column].ffill()

        elif self.fill_value == "bfill":
            return df[column].bfill()

        else:
            return df[column].fillna(self.fill_value)

        return df[column]

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление дублирующихся строк на основе указанных колонок"""
        if self.columns:
            return df.drop_duplicates(subset=self.columns)
        else:
            return df.drop_duplicates()

    def _replace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Замена значений в указанных колонках"""
        if self.columns and self.replace_values:
            result_df = df.copy()
            for col in self.columns:
                if col in result_df.columns:
                    result_df[col] = result_df[col].replace(self.replace_values)
            return result_df
        elif self.replace_values:
            return df.replace(self.replace_values)
        return df

    def _rename(self, df: pd.DataFrame) -> pd.DataFrame:
        """Переименование колонок"""
        if self.new_column_names:
            return df.rename(columns=self.new_column_names)
        return df


@dataclass
class LoadCsvOperation(BaseDfOperation):
    """Операция загрузки CSV файла"""
    file_path: str = ""
    sep: str = ","
    operation_type: str = "load_csv"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Загрузка данных из CSV файла"""
        return pd.read_csv(self.file_path, sep=self.sep)


@dataclass
class SaveCsvOperation(BaseDfOperation):
    """Операция сохранения в CSV файл"""
    file_path: str = ""
    sep: str = ","
    operation_type: str = "save_csv"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Сохранение данных в CSV файл"""
        df.to_csv(self.file_path, sep=self.sep, index=False)
        return df