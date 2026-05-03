from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
import streamlit as st
import pandas as pd
import json

class GigaChatAssistant:
    """AI-ассистент на базе GigaChat для анализа данных"""
    
    def __init__(self, api_key=None):
        """
        Инициализация ассистента
        
        Args:
            api_key: Ключ API GigaChat. Если None, берется из secrets или переменных окружения
        """
        if api_key is None:
            # Пробуем получить ключ из разных источников
            api_key = self._get_api_key()
        
        self.api_key = api_key
        self.client = None
        self.initialize_client()
        
    def _get_api_key(self):
        """Получение API ключа из разных источников"""
        # Проверяем secrets.toml
        try:
            return st.secrets["gigachat"]["api_key"]
        except:
            pass
        
        # Проверяем переменные окружения
        import os
        api_key = os.environ.get("GIGACHAT_API_KEY")
        if api_key:
            return api_key
            
        return None
    
    def initialize_client(self):
        """Инициализация клиента GigaChat"""
        if self.api_key:
            try:
                self.client = GigaChat(credentials=self.api_key, verify_ssl_certs=False)
                return True
            except Exception as e:
                st.error(f"Ошибка инициализации GigaChat: {e}")
                return False
        return False
    
    def generate_data_analysis(self, df_info, user_query):
        """
        Генерация анализа данных на основе информации о датафрейме
        
        Args:
            df_info: Словарь с информацией о датафрейме
            user_query: Запрос пользователя
            
        Returns:
            str: Ответ ассистента
        """
        if not self.client:
            return "❌ GigaChat не инициализирован. Проверьте API ключ."
        
        # Формируем контекст из информации о датафрейме
        context = self._create_context(df_info)
        
        # Создаем системный промпт
        system_prompt = """Ты - AI-ассистент для анализа данных. Твоя задача - помогать пользователю анализировать данные, 
        предлагать визуализации, находить закономерности и давать рекомендации по анализу данных.
        
        Отвечай на русском языке. Будь полезным, точным и конкретным.
        
        При анализе данных учитывай:
        - Типы данных в колонках
        - Статистические характеристики
        - Возможные взаимосвязи между переменными
        - Рекомендации по визуализации
        - Предложения по предобработке данных
        
        Если пользователь спрашивает о визуализации, предложи конкретные типы графиков 
        с указанием какие колонки использовать."""
        
        # Формируем сообщения
        payload = {
            "model": "GigaChat",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Контекст данных:\n{context}\n\nВопрос пользователя: {user_query}"
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        try:
            response = self.client.chat(payload)
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Ошибка при запросе к GigaChat: {str(e)}"
    
    def _create_context(self, df_info):
        """Создание контекста из информации о датафрейме"""
        context = []
        
        if 'shape' in df_info:
            context.append(f"Размер данных: {df_info['shape']['rows']} строк, {df_info['shape']['columns']} колонок")
        
        if 'columns' in df_info:
            context.append("\nКолонки и их типы:")
            for col in df_info['columns']:
                dtype = df_info.get('dtypes', {}).get(col, 'unknown')
                missing = df_info.get('missing_values', {}).get(col, {})
                missing_str = ""
                if missing:
                    missing_str = f" (пропущено: {missing.get('count', 0)}, {missing.get('percentage', 0):.1f}%)"
                context.append(f"  - {col}: {dtype}{missing_str}")
        
        if 'numeric_stats' in df_info and df_info['numeric_stats']:
            context.append("\nСтатистика числовых колонок:")
            for col, stats in df_info['numeric_stats'].items():
                stats_str = ", ".join([f"{k}: {v}" for k, v in stats.items()])
                context.append(f"  - {col}: {stats_str}")
        
        return "\n".join(context)
    
    def analyze_dataframe(self, df):
        """
        Анализ датафрейма и генерация рекомендаций
        
        Args:
            df: pandas DataFrame
            
        Returns:
            str: Рекомендации по анализу
        """
        if not self.client:
            return None
        
        # Собираем информацию о датафрейме
        df_info = {
            'shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'columns': list(df.columns),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'missing_values': {
                col: {
                    'count': int(df[col].isna().sum()),
                    'percentage': (df[col].isna().sum() / len(df) * 100)
                } for col in df.columns
            },
            'numeric_stats': {}
        }
        
        # Добавляем статистику для числовых колонок
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe()
            for col in numeric_cols:
                df_info['numeric_stats'][col] = {
                    'mean': round(stats[col]['mean'], 2),
                    'std': round(stats[col]['std'], 2),
                    'min': round(stats[col]['min'], 2),
                    'max': round(stats[col]['max'], 2)
                }
        
        # Формируем запрос на анализ
        query = """Проанализируй эти данные и предложи:
        1. Какие визуализации стоит построить в первую очередь
        2. Какие статистические тесты применить
        3. Есть ли проблемы с данными (пропуски, выбросы)
        4. Какие взаимосвязи между переменными стоит исследовать
        5. Конкретные рекомендации по дальнейшему анализу
        
        Отвечай структурированно, но без излишней детализации."""
        
        return self.generate_data_analysis(df_info, query)
    
    def suggest_charts(self, df, columns=None):
        """
        Предложение типов графиков для визуализации
        
        Args:
            df: pandas DataFrame
            columns: список колонок для визуализации
            
        Returns:
            str: Предложения по визуализации
        """
        if not self.client:
            return None
        
        df_info = {
            'shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'columns': list(df.columns),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
        }
        
        if columns:
            query = f"Предложи подходящие типы графиков для визуализации следующих колонок: {', '.join(columns)}. "
        else:
            query = "Предложи 5-7 наиболее информативных типов графиков для этих данных."
        
        query += """Для каждого графика укажи:
        - Тип графика
        - Какие колонки использовать
        - Почему этот график полезен"""
        
        return self.generate_data_analysis(df_info, query)
    
    def explain_operation(self, df_info, operation_type, params):
        """
        Объяснение операции с данными
        
        Args:
            df_info: информация о датафрейме
            operation_type: тип операции
            params: параметры операции
            
        Returns:
            str: Объяснение операции
        """
        if not self.client:
            return None
        
        query = f"""Объясни простыми словами, что делает операция '{operation_type}' 
        с параметрами {params} и как она повлияет на данные.
        Контекст данных: {json.dumps(df_info, ensure_ascii=False)}
        Отвечай кратко, 2-3 предложения."""
        
        return self.generate_data_analysis(df_info, query)


def get_dataframe_info(df):
    """Получение базовой информации о датафрейме для контекста"""
    info = {
        'shape': {
            'rows': len(df),
            'columns': len(df.columns)
        },
        'columns': list(df.columns),
        'dtypes': {col: str(df[col].dtype) for col in df.columns},
        'missing_values': {
            col: {
                'count': int(df[col].isna().sum()),
                'percentage': round(df[col].isna().sum() / len(df) * 100, 2)
            } for col in df.columns
        }
    }
    
    # Добавляем базовую статистику для числовых колонок
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        info['numeric_stats'] = {}
        stats = df[numeric_cols].describe()
        for col in numeric_cols:
            info['numeric_stats'][col] = {
                'mean': round(stats[col]['mean'], 2) if 'mean' in stats[col] else None,
                'std': round(stats[col]['std'], 2) if 'std' in stats[col] else None,
                'min': round(stats[col]['min'], 2) if 'min' in stats[col] else None,
                'max': round(stats[col]['max'], 2) if 'max' in stats[col] else None
            }
    
    return info