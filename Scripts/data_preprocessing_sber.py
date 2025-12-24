import pandas as pd
import numpy as np
import copy
import os
import glob
from datetime import datetime
import scipy.stats as sts
import re

def preprocessing_data_from_sber(folder_path: str = 'Data') -> pd.DataFrame:
    folder_path = os.path.abspath(folder_path)
    print("Ищем CSV-файлы в:", folder_path)
    
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print("Найдено файлов:", len(csv_files))
    print("Файлы:", csv_files)
    
    if not csv_files:
        raise ValueError(f"Нет CSV-файлов в папке: {folder_path}")
    
    processed_dfs = []

    # Столбцы, которые должны читаться как строки
    string_columns = [
        'IS_OWN_TERMINAL',
        'MCC',
        'ACCOUNT_ID',
        'CARD',
        'CustomerKey',
        'ID',
        'TRAN_METHOD',
        'DETAILED_CARD_TYPE',
        "DEVICE_TYPE",
        'ADDRESS',
        'CurrencyName',
        'TERMINAL_CODE',
        'IS_VIRTUAL_TRANSACTION',
        'REGION',
        'CITY'
    ]

    for file in csv_files:
        print(f"Загрузка и обработка файла: {file}")
        
        # Получаем колонки файла
        cols_in_file = pd.read_csv(file, nrows=0).columns.tolist()
        dtype_dict = {col: str for col in string_columns if col in cols_in_file}
        # Загружаем с правильными типами
        data = pd.read_csv(file, dtype=dtype_dict, low_memory=False)
        data['IS_OWN_TERMINAL'] = data['IS_OWN_TERMINAL'].apply(categorate_is_own_terminal)

        data = device_type_fill(data)
        data = correct_address_handling_pipeline(data)
        validate_address_logic(data)
        # Удаление ненужных колонок
        cols_to_drop = ['CHANNEL', 'RETAILER', 'USED_PAY_SERVICE']
        data = data.drop(columns=[col for col in cols_to_drop if col in data.columns], errors='ignore')

        # Удаление строк с пропусками
        data = data.dropna().reset_index(drop=True)
        data['TERMINAL_CODE'] = data['TERMINAL_CODE'].replace('NONE', 'virtual')
        # Переименование колонок
        data = data.rename(columns={
            'IS_OWN_TERMINAL': 'ISOWNTERMINAL',
            'TRAN_METHOD': 'TRANMETHOD',
            'DETAILED_CARD_TYPE': 'DETAILEDCARDTYPE',
            'DEVICE_TYPE': 'DEVICETYPE',
            'IS_VIRTUAL_TRANSACTION': 'ISVIRTUALTRANSACTION'
        })

        # Приведение типов (теперь безопасно, так как всё — строки)
        type_mapping = {
            'ISOWNTERMINAL': lambda x: str(int(float(x))),
            'MCC': lambda x: str(int(float(x))),          # на случай, если float-строка
            'ACCOUNT_ID': lambda x: str(int(float(x))),
            'CARD': lambda x: str(int(float(x))),
            'CustomerKey': lambda x: str(int(float(x))),
            'ID': lambda x: str(int(float(x))),
            'TRANMETHOD': lambda x: str(int(float(x))),
        }

        for col, converter in type_mapping.items():
            if col in data.columns:
                data[col] = data[col].apply(converter)


        # Удаление выбросов по AMOUNT_EQ
        if 'AMOUNT_EQ' in data.columns:
            data = data.loc[remove_outliers(data[['AMOUNT_EQ']]).index]

        if 'PAY_AMT' in data.columns:
            data = data.loc[remove_outliers(data[['PAY_AMT']]).index]    

        data['EXCHANGE_RATA'] = data['PAY_AMT'] / data['AMOUNT_EQ']

        # data['DATE'] = pd.to_datetime(data['DATE'], format='%Y-%m-%d')
        # data['DATETIME']
        # data['DATETIME'] = pd.to_datetime(
        #     data['DATE'].dt.strftime('%Y-%m-%d') + ' ' + data['TRANS_TIME'],
        #     format='%Y-%m-%d %H:%M:%S'
        # )

        # Обработка даты и времени
        data['DATE'] = pd.to_datetime(data['DATE'], format='%Y-%m-%d')
        data['TRANS_TIME'] = pd.to_datetime(data['TRANS_TIME'], format='%H:%M:%S')

        processed_dfs.append(data)

    # Объединение
    combined_data = pd.concat(processed_dfs, ignore_index=True)

    # Фильтрация аккаунтов (>2 транзакций)
    account_counts = combined_data['ACCOUNT_ID'].value_counts()
    valid_accounts = account_counts[account_counts > 2].index
    combined_data = combined_data[combined_data['ACCOUNT_ID'].isin(valid_accounts)]

    # Сортировка (исправлено!)
    combined_data = combined_data.sort_values(['DATE' , 'TRANS_TIME'], ascending=[True, True])

    # Извлечение временных компонентов
    combined_data['HOUR'] = combined_data['TRANS_TIME'].dt.hour
    combined_data['MINUTE'] = combined_data['TRANS_TIME'].dt.minute
    combined_data['SECOND'] = combined_data['TRANS_TIME'].dt.second
    # combined_data = combined_data.drop(columns=['DATE', 'TRANS_TIME'], errors='ignore')
    print(f"Итоговый датафрейм содержит {combined_data.shape[0]} строк и {combined_data.shape[1]} колонок.")
    return combined_data


def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    data = copy.deepcopy(data)
    for col in data.columns:
        q25 = np.quantile(data[col], 0.25)
        q75 = np.quantile(data[col], 0.75)
        iqr_val = sts.iqr(data[col])
        lower_bound = q25 - 3 * iqr_val
        upper_bound = q75 + 3 * iqr_val
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data


def create_transaction_type_features(data):
    """
    Создание признаков, отражающих тип транзакции
    """
    data_processed = data.copy()
    
    # Определяем, является ли транзакция оффлайн
    def is_offline_transaction(row):
        # Критерии оффлайн-транзакции

        trans_detail = str(row['DEVICE_TYPE'])

        if trans_detail == "VIRTUAL":
            return False
        
        # По умолчанию считаем оффлайн (для физических терминалов)
        return True
    
    # Создаем индикатор
    data_processed['IS_OFFLINE'] = data_processed.apply(is_offline_transaction, axis=1).astype(int)
    data_processed['IS_ONLINE'] = (~data_processed['IS_OFFLINE'].astype(bool)).astype(int)
    
    return data_processed

def handle_address_with_transaction_type(data):
    """
    Корректная обработка ADDRESS с учетом типа транзакции
    """
    data_processed = data.copy()
    
    # Шаг 1: Создаем индикатор наличия адреса
    data_processed['HAS_ADDRESS'] = data_processed['ADDRESS'].notnull().astype(int)
    
    # Шаг 2: Для оффлайн-транзакций пробуем восстановить адрес
    offline_mask = data_processed['IS_OFFLINE'] == 1
    address_null_mask = data_processed['ADDRESS'].isnull()
    
    # Только для оффлайн-транзакций с пропущенным адресом
    offline_null_mask = offline_mask & address_null_mask
    
    if offline_null_mask.any():
        # Пытаемся восстановить адрес для оффлайн-транзакций
        terminal_address_map = data_processed[data_processed['ADDRESS'].notnull()]\
            .groupby('TERMINAL_CODE')['ADDRESS'].first()
        
        for idx in data_processed[offline_null_mask].index:
            terminal = data_processed.loc[idx, 'TERMINAL_CODE']
            if terminal in terminal_address_map:
                data_processed.loc[idx, 'ADDRESS'] = terminal_address_map[terminal]


    
    # Шаг 3: Для онлайн-транзакций заполняем осмысленными значениями
    online_mask = data_processed['IS_ONLINE'] == 1
    
    data_processed.loc[online_mask & address_null_mask, 'ADDRESS'] = 'VIRTUAL_TRANSACTION'
    data_processed.loc[online_mask & address_null_mask, 'ADDRESS_TYPE'] = 'VIRTUAL'
    
    # Шаг 4: Классификация адресов
    def classify_address(address):
        if pd.isna(address):
            return 'MISSING'
        
        address_str = str(address).lower()
        
        if 'виртуал' in address_str or 'virtual' in address_str:
            return 'VIRTUAL'
        elif 'онлайн' in address_str or 'online' in address_str:
            return 'ONLINE'
        elif 'неизвест' in address_str or 'unknown' in address_str:
            return 'UNKNOWN_OFFLINE'
        elif any(word in address_str for word in ['россия', 'ru', 'г ', 'ул ', 'дом']):
            return 'PHYSICAL'
        else:
            return 'OTHER'
    
    data_processed['ADDRESS_TYPE'] = data_processed['ADDRESS'].apply(classify_address)
    
    return data_processed

def create_smart_address_features(data):

    def has_no_digits(s):
        return not re.search(r'\d', s)
    """
    Создание признаков из адреса без искажения информации
    """
    data_features = data.copy()
    
    # 1. Географические признаки (только для физических адресов)
    def extract_geographic_features(address):
        if pd.isna(address):
            return {'CITY': 'NO_CITY', 'REGION': 'NO_REGION', 'IS_PHYSICAL': 0}
        
        address_str = str(address)
        
        # Если это виртуальная транзакция
        if 'VIRTUAL' in address_str or 'ONLINE' in address_str:
            return {'CITY': 'VIRTUAL', 'REGION': 'VIRTUAL', 'IS_PHYSICAL': 0}
        
        # Парсим физический адрес
        result = {'CITY': 'UNKNOWN', 'REGION': 'UNKNOWN', 'IS_PHYSICAL': 1}
        
        try:
            parts = address_str.split(',')
            for part in parts:
                part = part.strip()
                if 'г ' in part and (not ('г -' in part)):
                    city = part.split('г ', 1)[1]
                    if has_no_digits(city):
                        result['CITY'] = city
                    if 'санкт' in part.lower():
                        result['REGION'] = 'Санкт-Петербург'
                elif 'п ' in part :
                    city = part.split('п ', 1)[1]  
                    if has_no_digits(city):
                        result['CITY'] = city
                elif 'обл ' in part:
                    result['REGION'] = part
                elif 'санкт' in part.lower():
                    result['REGION'] = 'Санкт-Петербург'
                elif 'ленин' in part.lower():
                    result['REGION'] = 'обл Ленинградская'
                elif 'мурино' in part.lower():
                    result['CITY'] = 'Мурино'
                elif 'д ' in part.lower() and (not 'д (' in part.lower()):
                    city = part.split('д ', 1)[1]  
                    if has_no_digits(city):
                        result['CITY'] =city                        
            if result['CITY'] == 'Санкт-Петербург' and result['REGION'] == 'UNKNOWN':
                result['REGION'] = 'Санкт-Петербург'
            if result['REGION'] == 'Санкт-Петербург' and result['CITY'] == 'UNKNOWN':
                result['CITY'] = 'Санкт-Петербург'    
            
        except:
            pass
        
        return result
    
    # Применяем и создаем отдельные колонки
    geo_features = data_features['ADDRESS'].apply(extract_geographic_features)
    data_features['CITY'] = geo_features.apply(lambda x: x['CITY'])
    data_features['REGION'] = geo_features.apply(lambda x: x['REGION'])
    data_features['IS_PHYSICAL_LOCATION'] = geo_features.apply(lambda x: x['IS_PHYSICAL'])
    
    # 2. Признаки уровня детализации адреса
    def address_detail_level(address):
        if pd.isna(address):
            return 0
        
        address_str = str(address)
        
        if 'VIRTUAL' in address_str:
            return 1  # Виртуальный
        elif 'UNKNOWN' in address_str:
            return 2  # Неизвестный оффлайн
        
        # Считаем количество элементов в адресе
        parts = [p.strip() for p in address_str.split(',') if p.strip()]
        return len(parts)
    
    data_features['ADDRESS_DETAIL_LEVEL'] = data_features['ADDRESS'].apply(address_detail_level)
    
    # 3. Бинарные индикаторы
    data_features['HAS_PHYSICAL_ADDRESS'] = (data_features['IS_PHYSICAL_LOCATION'] == 1).astype(int)
    data_features['IS_VIRTUAL_TRANSACTION'] = (data_features['CITY'] == 'VIRTUAL').astype(int)
    
    return data_features



def correct_address_handling_pipeline(data):
    """
    Корректный пайплайн обработки адресов
    """
    print("Корректная обработка ADDRESS...")
    
    # Шаг 1: Определяем тип транзакции
    data = create_transaction_type_features(data)
    
    # Шаг 2: Обрабатываем адреса в зависимости от типа транзакции
    data = handle_address_with_transaction_type(data)
    
    # Шаг 3: Создаем информативные признаки
    data = create_smart_address_features(data)
    
    # Шаг 4: Создаем интерактивные признаки
    data['OFFLINE_WITH_ADDRESS'] = data['IS_OFFLINE'] * data['HAS_PHYSICAL_ADDRESS']
    data['ONLINE_VIRTUAL'] = data['IS_ONLINE'] * data['IS_VIRTUAL_TRANSACTION']
    
    print(f"Статистика:")
    print(f"  Оффлайн транзакций: {data['IS_OFFLINE'].sum()} ({data['IS_OFFLINE'].mean()*100:.1f}%)")
    print(f"  Онлайн транзакций: {data['IS_ONLINE'].sum()} ({data['IS_ONLINE'].mean()*100:.1f}%)")
    print(f"  Физических адресов: {data['HAS_PHYSICAL_ADDRESS'].sum()} ({data['HAS_PHYSICAL_ADDRESS'].mean()*100:.1f}%)")
    print(f"  Виртуальных транзакций: {data['IS_VIRTUAL_TRANSACTION'].sum()} ({data['IS_VIRTUAL_TRANSACTION'].mean()*100:.1f}%)")
    
    return data


def validate_address_logic(data):
    """
    Проверка корректности логики обработки адресов
    """
    print("Валидация логики обработки ADDRESS:")
    print("-" * 50)
    
    # Проверка 1: Все онлайн-транзакции должны быть виртуальными
    online_virtual = data[(data['IS_ONLINE'] == 1)].shape[0]
    online_total = data[data['IS_ONLINE'] == 1].shape[0]
    
    print(f"Онлайн транзакций: {online_total}")
    print(f"Онлайн транзакций, помеченных как виртуальные: {online_virtual}")
    print(f"Совпадение: {online_virtual/online_total*100:.1f}%")
    
    # Проверка 2: Оффлайн с физическими адресами
    offline_physical = data[(data['IS_OFFLINE'] == 1) & (data['HAS_PHYSICAL_ADDRESS'] == 1)].shape[0]
    offline_total = data[data['IS_OFFLINE'] == 1].shape[0]
    
    print(f"\nОффлайн транзакций: {offline_total}")
    print(f"Оффлайн с физическими адресами: {offline_physical}")
    print(f"Покрытие: {offline_physical/offline_total*100:.1f}%")
    
    # Проверка 3: Распределение типов адресов
    print(f"\nРаспределение ADDRESS_TYPE:")
    if 'ADDRESS_TYPE' in data.columns:
        print(data['ADDRESS_TYPE'].value_counts())

def device_type_fill(data, field = "DEVICE_TYPE"):
    data_filled = data.copy()
    
    # Стратегия 1: Использовать TERMINAL_CODE для поиска адресов
    terminal_address_map = data_filled.dropna(subset=[field, 'TERMINAL_CODE'])\
                                     .groupby('TERMINAL_CODE')[field].first()
    
    # Заполняем пропуски по TERMINAL_CODE
    mask_null_address = data_filled[field].isnull()
    for idx in data_filled[mask_null_address].index:
        terminal = data_filled.loc[idx, 'TERMINAL_CODE']
        if terminal in terminal_address_map:
            data_filled.loc[idx, field] = terminal_address_map[terminal]

    data_filled.loc[(data_filled['IS_OWN_TERMINAL'] == 0.0) & (data_filled[field].isna()), field] = 'VIRTUAL'

    return data_filled        
    
 

def categorate_is_own_terminal(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return 1 if value else 0
    value_str = str(value).strip().lower()
    if value_str == 'true':
        return 1
    elif value_str == 'false':
        return 0
    else:
        return np.nan
    
def shop_ralative(data):
    print("\n1. Текущее распределение DEVICE_TYPE и IS_OWN_TERMINAL:")
    cross_tab = pd.crosstab(data['DEVICE_TYPE'], data['IS_OWN_TERMINAL_BINARY'], margins=True)
    print(cross_tab)

    print("\n2. Процентное соотношение:")
    cross_tab_percent = pd.crosstab(
        data['IS_OWN_TERMINAL_BINARY'], 
        data['DEVICE_TYPE'].isnull(), 
        normalize='index'
    ) * 100
    print(cross_tab_percent.round(1))



if __name__ == "__main__":
    # df = preprocessing_data_from_sber(folder_path=r'Data\Sber\ditry')
    path = "Data\Sber\ditry"
    name = "transact18_19K"
    data = pd.read_csv(f'{path}\{name}.csv')
    print(data.info())
    # print(data["RETAILER"].unique())
    # print("\nРаспределение CHANNEL до обработки:")
    # print(data.info())
    # channel_counts = data['RETAILER'].value_counts(dropna=False)
    # print(channel_counts)
    # for channel, count in channel_counts.items():
    #     percent = count / len(data) * 100
    #     print(f"  '{str(channel)}': {count:6} ({percent:.1f}%)")
    # print(data['IS_OWN_TERMINAL'].value_counts(dropna=False))
    data['IS_OWN_TERMINAL_BINARY'] = data['IS_OWN_TERMINAL'].apply(categorate_is_own_terminal)
    # print(data['IS_OWN_TERMINAL_BINARY'].value_counts(dropna=False))
    # data = data.dropna(subset="IS_OWN_TERMINAL_BINARY")
    # print(data['IS_OWN_TERMINAL_BINARY'].value_counts(dropna=False))
    # shop_ralative(data)
    data = device_type_fill(data)
    # print(data.head())

    # shop_ralative(data)
    # cur_data = data[(data['IS_OWN_TERMINAL_BINARY'] == 0.0) & (data['ADDRESS'].notna())]
    # print(cur_data[["TRANS_DETAIL", "ADDRESS"]])
    # shop_ralative(data)
    # df_1 =pd.read_csv("Data\Sber\ditry\transact1_2K.csv")
    # validate_address_logic
    final_data = correct_address_handling_pipeline(data)
    validate_address_logic(final_data)
    # final_data = final_data[final_data["ADDRESS_TYPE"] == "MISSING" ]
    # final_data.to_csv(f'Data/Sber/Clear/{name}', 
    #       index=False,           # Не записывать индексы
    #       sep=',',               # Разделитель
    #       encoding='utf-8',      # Кодировка
    #       header=True,           # Записывать заголовки
    #       na_rep='NULL')         # Замена NaN значений
    # print(final_data[final_data["ADDRESS_TYPE"] == "MISSING" ]["TRANS_DETAIL", ""])
    # print(str(bool("false")))
    # Теперь df — ваш датафрейм, а pd — по-прежнему модуль pandas