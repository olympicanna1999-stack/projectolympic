import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from io import BytesIO
import base64

# ---------------------------
# In-memory temporary DB stub
# ---------------------------
import re
from contextlib import contextmanager

# initial empty tables
_inmem_athletes = pd.DataFrame(columns=['id', 'Имя', 'Возраст', 'Вид спорта'])
_inmem_measurements = pd.DataFrame(columns=['id', 'athlete_id', 'date', 'metric', 'value'])

# simple id counters
_next_athlete_id = 1
_next_measurement_id = 1

class InMemorySession:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def execute(self, sql, params=None):
        # Normalize
        s = (sql or "").strip().lower()
        p = params or {}

        global _inmem_athletes, _inmem_measurements, _next_athlete_id, _next_measurement_id

        # INSERT INTO athletes ("Имя", "Возраст", "Вид спорта") VALUES (:Имя, :Возраст, :Вид_спорта)
        if 'insert into athletes' in s:
            # Try to get fields from params
            # params keys may be in different names depending on caller; handle common ones
            name = p.get('Имя') or p.get('name') or p.get('Имя'.lower())
            age = p.get('Возраст') or p.get('age') or p.get('Возраст'.lower())
            sport = p.get('Вид_спорта') or p.get('Вид спорта') or p.get('sport') or p.get('Вид спорта'.lower())
            if name is None and isinstance(params, (list, tuple)):
                # support executemany with positional tuple (unlikely here)
                pass
            row = {'id': _next_athlete_id, 'Имя': name, 'Возраст': age, 'Вид спорта': sport}
            _inmem_athletes = pd.concat([_inmem_athletes, pd.DataFrame([row])], ignore_index=True)
            _next_athlete_id += 1
            return

        # INSERT INTO measurements (athlete_id, "Date", metric, value) VALUES (...)
        if 'insert into measurements' in s:
            aid = p.get('athlete_id') or p.get('Athlete_ID') or p.get('Athlete_Id') or p.get('Athlete_ID'.lower()) or p.get('Athlete_ID'.replace('_',''))
            date = p.get('Date') or p.get('date')
            metric = p.get('metric') or p.get('Metric')
            value = p.get('value') or p.get('Value')
            row = {'id': _next_measurement_id, 'athlete_id': aid, 'date': date, 'metric': metric, 'value': value}
            _inmem_measurements = pd.concat([_inmem_measurements, pd.DataFrame([row])], ignore_index=True)
            _next_measurement_id += 1
            return

        # executemany style: sometimes code uses session.execute with many rows via loop — handled above per call
        # UPDATE / CREATE TABLE / other statements: ignore (no-op)
        return

    def executemany(self, sql, param_list):
        for p in param_list:
            self.execute(sql, params=p)

    def commit(self):
        return

class InMemoryConnWrapper:
    @property
    def session(self):
        return InMemorySession()

    def query(self, sql, params=None):
        s = (sql or "").strip().lower()
        # SELECT * FROM athletes
        if 'from athletes' in s:
            return _inmem_athletes.copy()
        if 'from measurements' in s:
            return _inmem_measurements.copy()
        # Fallback: try to detect simple where athlete_id IN (...) by numeric tuple in SQL
        # Very simple parser for queries like "select * from measurements where athlete_id in (1,2,3)"
        if 'from measurements' in s and 'where' in s:
            # attempt to extract athlete_id values
            m = re.search(r'athlete_id\s+in\s*\(([^)]+)\)', s)
            if m:
                vals = [int(x.strip()) for x in m.group(1).split(',') if x.strip().isdigit()]
                return _inmem_measurements[_inmem_measurements['athlete_id'].isin(vals)].copy()
        # default empty DF
        return pd.DataFrame()

# Instantiate global conn used by app
conn = InMemoryConnWrapper()
# Provide a simple 'text' passthrough to satisfy code using text(...)
def text(s):
    return s
# ---------------------------
# End of in-memory stub
# ---------------------------

class InMemoryConn:
    def __init__(self, athletes_df, measurements_df):
        self._athletes = athletes_df.copy()
        self._measurements = measurements_df.copy()

    @property
    def session(self):
        # заглушка: предоставляет контекст-менеджер с пустыми commit/execute
        class DummySession:
            def __enter__(self_inner): return self_inner
            def __exit__(self_inner, exc_type, exc, tb): return False
            def execute(self_inner, *args, **kwargs): pass
            def executemany(self_inner, *args, **kwargs): pass
            def commit(self_inner): pass
        return DummySession()

    def query(self, sql, params=None):
        # минимальная поддержка SELECT * FROM athletes / measurements
        sql_low = sql.strip().lower()
        if 'from athletes' in sql_low:
            return self._athletes.copy()
        if 'from measurements' in sql_low:
            return self._measurements.copy()
        return pd.DataFrame()

# Использование:
# conn = InMemoryConn(athletes_df, measurements_df)

# Create tables if not exist
with conn.session as session:
    session.execute(text('''
    CREATE TABLE IF NOT EXISTS athletes (
        ID SERIAL PRIMARY KEY,
        "Имя" TEXT,
        "Возраст" INTEGER,
        "Вид спорта" TEXT
    )
    '''))
    session.execute(text('''
    CREATE TABLE IF NOT EXISTS measurements (
        ID SERIAL PRIMARY KEY,
        Athlete_ID INTEGER,
        "Date" DATE,
        Metric TEXT,
        Value REAL,
        FOREIGN KEY (Athlete_ID) REFERENCES athletes(ID)
    )
    '''))
    session.commit()

# Function to load data from DB
def load_athletes():
    df = conn.query('SELECT * FROM athletes')
    # Normalize column names to lowercase for consistent usage
    df.columns = [c.lower() for c in df.columns]
    return df

def load_measurements():
    df = conn.query('SELECT * FROM measurements')
    df.columns = [c.lower() for c in df.columns]
    return df

# Mock data generation (only if DB is empty)
athletes_df = load_athletes()
if athletes_df.empty:
    sports = ['Лыжные гонки', 'Биатлон', 'Академическая гребля']
    first_names_male = ['Александр', 'Дмитрий', 'Иван', 'Максим', 'Никита', 'Сергей', 'Артем', 'Егор', 'Кирилл', 'Михаил']
    last_names_male = ['Иванов', 'Петров', 'Сидоров', 'Кузнецов', 'Смирнов', 'Попов', 'Васильев', 'Михайлов', 'Новиков', 'Федоров']
    first_names_female = ['Анастасия', 'Мария', 'София', 'Анна', 'Дарья', 'Виктория', 'Елизавета', 'Полина', 'Ксения', 'Екатерина']
    last_names_female = ['Иванова', 'Петрова', 'Сидорова', 'Кузнецова', 'Смирнова', 'Попова', 'Васильева', 'Михайлова', 'Новикова', 'Федорова']

    def generate_russian_name(gender='male'):
        if gender == 'male':
            return random.choice(first_names_male) + ' ' + random.choice(last_names_male)
        else:
            return random.choice(first_names_female) + ' ' + random.choice(last_names_female)

    athletes = []
    for sport in sports:
        for _ in range(15):
            gender = random.choice(['male', 'female'])
            name = generate_russian_name(gender)
            age = random.randint(14, 18)
            athletes.append({'Имя': name, 'Возраст': age, 'Вид спорта': sport})

    # Insert into DB
    with conn.session as session:
        for a in athletes:
            session.execute(text('INSERT INTO athletes ("Имя", "Возраст", "Вид спорта") VALUES (:Имя, :Возраст, :Вид_спорта)'),
                            {'Имя': a['Имя'], 'Возраст': a['Возраст'], 'Вид_спорта': a['Вид спорта']})
        session.commit()

    athletes_df = load_athletes()

# Generate mock measurements if none exist
measurements_df = load_measurements()
if measurements_df.empty:
    metrics_list = ['МПК', 'Максимальная сила', 'Безжировая масса тела', 'ПАНО', 'ЧСС в покое', 'Максимальная ЧСС', 'Ударный объем сердца']
    ranges = {
        'МПК': (50, 80),
        'Максимальная сила': (100, 300),
        'Безжировая масса тела': (40, 80),
        'ПАНО': (3.5, 5.5),
        'ЧСС в покое': (50, 70),
        'Максимальная ЧСС': (180, 220),
        'Ударный объем сердца': (100, 200)
    }
    today = datetime.now()
    measurements = []

    # Expect athletes_df has 'id' column after normalization
    id_col = 'id' if 'id' in athletes_df.columns else 'id'  # keep for clarity
    for aid in athletes_df[id_col]:
        for i in range(4):
            date = (today - timedelta(days=random.randint(30 * i, 30 * (i + 1) + 30))).date()
            for metric in metrics_list:
                base_value = random.uniform(ranges[metric][0], ranges[metric][1])
                progress_factor = 1 + (0.05 * (3 - i))
                if metric in ['ЧСС в покое', 'Максимальная ЧСС']:
                    value = int(base_value * progress_factor)
                else:
                    value = round(base_value * progress_factor, 1)
                if metric == 'ЧСС в покое':
                    value = int(base_value * (1 - 0.05 * (3 - i)))
                measurements.append({'athlete_id': aid, 'date': date, 'metric': metric, 'value': value})

    # Bulk insert
    with conn.session as session:
        for m in measurements:
            session.execute(text('INSERT INTO measurements (athlete_id, "Date", metric, value) VALUES (:athlete_id, :Date, :metric, :value)'),
                            {'athlete_id': m['athlete_id'], 'Date': m['date'], 'metric': m['metric'], 'value': m['value']})
        session.commit()

    measurements_df = load_measurements()

# Ensure metrics_list and ranges present if not defined above
if 'metrics_list' not in globals():
    metrics_list = ['МПК', 'Максимальная сила', 'Безжировая масса тела', 'ПАНО', 'ЧСС в покое', 'Максимальная ЧСС', 'Ударный объем сердца']
if 'ranges' not in globals():
    ranges = {
        'МПК': (50, 80),
        'Максимальная сила': (100, 300),
        'Безжировая масса тела': (40, 80),
        'ПАНО': (3.5, 5.5),
        'ЧСС в покое': (50, 70),
        'Максимальная ЧСС': (180, 220),
        'Ударный объем сердца': (100, 200)
    }

# User roles and access
users = {
    'project_leader': {'password': 'admin_pass', 'role': 'leader'},  # Full access
    'curator_ski_racing': {'password': 'ski_pass', 'role': 'curator', 'sport': 'Лыжные гонки'},
    'curator_biathlon': {'password': 'biathlon_pass', 'role': 'curator', 'sport': 'Биатлон'},
    'curator_rowing': {'password': 'rowing_pass', 'role': 'curator', 'sport': 'Академическая гребля'}
}

# Helper: get athlete data with filters (SQL-level)
def get_athlete_data(user_info, athlete_id=None, search_name=None, min_age=None, max_age=None, sport_filter=None):
    query = 'SELECT * FROM athletes'
    params = {}
    conditions = []
    if user_info['role'] == 'curator':
        conditions.append('"Вид спорта" = :sport')
        params['sport'] = user_info['sport']
    if athlete_id:
        conditions.append('id = :id')
        params['id'] = athlete_id
    if search_name:
        conditions.append('"Имя" ILIKE :name')
        params['name'] = f'%{search_name}%'
    if min_age is not None:
        conditions.append('"Возраст" >= :min_age')
        params['min_age'] = min_age
    if max_age is not None:
        conditions.append('"Возраст" <= :max_age')
        params['max_age'] = max_age
    if sport_filter and user_info['role'] == 'leader':
        conditions.append('"Вид спорта" = :sport_filter')
        params['sport_filter'] = sport_filter
    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)
    df = conn.query(query, params=params)
    df.columns = [c.lower() for c in df.columns]
    return df

# Measurements loader with role filtering (SQL-level)
def get_measurements(user_info, athlete_id=None):
    query = 'SELECT * FROM measurements'
    params = {}
    conditions = []
    if user_info['role'] == 'curator':
        athlete_ids_df = get_athlete_data(user_info)
        if not athlete_ids_df.empty:
            athlete_ids = tuple(athlete_ids_df['id'].tolist())
            if len(athlete_ids) == 1:
                conditions.append('athlete_id = :single_id')
                params['single_id'] = athlete_ids[0]
            else:
                conditions.append(f'athlete_id IN {athlete_ids}')
    if athlete_id:
        conditions.append('athlete_id = :id')
        params['id'] = athlete_id
    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)
    df = conn.query(query, params=params)
    df.columns = [c.lower() for c in df.columns]
    return df

# Add measurement into DB
def add_measurement(user_info, athlete_id, date, updates):
    # check rights: leader can always, curator only for their sport
    if user_info['role'] == 'leader' or (user_info['role'] == 'curator' and get_athlete_data(user_info, athlete_id)['в��д спорта'].values[0] == user_info['sport']):
        with conn.session as session:
            for metric, value in updates.items():
                if metric in metrics_list:
                    session.execute(text('INSERT INTO measurements (athlete_id, "Date", metric, value) VALUES (:Athlete_ID, :Date, :Metric, :Value)'),
                                    {'Athlete_ID': athlete_id, 'Date': date, 'Metric': metric, 'Value': value})
            session.commit()
        return True
    return False

# Update athlete
def update_athlete(user_info, athlete_id, updates):
    # rights check similar to add_measurement
    if user_info['role'] == 'leader' or (user_info['role'] == 'curator' and get_athlete_data(user_info, athlete_id)['в��д спорта'].values[0] == user_info['sport']):
        set_clause = []
        params = {'id': athlete_id}
        for key, value in updates.items():
            set_clause.append(f'"{key}" = :{key}')
            params[key] = value
        query = text(f'UPDATE athletes SET {", ".join(set_clause)} WHERE id = :id')
        with conn.session as session:
            session.execute(query, params)
            session.commit()
        return True
    return False

# Visualizations
@st.cache_data
def visualize_average_metrics(user_info):
    data = get_measurements(user_info)
    if data.empty:
        return None

    # Ensure date column is datetime
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    else:
        return None

    latest = data.loc[data.groupby(['athlete_id', 'metric'])['date'].idxmax()]
    pivot = latest.pivot(index='athlete_id', columns='metric', values='value')
    athletes_with_sport = get_athlete_data(user_info)[['id', 'Вид спорта'.lower() if 'в��д спорта' in get_athlete_data(user_info).columns else 'в��д спорта']].set_index('id')
    # safer join: if pivot empty, return None
    if pivot.empty:
        return None
    combined = pivot.join(athletes_with_sport, how='left')
    if 'Вид спорта' in combined.columns or 'в��д спорта' in combined.columns:
        sport_col = 'Вид спорта' if 'Вид спорта' in combined.columns else 'в��д спорта'
    else:
        # fallback: if column names differ, try 'вид спорта'
        sport_col = [c for c in combined.columns if 'вид' in c.lower()]
        sport_col = sport_col[0] if sport_col else None
    if sport_col is None:
        return None

    averages = combined.groupby(sport_col).mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    averages.plot(kind='bar', ax=ax)
    ax.set_title('Средние показатели по видам спорта (последние измерения)')
    ax.set_ylabel('Значения')
    ax.set_xlabel('Вид спорта')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

@st.cache_data
def visualize_athlete_metrics(user_info, athlete_id):
    data = get_measurements(user_info, athlete_id)
    if data.empty:
        return None

    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    else:
        return None

    # safe groupby idxmax (dropna)
    idx = data.groupby('metric')['date'].idxmax().dropna().astype(int)
    latest = data.loc[idx]
    values = latest.set_index('metric')['value'].to_dict()

    metrics = metrics_list
    vals = [values.get(m, 0) for m in metrics]

    normalized = []
    for m, v in zip(metrics, vals):
        mn, mx = ranges.get(m, (0, 1))
        denom = mx - mn if mx - mn != 0 else 1
        normalized.append((v - mn) / denom)

    angles = [n / float(len(metrics)) * 2 * 3.14159 for n in range(len(metrics))]
    angles += angles[:1]
    normalized += normalized[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, normalized, color='blue', alpha=0.25)
    ax.plot(angles, normalized, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)

    # safe athlete name lookup
    a = get_athlete_data(user_info, athlete_id)
    athlete_name = a['имя'].values[0] if not a.empty and 'имя' in a.columns else f'ID {athlete_id}'
    ax.set_title(f"Показатели спортсмена {athlete_name} (последние)")
    plt.tight_layout()
    return fig

@st.cache_data
def visualize_progress(user_info, athlete_id):
    data = get_measurements(user_info, athlete_id)
    if data.empty:
        return None

    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    else:
        return None

    fig, axs = plt.subplots(len(metrics_list), 1, figsize=(10, 15), sharex=True)
    fig.suptitle(f'Прогресс спортсмена ID {athlete_id} со временем')

    for i, metric in enumerate(metrics_list):
        metric_data = data[data['metric'] == metric].sort_values('date')
        if not metric_data.empty:
            axs[i].plot(metric_data['date'], metric_data['value'], marker='o')
            axs[i].set_ylabel(metric)
            axs[i].grid(True)
        else:
            axs[i].set_ylabel(metric)
            axs[i].text(0.5, 0.5, "Нет данных", ha='center', va='center', transform=axs[i].transAxes)

    axs[-1].set_xlabel('Дата')
    plt.setp(axs[-1].xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    return fig

def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

def generate_pdf_report(athlete_id):
    athlete = get_athlete_data({'role': 'leader'}, athlete_id)
    if athlete.empty:
        return None

    athlete = athlete.iloc[0]
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, height - 1 * inch, "Паспорт спортсмена")

    # Athlete info
    c.setFont("Helvetica", 12)
    y = height - 1.5 * inch
    # safe keys
    aid = athlete.get('id') if 'id' in athlete else athlete.get('ID', '')
    name = athlete.get('имя') if 'имя' in athlete else athlete.get('Имя', '')
    age = athlete.get('возраст') if 'возраст' in athlete else athlete.get('Возраст', '')
    sport = athlete.get('вид спорта') if 'вид спорта' in athlete else athlete.get('Вид спорта', '')

    c.drawString(1 * inch, y, f"ID: {aid}")
    y -= 0.4 * inch
    c.drawString(1 * inch, y, f"Имя: {name}")
    y -= 0.4 * inch
    c.drawString(1 * inch, y, f"Возраст: {age}")
    y -= 0.4 * inch
    c.drawString(1 * inch, y, f"Вид спорта: {sport}")
    y -= 0.5 * inch

    # Latest physical metrics
    latest_data = get_measurements({'role': 'leader'}, athlete_id)
    if not latest_data.empty and 'date' in latest_data.columns:
        latest_data['date'] = pd.to_datetime(latest_data['date'])
        idx = latest_data.groupby('metric')['date'].idxmax().dropna().astype(int)
        latest = latest_data.loc[idx]
        c.drawString(1 * inch, y, "Последние показатели физического тестирования:")
        y -= 0.4 * inch
        for _, row in latest.iterrows():
            row_date = row['date'].date() if hasattr(row['date'], 'date') else row['date']
            c.drawString(1.2 * inch, y, f"{row['metric']}: {row['value']} (Дата: {row_date})")
            y -= 0.3 * inch
            if y < 1 * inch:
                c.showPage()
                y = height - 1 * inch
    else:
        c.drawString(1 * inch, y, "Данных по измерениям не найдено.")
        y -= 0.3 * inch

    # Add radar chart
    radar_fig = visualize_athlete_metrics({'role': 'leader'}, athlete_id)
    if radar_fig:
        radar_img = fig_to_image(radar_fig)
        img_reader = ImageReader(radar_img)
        # ensure enough space
        c.drawImage(img_reader, 1 * inch, y - 4 * inch, width=4 * inch, height=4 * inch)
        y -= 4.5 * inch

    # Add progress chart on new page
    progress_fig = visualize_progress({'role': 'leader'}, athlete_id)
    if progress_fig:
        c.showPage()
        y = height - 1 * inch
        progress_img = fig_to_image(progress_fig)
        img_reader = ImageReader(progress_img)
        c.drawImage(img_reader, 1 * inch, y - 8 * inch, width=6 * inch, height=8 * inch)

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

# ------------------------------
# UI: themes and main app flow
# ------------------------------
light_theme = """
    <style>
        .stApp { background-color: #f8f9fa; color: black; }
        .stButton > button { background-color: #007bff; color: white; border-radius: 4px; }
        .stTextInput > div > div > input { border: 1px solid #ced4da; border-radius: 4px; }
        .dataframe th { background-color: #e9ecef; }
        .stDataFrame { border: 1px solid #dee2e6; border-radius: 4px; }
        .dataframe tr:hover { background-color: #f1f3f5; }
    </style>
"""

dark_theme = """
    <style>
        .stApp { background-color: #1e1e1e; color: white; }
        h1,h2,h3,h4,h5,h6 { color: white; }
        .stButton > button { background-color: #0d6efd; color: white; border-radius: 4px; }
        .stTextInput > div > div > input { background-color: #2c2c2c; color: white; border: 1px solid #495057; border-radius: 4px; }
        .dataframe { color: white; border: 1px solid #495057; border-radius: 4px; }
        .dataframe th { background-color: #343a40; }
        .dataframe tr:hover { background-color: #495057; }
    </style>
"""

st.title("Цифровой реестр и паспорт спортсмена")

if 'theme' not in st.session_state:
    st.session_state.theme = 'Светлая'

st.sidebar.title("Настройки")
theme_choice = st.sidebar.radio("Тема интерфейса", ['Светлая', 'Тёмная'])
if theme_choice != st.session_state.theme:
    st.session_state.theme = theme_choice
    # безопасно перезапустить рендеринг
    st.experimental_set_query_params(theme=st.session_state.theme)
    st.stop()

if st.session_state.theme == 'Тёмная':
    st.markdown(dark_theme, unsafe_allow_html=True)
else:
    st.markdown(light_theme, unsafe_allow_html=True)

# Authentication (пример простой авторизации на основе users)
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

if st.session_state.user_info is None:
    username = st.text_input("Логин")
    password = st.text_input("Пароль", type="password")
    if st.button("Войти"):
        user_info = None
        if username in users and users[username]['password'] == password:
            user_info = users[username]
        if user_info:
            st.session_state.user_info = user_info
            st.success("Успешный вход!")
            # безопасный ре-рендер
            st.experimental_set_query_params(logged_in='1')
            st.stop()
        else:
            st.error("Неверный логин или пароль")
else:
    user_info = st.session_state.user_info
    st.sidebar.markdown(f"**Пользователь:** {user_info['role']}")
    if st.sidebar.button("Выйти"):
        st.session_state.user_info = None
        st.experimental_set_query_params(logged_in='0')
        st.stop()

    # Sidebar filters and main UI
    st.sidebar.title("Фильтры")
    search_name = st.sidebar.text_input("Поиск по имени")
    min_age, max_age = st.sidebar.slider("Возраст", 14, 18, (14, 18))
    if user_info['role'] == 'leader':
        # get list of sports from DB
        sports_list = load_athletes()['вид спорта'].unique().tolist() if 'вид спорта' in load_athletes().columns else []
        sport_filter = st.sidebar.selectbox("Вид спорта", ['Все'] + sports_list)
        sport_filter = None if sport_filter == 'Все' else sport_filter
    else:
        sport_filter = None

    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("Список спортсменов")
        athletes_data = get_athlete_data(user_info, search_name=search_name, min_age=min_age, max_age=max_age, sport_filter=sport_filter)
        st.dataframe(athletes_data)

        # Select athlete for details
        if not athletes_data.empty and 'id' in athletes_data.columns:
            st.session_state.selected_athlete_id = st.selectbox("Выберите спортсмена по ID", athletes_data['id'].unique(), key="athlete_select")
        else:
            st.info("Нет данных спортсменов для выбора")
            st.session_state.selected_athlete_id = None

    if st.session_state.get('selected_athlete_id'):
        selected_athlete_id = st.session_state.selected_athlete_id

        with right_col:
            st.subheader("Детали спортсмена")
            athlete_details = get_athlete_data(user_info, selected_athlete_id)
            st.write(athlete_details)

            st.subheader("Измерения")
            meas_data = get_measurements(user_info, selected_athlete_id)
            st.dataframe(meas_data)

            with st.expander("Добавить новые измерения"):
                new_date = st.date_input("Дата")
                updates = {}
                cols = st.columns(2)
                for i, metric in enumerate(metrics_list):
                    with cols[i % 2]:
                        value = st.number_input(f"{metric}", value=0.0)
                        if value != 0.0:
                            updates[metric] = value
                if st.button("Добавить"):
                    if add_measurement(user_info, selected_athlete_id, new_date, updates):
                        st.success("Измерения добавлены!")
                        st.experimental_set_query_params(measure_added='1')
                        st.stop()
                    else:
                        st.error("Нет доступа для обновления")

        st.subheader("Визуализации")
     def visualize_average_metrics(user_info):
    data = get_measurements(user_info)
    if data.empty:
        return None

    # Удобный локатор имён колонок: ищет первый совпадающий вариант
    def find_col(df, *candidates):
        cols = [c.lower() for c in df.columns]
        for c in candidates:
            if c is None:
                continue
            if c.lower() in cols:
                # вернуть оригинальное имя из df.columns
                idx = cols.index(c.lower())
                return list(df.columns)[idx]
        return None

    date_col = find_col(data, 'date', 'Date', '"Date"')
    athlete_col = find_col(data, 'athlete_id', 'Athlete_ID', 'Athlete_Id', 'athlete')
    metric_col = find_col(data, 'metric', 'Metric')
    value_col = find_col(data, 'value', 'Value')

    if not all([date_col, athlete_col, metric_col, value_col]):
        # Не хватает необходимых столбцов — вернуть None, не падать
        return None

    # Приводим даты
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.dropna(subset=[date_col])
    if data.empty:
        return None

    # безопасная группировка: idxmax с dropna
    try:
        idx = data.groupby([athlete_col, metric_col])[date_col].idxmax().dropna().astype(int)
        latest = data.loc[idx]
    except Exception:
        # Если что-то пошло не так — аккуратно выйти
        return None

    # Пивот (если нет — вернём None)
    try:
        pivot = latest.pivot(index=athlete_col, columns=metric_col, values=value_col)
    except Exception:
        return None
    if pivot.empty:
        return None

    # Получаем колонку "вид спорта" из таблицы спортсменов (гибко по именам)
    athletes_df_local = get_athlete_data(user_info)
    sport_col = find_col(athletes_df_local, 'Вид спорта', 'вид спорта', 'sport', '"Вид спорта"')
    id_col = find_col(athletes_df_local, 'id', 'ID')

    if sport_col is None or id_col is None:
        return None

    # Подготовить таблицу со спортом по id
    athletes_with_sport = athletes_df_local[[id_col, sport_col]].set_index(id_col)

    # Подключаем (join) — быть готовым, если индексы не совпадают
    try:
        combined = pivot.join(athletes_with_sport, how='left')
    except Exception:
        return None

    # выбрать имя колонки спорта в combined (может быть разное имя)
    sport_col_in_combined = sport_col if sport_col in combined.columns else None
    if sport_col_in_combined is None:
        # попробуем найти столбец с "вид" в имени
        sport_col_candidates = [c for c in combined.columns if 'вид' in str(c).lower() or 'sport' in str(c).lower()]
        sport_col_in_combined = sport_col_candidates[0] if sport_col_candidates else None
    if sport_col_in_combined is None:
        return None

    # Рассчитать средние по виду спорта
    try:
        averages = combined.groupby(sport_col_in_combined).mean(numeric_only=True)
    except Exception:
        return None

    if averages.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    averages.plot(kind='bar', ax=ax)
    ax.set_title('Средние показатели по видам спорта (последние измерения)')
    ax.set_ylabel('Значения')
    ax.set_xlabel('Вид спорта')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

