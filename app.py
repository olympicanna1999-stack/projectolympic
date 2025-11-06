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
from sqlalchemy import text

# Database setup using Streamlit's SQLConnection
# Assume secrets.toml has [connections.postgresql] with dbname, user, password, host, port
conn = st.connection('postgresql', type='sql')

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
    return conn.query('SELECT * FROM athletes')

def load_measurements():
    return conn.query('SELECT * FROM measurements')

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
            athletes.append((name, age, sport))

    with conn.session as session:
        session.executemany(text('INSERT INTO athletes ("Имя", "Возраст", "Вид спорта") VALUES (:Имя, :Возраст, :"Вид спорта")'), athletes)
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
    for aid in athletes_df['id']:
        for i in range(4):
            date = (today - timedelta(days=random.randint(30*i, 30*(i+1) + 30))).date()
            for metric in metrics_list:
                base_value = random.uniform(ranges[metric][0], ranges[metric][1])
                progress_factor = 1 + (0.05 * (3 - i))
                value = round(base_value * progress_factor, 1) if metric not in ['ЧСС в покое', 'Максимальная ЧСС'] else int(base_value * progress_factor)
                if metric == 'ЧСС в покое':
                    value = int(base_value * (1 - 0.05 * (3 - i)))
                measurements.append((aid, date, metric, value))

    with conn.session as session:
        session.executemany(text('INSERT INTO measurements (Athlete_ID, "Date", Metric, Value) VALUES (:Athlete_ID, :Date, :Metric, :Value)'), measurements)
        session.commit()

    measurements_df = load_measurements()

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

# User roles and access
users = {
    'project_leader': {'password': 'admin_pass', 'role': 'leader'},  # Full access
    'curator_ski_racing': {'password': 'ski_pass', 'role': 'curator', 'sport': 'Лыжные гонки'},
    'curator_biathlon': {'password': 'biathlon_pass', 'role': 'curator', 'sport': 'Биатлон'},
    'curator_rowing': {'password': 'rowing_pass', 'role': 'curator', 'sport': 'Академическая гребля'}
}

# Function to get athlete basic data based on role
def get_athlete_data(user_info, athlete_id=None, search_name=None, min_age=None, max_age=None, sport_filter=None):
    query = 'SELECT * FROM athletes'
    params = {}
    conditions = []
    if user_info['role'] == 'curator':
        conditions.append('"Вид спорта" = :sport')
        params['sport'] = user_info['sport']
    if athlete_id:
        conditions.append('ID = :id')
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
    return conn.query(query, params=params)

# Function to get measurements based on role
def get_measurements(user_info, athlete_id=None):
    query = 'SELECT * FROM measurements'
    params = {}
    conditions = []
    if user_info['role'] == 'curator':
        sport = user_info['sport']
        athlete_ids_df = get_athlete_data(user_info)
        if not athlete_ids_df.empty:
            athlete_ids = tuple(athlete_ids_df['id'].tolist())
            conditions.append(f'Athlete_ID IN {athlete_ids}')
    if athlete_id:
        conditions.append('Athlete_ID = :id')
        params['id'] = athlete_id
    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)
    return conn.query(query, params=params)

# Function to add measurement
def add_measurement(user_info, athlete_id, date, updates):
    if user_info['role'] == 'leader' or (user_info['role'] == 'curator' and get_athlete_data(user_info, athlete_id)['Вид спорта'].values[0] == user_info['sport']):
        with conn.session as session:
            for metric, value in updates.items():
                if metric in metrics_list:
                    session.execute(text('INSERT INTO measurements (Athlete_ID, "Date", Metric, Value) VALUES (:Athlete_ID, :Date, :Metric, :Value)'), 
                                    {'Athlete_ID': athlete_id, 'Date': date, 'Metric': metric, 'Value': value})
            session.commit()
        return True
    return False

# Function to update athlete
def update_athlete(user_info, athlete_id, updates):
    if user_info['role'] == 'leader' or (user_info['role'] == 'curator' and get_athlete_data(user_info, athlete_id)['Вид спорта'].values[0] == user_info['sport']):
        set_clause = []
        params = {'id': athlete_id}
        for key, value in updates.items():
            set_clause.append(f'"{key}" = :{key}')
            params[key] = value
        query = text(f'UPDATE athletes SET {", ".join(set_clause)} WHERE ID = :id')
        with conn.session as session:
            session.execute(query, params)
            session.commit()
        return True
    return False

# Function to visualize average metrics per sport or overall (latest measurements)
@st.cache_data
def visualize_average_metrics(user_info):
    data = get_measurements(user_info)
    if data.empty:
        return None
    
    # Get latest date per athlete per metric
    data['date'] = pd.to_datetime(data['date'])
    latest = data.loc[data.groupby(['athlete_id', 'metric'])['date'].idxmax()]
    pivot = latest.pivot(index='athlete_id', columns='metric', values='value')
    athletes_with_sport = get_athlete_data(user_info)[['id', 'Вид спорта']].set_index('id')
    combined = pivot.join(athletes_with_sport)
    averages = combined.groupby('Вид спорта').mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    averages.plot(kind='bar', ax=ax)
    ax.set_title('Средние показатели по видам спорта (последние измерения)')
    ax.set_ylabel('Значения')
    ax.set_xlabel('Вид спорта')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Function to visualize individual athlete metrics as a radar chart (latest)
@st.cache_data
def visualize_athlete_metrics(user_info, athlete_id):
    data = get_measurements(user_info, athlete_id)
    if data.empty:
        return None
    
    # Get latest measurements
    data['date'] = pd.to_datetime(data['date'])
    latest = data.loc[data.groupby('metric')['date'].idxmax()]
    values = latest.set_index('metric')['value']
    
    metrics = metrics_list
    vals = [values.get(m, 0) for m in metrics]
    
    # Normalize
    normalized = [(v - ranges[m][0]) / (ranges[m][1] - ranges[m][0]) for m, v in zip(metrics, vals)]
    
    angles = [n / float(len(metrics)) * 2 * 3.14159 for n in range(len(metrics))]
    angles += angles[:1]
    normalized += normalized[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, normalized, color='blue', alpha=0.25)
    ax.plot(angles, normalized, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    athlete_name = get_athlete_data(user_info, athlete_id)['Имя'].values[0]
    ax.set_title(f"Показатели спортсмена {athlete_name} (последние)")
    plt.tight_layout()
    return fig

# Function: Visualize progress over time for an athlete
@st.cache_data
def visualize_progress(user_info, athlete_id):
    data = get_measurements(user_info, athlete_id)
    if data.empty:
        return None
    
    data['date'] = pd.to_datetime(data['date'])
    fig, axs = plt.subplots(len(metrics_list), 1, figsize=(10, 15), sharex=True)
    fig.suptitle(f'Прогресс спортсмена ID {athlete_id} со временем')
    
    for i, metric in enumerate(metrics_list):
        metric_data = data[data['metric'] == metric].sort_values('date')
        if not metric_data.empty:
            axs[i].plot(metric_data['date'], metric_data['value'], marker='o')
            axs[i].set_ylabel(metric)
            axs[i].grid(True)
    
    axs[-1].set_xlabel('Дата')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Function to generate image from fig
def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

# Function to generate PDF report with charts
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
    c.drawString(1*inch, height - 1*inch, "Паспорт спортсмена")
    
    # Athlete info
    c.setFont("Helvetica", 12)
    y = height - 1.5*inch
    c.drawString(1*inch, y, f"ID: {athlete['id']}")
    y -= 0.5*inch
    c.drawString(1*inch, y, f"Имя: {athlete['Имя']}")
    y -= 0.5*inch
    c.drawString(1*inch, y, f"Возраст: {athlete['Возраст']}")
    y -= 0.5*inch
    c.drawString(1*inch, y, f"Вид спорта: {athlete['Вид спорта']}")
    y -= 0.5*inch# Latest physical metrics
    latest_data = get_measurements({'role': 'leader'}, athlete_id)
    latest_data['date'] = pd.to_datetime(latest_data['date'])
    latest = latest_data.loc[latest_data.groupby('metric')['date'].idxmax()]
    c.drawString(1*inch, y, "Последние показатели физического тестирования:")
    y -= 0.5*inch
    for _, row in latest.iterrows():
        c.drawString(1.5*inch, y, f"{row['metric']}: {row['value']} (Дата: {row['date'].date()})")
        y -= 0.3*inch
    
    # Add radar chart
    radar_fig = visualize_athlete_metrics({'role': 'leader'}, athlete_id)
    if radar_fig:
        radar_img = fig_to_image(radar_fig)
        img_reader = ImageReader(radar_img)
        c.drawImage(img_reader, 1*inch, y - 4*inch, width=4*inch, height=4*inch)
        y -= 4.5*inch

    # Add progress chart (simplified, or split if too large)
    progress_fig = visualize_progress({'role': 'leader'}, athlete_id)
    if progress_fig:
        c.showPage()  # New page for progress
        y = height - 1*inch
        progress_img = fig_to_image(progress_fig)
        img_reader = ImageReader(progress_img)
        c.drawImage(img_reader, 1*inch, y - 8*inch, width=6*inch, height=8*inch)
    
    # Save the PDF
    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

# Custom CSS for themes and Retool-like design
light_theme = """
    <style>
        .stApp {
            background-color: #f8f9fa;
            color: black;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border-radius: 4px;
        }
        .stTextInput > div > div > input {
            border: 1px solid #ced4da;
            border-radius: 4px;
        }
        /* Table styles */
        .dataframe th {
            background-color: #e9ecef;
        }
        .stDataFrame {
            border: 1px solid #dee2e6;
            border-radius: 4px;
        }
        /* Add hover effect */
        .dataframe tr:hover {
            background-color: #f1f3f5;
        }
    </style>
"""

dark_theme = """
    <style>
        .stApp {
            background-color: #1e1e1e;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white;
        }
        .stButton > button {
            background-color: #0d6efd;
            color: white;
            border-radius: 4px;
        }
        .stTextInput > div > div > input {
            background-color: #2c2c2c;
            color: white;
            border: 1px solid #495057;
            border-radius: 4px;
        }
        .stNumberInput > div > div > input {
            background-color: #2c2c2c;
            color: white;
        }
        .stDateInput > div > div > input {
            background-color: #2c2c2c;
            color: white;
        }
        .stSelectbox > div > div > div {
            background-color: #2c2c2c;
            color: white;
        }
        /* Dataframe styles */
        .dataframe {
            color: white;
            border: 1px solid #495057;
            border-radius: 4px;
        }
        .dataframe th {
            background-color: #343a40;
        }
        .dataframe tr:hover {
            background-color: #495057;
        }
    </style>
"""

# Streamlit app
st.title("Цифровой реестр и паспорт спортсмена")

# Theme selection in sidebar
if 'theme' not in st.session_state:
    st.session_state.theme = 'Светлая'

st.sidebar.title("Настройки")
theme_choice = st.sidebar.radio("Тема интерфейса", ['Светлая', 'Тёмная'])
if theme_choice != st.session_state.theme:
    st.session_state.theme = theme_choice
    st.rerun()

# Apply theme
if st.session_state.theme == 'Тёмная':
    st.markdown(dark_theme, unsafe_allow_html=True)
else:
    st.markdown(light_theme, unsafe_allow_html=True)

    # Sidebar filters
    st.sidebar.title("Фильтры")
    search_name = st.sidebar.text_input("Поиск по имени")
    min_age, max_age = st.sidebar.slider("Возраст", 14, 18, (14, 18))
    if user_info['role'] == 'leader':
        sport_filter = st.sidebar.selectbox("Вид спорта", ['Все'] + list(athletes_df['Вид спорта'].unique()))
        sport_filter = None if sport_filter == 'Все' else sport_filter
    else:
        sport_filter = None

    # Main layout with columns for Retool-like design
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("Список спортсменов")
        athletes_data = get_athlete_data(user_info, search_name=search_name, min_age=min_age, max_age=max_age, sport_filter=sport_filter)
        # Editable table
        edited_df = st.data_editor(athletes_data, num_rows="dynamic", use_container_width=True)
        if st.button("Сохранить изменения в списке"):
            for idx, row in edited_df.iterrows():
                if not row.equals(athletes_data.loc[idx]):
                    updates = row.drop('id').to_dict()
                    update_athlete(user_info, row['id'], updates)
            st.success("Изменения сохранены!")
            st.rerun()

        # Select athlete
        st.session_state.selected_athlete_id = st.selectbox("Выберите спортсмена по ID", athletes_data['id'].unique(), key="athlete_select")

    if 'selected_athlete_id' in st.session_state:
        selected_athlete_id = st.session_state.selected_athlete_id

        with right_col:
            st.subheader("Детали спортсмена")
            athlete_details = get_athlete_data(user_info, selected_athlete_id)
            st.write(athlete_details)

            st.subheader("Измерения")
            meas_data = get_measurements(user_info, selected_athlete_id)
            st.dataframe(meas_data)

            # Add new measurement
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
                        st.rerun()
                    else:
                        st.error("Нет доступа для обновления")

        # Visualizations below
        st.subheader("Визуализации")
        col1, col2 = st.columns(2)
        with col1:
            avg_fig = visualize_average_metrics(user_info)
            if avg_fig:
                st.pyplot(avg_fig)

            athlete_fig = visualize_athlete_metrics(user_info, selected_athlete_id)
            if athlete_fig:
                st.pyplot(athlete_fig)

        with col2:
            progress_fig = visualize_progress(user_info, selected_athlete_id)
            if progress_fig:
                st.pyplot(progress_fig)

        # Reports
        st.subheader("Отчеты")
        if st.button("Сгенерировать PDF"):
            pdf_buffer = generate_pdf_report(selected_athlete_id)
            if pdf_buffer:
                b64 = base64.b64encode(pdf_buffer.read()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="athlete_{selected_athlete_id}_report.pdf">Скачать PDF</a>'st.markdown(href, unsafe_allow_html=True)

        if st.button("Экспорт данных в CSV"):
            csv = meas_data.to_csv(index=False).encode('utf-8')
            st.download_button("Скачать CSV", csv, "measurements.csv", "text/csv")
