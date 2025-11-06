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

# Mock data generation
sports = ['Лыжные гонки', 'Биатлон', 'Академическая гребля']

# Russian names (first and last)
first_names_male = ['Александр', 'Дмитрий', 'Иван', 'Максим', 'Никита', 'Сергей', 'Артем', 'Егор', 'Кирилл', 'Михаил']
last_names_male = ['Иванов', 'Петров', 'Сидоров', 'Кузнецов', 'Смирнов', 'Попов', 'Васильев', 'Михайлов', 'Новиков', 'Федоров']
first_names_female = ['Анастасия', 'Мария', 'София', 'Анна', 'Дарья', 'Виктория', 'Елизавета', 'Полина', 'Ксения', 'Екатерина']
last_names_female = ['Иванова', 'Петрова', 'Сидорова', 'Кузнецова', 'Смирнова', 'Попова', 'Васильева', 'Михайлова', 'Новикова', 'Федорова']

def generate_russian_name(gender='male'):
    if gender == 'male':
        return random.choice(first_names_male) + ' ' + random.choice(last_names_male)
    else:
        return random.choice(first_names_female) + ' ' + random.choice(last_names_female)

# Generate mock athletes: 15 per sport, mixed gender, age 14-18
athletes = []
athlete_id = 1
for sport in sports:
    for _ in range(15):
        gender = random.choice(['male', 'female'])
        name = generate_russian_name(gender)
        age = random.randint(14, 18)
        athletes.append({
            'ID': athlete_id,
            'Имя': name,
            'Возраст': age,
            'Вид спорта': sport,
        })
        athlete_id += 1

# Create DataFrame for athletes basic info
athletes_df = pd.DataFrame(athletes)

# Generate mock historical measurements: 4 per athlete over the past year
measurements = []
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
for aid in athletes_df['ID']:
    for i in range(4):  # 4 measurements
        date = today - timedelta(days=random.randint(30*i, 30*(i+1) + 30))
        for metric in metrics_list:
            base_value = random.uniform(ranges[metric][0], ranges[metric][1])
            # Simulate progress: slight improvement over time
            progress_factor = 1 + (0.05 * (3 - i))  # Newer measurements better
            value = round(base_value * progress_factor, 1) if metric not in ['ЧСС в покое', 'Максимальная ЧСС'] else int(base_value * progress_factor)
            # For HR, lower resting is better, but for simplicity, random
            if metric == 'ЧСС в покое':
                value = int(base_value * (1 - 0.05 * (3 - i)))  # Decrease over time
            measurements.append({
                'Athlete_ID': aid,
                'Date': date.date(),
                'Metric': metric,
                'Value': value
            })

# Create DataFrame for measurements (long format)
measurements_df = pd.DataFrame(measurements)

# User roles and access
users = {
    'project_leader': {'password': 'admin_pass', 'role': 'leader'},  # Full access
    'curator_ski_racing': {'password': 'ski_pass', 'role': 'curator', 'sport': 'Лыжные гонки'},
    'curator_biathlon': {'password': 'biathlon_pass', 'role': 'curator', 'sport': 'Биатлон'},
    'curator_rowing': {'password': 'rowing_pass', 'role': 'curator', 'sport': 'Академическая гребля'}
}

# Function to get athlete basic data based on role
def get_athlete_data(user_info, athlete_id=None):
    if user_info['role'] == 'leader':
        if athlete_id:
            return athletes_df[athletes_df['ID'] == athlete_id]
        return athletes_df
    elif user_info['role'] == 'curator':
        sport = user_info['sport']
        filtered = athletes_df[athletes_df['Вид спорта'] == sport]
        if athlete_id:
            return filtered[filtered['ID'] == athlete_id]
        return filtered
    return pd.DataFrame()

# Function to get measurements based on role
def get_measurements(user_info, athlete_id=None):
    if user_info['role'] == 'leader':
        if athlete_id:
            return measurements_df[measurements_df['Athlete_ID'] == athlete_id]
        return measurements_df
    elif user_info['role'] == 'curator':
        sport = user_info['sport']
        athlete_ids = athletes_df[athletes_df['Вид спорта'] == sport]['ID']
        filtered = measurements_df[measurements_df['Athlete_ID'].isin(athlete_ids)]
        if athlete_id:
            return filtered[filtered['Athlete_ID'] == athlete_id]
        return filtered
    return pd.DataFrame()

# Function to update measurements (add new entry)
def add_measurement(user_info, athlete_id, date, updates):
    if user_info['role'] == 'leader' or (user_info['role'] == 'curator' and athletes_df.loc[athletes_df['ID'] == athlete_id, 'Вид спорта'].values[0] == user_info['sport']):
        new_rows = []
        for metric, value in updates.items():
            if metric in metrics_list:
                new_rows.append({'Athlete_ID': athlete_id, 'Date': date, 'Metric': metric, 'Value': value})
        global measurements_df
        measurements_df = pd.concat([measurements_df, pd.DataFrame(new_rows)], ignore_index=True)
        return True
    return False

# Function to visualize average metrics per sport or overall (latest measurements)
@st.cache_data
def visualize_average_metrics(user_info):
    data = get_measurements(user_info)
    if data.empty:
        return None
    
    # Get latest date per athlete per metric
    latest = data.loc[data.groupby(['Athlete_ID', 'Metric'])['Date'].idxmax()]
    pivot = latest.pivot(index='Athlete_ID', columns='Metric', values='Value')
    athletes_with_sport = athletes_df.set_index('ID')[['Вид спорта']]
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
    latest = data.loc[data.groupby('Metric')['Date'].idxmax()]
    values = latest.set_index('Metric')['Value']
    
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
    athlete_name = athletes_df[athletes_df['ID'] == athlete_id]['Имя'].values[0]
    ax.set_title(f"Показатели спортсмена {athlete_name} (последние)")
    plt.tight_layout()
    return fig

# Function: Visualize progress over time for an athlete
@st.cache_data
def visualize_progress(user_info, athlete_id):
    data = get_measurements(user_info, athlete_id)
    if data.empty:
        return None
    
    fig, axs = plt.subplots(len(metrics_list), 1, figsize=(10, 15), sharex=True)
    fig.suptitle(f'Прогресс спортсмена ID {athlete_id} со временем')
    
    for i, metric in enumerate(metrics_list):
        metric_data = data[data['Metric'] == metric].sort_values('Date')
        if not metric_data.empty:
            axs[i].plot(metric_data['Date'], metric_data['Value'], marker='o')
            axs[i].set_ylabel(metric)
            axs[i].grid(True)
    
    axs[-1].set_xlabel('Дата')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Function to generate PDF report
def generate_pdf_report(athlete_id):
    athlete = athletes_df[athletes_df['ID'] == athlete_id]
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
    c.drawString(1*inch, y, f"ID: {athlete['ID']}")
    y -= 0.5*inch
    c.drawString(1*inch, y, f"Имя: {athlete['Имя']}")
    y -= 0.5*inch
    c.drawString(1*inch, y, f"Возраст: {athlete['Возраст']}")
    y -= 0.5*inch
    c.drawString(1*inch, y, f"Вид спорта: {athlete['Вид спорта']}")
    y -= 0.5*inch
    
    # Latest physical metrics
    latest_data = measurements_df[measurements_df['Athlete_ID'] == athlete_id].loc[measurements_df[measurements_df['Athlete_ID'] == athlete_id].groupby('Metric')['Date'].idxmax()]
    c.drawString(1*inch, y, "Последние показатели физического тестирования:")
    y -= 0.5*inch
    for _, row in latest_data.iterrows():
        c.drawString(1.5*inch, y, f"{row['Metric']}: {row['Value']} (Дата: {row['Date']})")
        y -= 0.3*inch
    
    # Save the PDF
    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

# Custom CSS for themes
light_theme = """
    <style>
        .stApp {
            background-color: white;
            color: black;
        }
        .stButton > button {
            background-color: #f0f0f0;
            color: black;
        }
        /* Add more styles as needed */
    </style>
"""

dark_theme = """
    <style>
        .stApp {
            background-color: #121212;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white;
        }
        .stButton > button {
            background-color: #333333;
            color: white;
        }
        .stTextInput > div > div > input {
            background-color: #333333;
            color: white;
        }
        .stNumberInput > div > div > input {
            background-color: #333333;
            color: white;
        }
        .stDateInput > div > div > input {
            background-color: #333333;
            color: white;
        }
        .stSelectbox > div > div > div {
            background-color: #333333;
            color: white;
        }
        /* Dataframe styles */
        .dataframe {
            color: white;
        }
        /* Add more styles for dark theme */
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

# Authentication
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
            st.rerun()
        else:
            st.error("Неверный логин или пароль")
else:
    user_info = st.session_state.user_info
    st.sidebar.markdown(f"**Пользователь:** {user_info['role']}")
    if st.sidebar.button("Выйти"):
        st.session_state.user_info = None
        st.rerun()

    # Use tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["Список спортсменов", "Измерения и обновления", "Визуализации", "Отчеты"])

    with tab1:
        st.subheader("Список спортсменов")
        athletes_data = get_athlete_data(user_info)
        st.dataframe(athletes_data.style.set_table_styles([{'selector': 'tr:hover', 'props': [('background-color', '#f5f5f5' if st.session_state.theme == 'Светлая' else '#333333')]}]))

        # Select athlete for other tabs
        st.session_state.selected_athlete_id = st.selectbox("Выберите спортсмена по ID", athletes_data['ID'].unique(), key="athlete_select")

    if 'selected_athlete_id' in st.session_state:
        athlete_id = st.session_state.selected_athlete_id

        with tab2:
            if athlete_id:
                # Measurements
                meas_data = get_measurements(user_info, athlete_id)
                st.subheader("Измерения")
                st.dataframe(meas_data)

                # Add new measurement with expander
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
                        if add_measurement(user_info, athlete_id, new_date, updates):
                            st.success("Измерения добавлены!")
                            st.rerun()
                        else:
                            st.error("Нет доступа для обновления")

        with tab3:
            if athlete_id:
                st.subheader("Визуализация среднего по видам спорта")
                avg_fig = visualize_average_metrics(user_info)
                if avg_fig:
                    st.pyplot(avg_fig)

                st.subheader("Визуализация показателей спортсмена (радар)")
                athlete_fig = visualize_athlete_metrics(user_info, athlete_id)
                if athlete_fig:
                    st.pyplot(athlete_fig)

                st.subheader("Визуализация прогресса")
                progress_fig = visualize_progress(user_info, athlete_id)
                if progress_fig:
                    st.pyplot(progress_fig)

        with tab4:
            if athlete_id:
                st.subheader("Генерация PDF-отчета")
                if st.button("Сгенерировать PDF"):
                    pdf_buffer = generate_pdf_report(athlete_id)
                    if pdf_buffer:
                        b64 = base64.b64encode(pdf_buffer.read()).decode()
                        href = f'<a href="data:application/pdf;base64,{b64}" download="athlete_{athlete_id}_report.pdf">Скачать PDF</a>'
                        st.markdown(href, unsafe_allow_html=True)
