# app.py
import streamlit as st
import pandas as pd
import io
import random
import sqlite3
from faker import Faker
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime, timezone

# ---------- Константы ----------
SPORTS = {
    'ski': 'Лыжные гонки',
    'biathlon': 'Биатлон',
    'rowing': 'Академическая гребля'
}

DB_FILE = 'athletes.db'

# ---------- Безопасный доступ к RerunException ----------
try:
    # Streamlit internal exception to request rerun
    from streamlit.runtime.scriptrunner import RerunException
except Exception:
    RerunException = None

# ---------- SQLite helpers ----------
def ensure_tables_sqlite():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cs = conn.cursor()
    cs.execute("""
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT,
          role TEXT,
          sport TEXT,
          password TEXT
        )
    """)
    cs.execute("""
        CREATE TABLE IF NOT EXISTS athletes (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          first_name TEXT,
          last_name TEXT,
          birth_year INTEGER,
          sex TEXT,
          sport TEXT,
          region TEXT,
          best_result TEXT,
          vo2max REAL,
          max_strength REAL,
          lean_mass REAL,
          pano REAL,
          hr_rest INTEGER,
          hr_max INTEGER,
          stroke_volume REAL,
          created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def query_df_sqlite(sql, params=None):
    params = params or ()
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    try:
        df = pd.read_sql_query(sql, conn, params=params)
        return df
    finally:
        conn.close()

def execute_sqlite(sql, params=None):
    params = params or ()
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cs = conn.cursor()
    try:
        cs.execute(sql, params)
        conn.commit()
    finally:
        cs.close()
        conn.close()

# ---------- Seed mock data ----------
def seed_if_empty_sqlite():
    ensure_tables_sqlite()
    df = query_df_sqlite("SELECT COUNT(*) AS cnt FROM users")
    if int(df['cnt'].iloc[0]) > 0:
        return

    fake = Faker('ru_RU')
    demo_users = [
        ('leader', 'leader', None, 'leaderpass'),
        ('curator_ski', 'curator', 'ski', 'curpass1'),
        ('curator_biathlon', 'curator', 'biathlon', 'curpass2'),
        ('curator_rowing', 'curator', 'rowing', 'curpass3'),
        ('coach_ski', 'coach', 'ski', 'coach1'),
        ('coach_biathlon', 'coach', 'biathlon', 'coach2'),
        ('coach_rowing', 'coach', 'rowing', 'coach3')
    ]
    for uname, role, sport, pwd in demo_users:
        execute_sqlite("INSERT INTO users (username, role, sport, password) VALUES (?,?,?,?)",
                       (uname, role, sport, pwd))

    birth_years = [2007,2008,2009,2010,2006]
    for sport_key in SPORTS.keys():
        for _ in range(15):
            first = fake.first_name()
            last = fake.last_name()
            by = int(random.choice(birth_years))
            sex = random.choice(['M','F'])
            region = fake.region()
            best = f"{random.randint(1,10)} место на юнош. первенстве"
            vo2 = round(random.uniform(45,80),1)
            ms = round(random.uniform(40,200),1)
            lm = round(random.uniform(40,80),1)
            pano = round(random.uniform(3,8),2)
            hr_rest = random.randint(30,70)
            hr_max = random.randint(160,210)
            sv = round(random.uniform(50,120),1)
            created_at = datetime.now(timezone.utc).isoformat()
            execute_sqlite(
                "INSERT INTO athletes (first_name,last_name,birth_year,sex,sport,region,best_result,vo2max,max_strength,lean_mass,pano,hr_rest,hr_max,stroke_volume,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (first,last,by,sex,sport_key,region,best,vo2,ms,lm,pano,hr_rest,hr_max,sv,created_at)
            )

# ---------- Authentication ----------
def authenticate_sqlite(username, password):
    if not username or not password:
        return None
    df = query_df_sqlite("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    if df.shape[0] == 0:
        return None
    row = df.iloc[0]
    return {'id': int(row.id), 'username': row.username, 'role': row.role, 'sport': row.sport}

# ---------- PDF generation ----------
def generate_pdf_bytes_sqlite(sport_key, df_athletes):
    buf = io.BytesIO()
    p = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 40
    p.setFont("Helvetica-Bold", 14)
    p.drawString(40, y, f"Отчёт по виду спорта: {SPORTS.get(sport_key,sport_key)}")
    p.setFont("Helvetica", 10)
    y -= 30
    for _, a in df_athletes.iterrows():
        last = a.get('last_name')
        first = a.get('first_name')
        by = int(a.get('birth_year') or 0)
        region = a.get('region','')
        line = f"{last} {first}, {by}, Регион: {region}"
        p.drawString(40, y, line)
        y -= 14
        sub = (f"МПК: {a.get('vo2max')}  Сила: {a.get('max_strength')}  БМ: {a.get('lean_mass')}  "
               f"ПАНО: {a.get('pano')}  ЧСС пок: {int(a.get('hr_rest') or 0)}  ЧСС макс: {int(a.get('hr_max') or 0)}  УО: {a.get('stroke_volume')}")
        p.drawString(56, y, sub)
        y -= 18
        if y < 80:
            p.showPage()
            y = h - 40
            p.setFont("Helvetica", 10)
    p.showPage()
    p.save()
    buf.seek(0)
    return buf.read()

# ---------- UI ----------
st.set_page_config(page_title="Реестр олимпийского резерва (Demo)", layout="wide")
st.title("Цифровой реестр / Паспорт спортсмена — demo (SQLite)")

# Инициализация БД и seed
with st.spinner("Инициализация локальной базы данных..."):
    try:
        seed_if_empty_sqlite()
    except Exception as e:
        st.error("Ошибка инициализации локальной БД: " + str(e))
        st.stop()

# Сессия пользователя
if 'user' not in st.session_state:
    st.session_state['user'] = None
if 'login_attempt' not in st.session_state:
    st.session_state['login_attempt'] = False

# Вход
if st.session_state['user'] is None:
    st.sidebar.header("Вход")
    username = st.sidebar.text_input("Логин")
    password = st.sidebar.text_input("Пароль", type="password")
    if st.sidebar.button("Войти"):
        user = authenticate_sqlite(username.strip(), password.strip())
        if user:
            st.session_state['user'] = user
            st.session_state['login_attempt'] = True
            # безопасный rerun: используем RerunException если доступен, иначе st.experimental_set_query_params + st.stop
            if RerunException is not None:
                raise RerunException
            else:
                # установка параметра запроса и остановка текущего запуска — простой обход
                st.experimental_set_query_params(logged_in='1')
                st.stop()
        else:
            st.sidebar.error("Неверные учетные данные")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Тестовые учётки (демо):")
    st.sidebar.write("- leader / leaderpass")
    st.sidebar.write("- curator_ski / curpass1")
    st.sidebar.write("- curator_biathlon / curpass2")
    st.sidebar.write("- curator_rowing / curpass3")
    st.sidebar.write("- coach_ski / coach1")
    # прерываем дальнейшее выполнение пока пользователь не вошёл
    st.stop()

# Если мы здесь — пользователь авторизован
user = st.session_state['user']
st.sidebar.success(f"Выполнен вход: {user['username']} ({user['role']})")

# Кнопка выхода
if st.sidebar.button("Выйти"):
    st.session_state['user'] = None
    # безопасный rerun при выходе
    if RerunException is not None:
        raise RerunException
    else:
        st.experimental_set_query_params(logged_in='0')
        st.stop()

page = st.sidebar.radio("Раздел", ["Dashboard", "Список спортсменов", "Паспорт спортсмена", "Генерация отчёта"])

def load_athletes_for_user(u):
    if u['role'] == 'leader':
        return query_df_sqlite("SELECT * FROM athletes ORDER BY sport, last_name")
    elif u['role'] == 'curator':
        return query_df_sqlite("SELECT * FROM athletes WHERE sport=? ORDER BY last_name", (u['sport'],))
    else:
        return query_df_sqlite("SELECT id, first_name, last_name, birth_year, sport FROM athletes WHERE sport=? ORDER BY last_name", (u['sport'],))

# Dashboard
if page == "Dashboard":
    st.header("Панель управления (демо)")
    if user['role'] == 'leader':
        counts = {}
        for s in SPORTS.keys():
            dfc = query_df_sqlite("SELECT COUNT(*) AS cnt FROM athletes WHERE sport=?", (s,))
            counts[s] = int(dfc['cnt'].iloc[0])
        cols = st.columns(3)
        i = 0
        for s, name in SPORTS.items():
            cols[i].metric(name, counts[s])
            i += 1
        st.markdown("---")
        st.write("Полный реестр доступен в 'Список спортсменов'")
    elif user['role'] == 'curator':
        st.write(f"Куратор вида: {SPORTS.get(user['sport'])}")
        df = load_athletes_for_user(user)
        # нормализуем имена столбцов для отображения
        if not df.empty:
            display_df = df.rename(columns={
                'id':'ID','last_name':'Фамилия','first_name':'Имя','birth_year':'Год'
            })
            st.dataframe(display_df[['Фамилия','Имя','Год','region']].head(100))
        else:
            st.info("Нет данных.")
    else:
        st.write(f"Тренер вида: {SPORTS.get(user['sport'])}")
        st.info("Тренерам прямой доступ к персональным данным закрыт. Для получения отчёта обратитесь к куратору или руководителю.")

# Список спортсменов
if page == "Список спортсменов":
    st.header("Список спортсменов")
    df = load_athletes_for_user(user)
    if df.empty:
        st.info("Нет данных для отображения в рамках вашей роли.")
    else:
        if user['role'] == 'coach':
            st.dataframe(df[['id','last_name','first_name','birth_year']])
        else:
            st.dataframe(df)

# Паспорт спортсмена
if page == "Паспорт спортсмена":
    st.header("Паспорт спортсмена")
    athlete_id = st.number_input("Введите ID спортсмена", min_value=1, step=1)
    if st.button("Загрузить паспорт"):
        try:
            df = query_df_sqlite("SELECT * FROM athletes WHERE id=?", (athlete_id,))
            if df.shape[0] == 0:
                st.error("Спортсмен не найден")
            else:
                a = df.iloc[0].to_dict()
                sport_key = a.get('sport')
                if user['role'] == 'leader' or (user['role'] == 'curator' and user['sport'] == sport_key):
                    st.subheader(f"{a.get('last_name')} {a.get('first_name')}, {int(a.get('birth_year') or 0)}")
                    st.write("Регион:", a.get('region',''))
                    st.write("Вид спорта:", SPORTS.get(sport_key, sport_key))
                    with st.form("edit_form"):
                        best = st.text_input("Лучший результат", value=a.get('best_result',''))
                        vo2 = st.text_input("МПК", value=str(a.get('vo2max','')))
                        maxs = st.text_input("Макс сила", value=str(a.get('max_strength','')))
                        lm = st.text_input("Безжировая масса", value=str(a.get('lean_mass','')))
                        pano = st.text_input("ПАНО", value=str(a.get('pano','')))
                        hr_rest = st.text_input("ЧСС в покое", value=str(a.get('hr_rest','')))
                        hr_max = st.text_input("Макс ЧСС", value=str(a.get('hr_max','')))
                        sv = st.text_input("Ударный объём", value=str(a.get('stroke_volume','')))
                        submitted = st.form_submit_button("Сохранить")
                        if submitted:
                            try:
                                execute_sqlite(
                                    "UPDATE athletes SET best_result=?, vo2max=?, max_strength=?, lean_mass=?, pano=?, hr_rest=?, hr_max=?, stroke_volume=? WHERE id=?",
                                    (best, float(vo2 or 0), float(maxs or 0), float(lm or 0), float(pano or 0),
                                     int(hr_rest or 0), int(hr_max or 0), float(sv or 0), athlete_id)
                                )
                                st.success("Паспорт обновлён")
                            except Exception as ee:
                                st.error("Ошибка при сохранении: " + str(ee))
                else:
                    st.error("У вас нет права просматривать/редактировать паспорт этого спортсмена")
        except Exception as e:
            st.error("Ошибка: " + str(e))

# Генерация отчёта
if page == "Генерация отчёта":
    st.header("Генерация PDF отчёта по виду спорта")
    sport_choice = st.selectbox("Выберите вид спорта", list(SPORTS.keys()), format_func=lambda k: SPORTS[k])
    if st.button("Сгенерировать PDF"):
        if user['role'] == 'coach':
            st.error("У тренера нет права генерировать отчёты. Обратитесь к куратору или руководителю.")
        elif user['role'] == 'curator' and user['sport'] != sport_choice:
            st.error("Куратор может генерировать отчёт только по своему виду спорта.")
        else:
            try:
                df = query_df_sqlite("SELECT * FROM athletes WHERE sport=? ORDER BY last_name", (sport_choice,))
                pdf_bytes = generate_pdf_bytes_sqlite(sport_choice, df)
                st.success("Отчёт сгенерирован")
                st.download_button("Скачать PDF", data=pdf_bytes, file_name=f"report_{sport_choice}.pdf", mime="application/pdf")
            except Exception as e:
                st.error("Ошибка генерации отчёта: " + str(e))
