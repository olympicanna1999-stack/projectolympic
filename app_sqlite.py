# app_sqlite.py
import streamlit as st
import pandas as pd
import io
import random
import sqlite3
from faker import Faker
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

# ---------- Константы ----------
SPORTS = {
    'ski': 'Лыжные гонки',
    'biathlon': 'Биатлон',
    'rowing': 'Академическая гребля'
}

# ---------- Хранилище: пытаемся Snowflake, иначе SQLite ----------
USE_SNOWFLAKE = False
try:
    if "snowflake" in st.secrets:
        # импорт внутри try чтобы не ломать среду без snowflake-connector
        import snowflake.connector
        # пробуем подключиться и сразу закрыть
        sf = st.secrets["snowflake"]
        conn_test = snowflake.connector.connect(
            user=sf["user"],
            password=sf["password"],
            account=sf["account"],
            warehouse=sf["warehouse"],
            database=sf["database"],
            schema=sf["schema"],
            role=sf.get("role")
        )
        conn_test.close()
        USE_SNOWFLAKE = True
except Exception as e:
    USE_SNOWFLAKE = False

# ---------- Абстракции запросов (унифицированный интерфейс) ----------
def ensure_tables(conn_mode):
    if conn_mode == "snowflake":
        sf = st.secrets["snowflake"]
        import snowflake.connector
        conn = snowflake.connector.connect(
            user=sf["user"],
            password=sf["password"],
            account=sf["account"],
            warehouse=sf["warehouse"],
            database=sf["database"],
            schema=sf["schema"],
            role=sf.get("role")
        )
        cs = conn.cursor()
        try:
            cs.execute("""
                CREATE TABLE IF NOT EXISTS USERS (
                  ID INTEGER AUTOINCREMENT PRIMARY KEY,
                  USERNAME STRING,
                  ROLE STRING,
                  SPORT STRING,
                  PASSWORD STRING
                )
            """)
            cs.execute("""
                CREATE TABLE IF NOT EXISTS ATHLETES (
                  ID INTEGER AUTOINCREMENT PRIMARY KEY,
                  FIRST_NAME STRING,
                  LAST_NAME STRING,
                  BIRTH_YEAR INTEGER,
                  SEX STRING,
                  SPORT STRING,
                  REGION STRING,
                  BEST_RESULT STRING,
                  VO2MAX FLOAT,
                  MAX_STRENGTH FLOAT,
                  LEAN_MASS FLOAT,
                  PANO FLOAT,
                  HR_REST INTEGER,
                  HR_MAX INTEGER,
                  STROKE_VOLUME FLOAT,
                  CREATED_AT TIMESTAMP
                )
            """)
            conn.commit()
        finally:
            cs.close()
            conn.close()
    else:
        # sqlite
        conn = sqlite3.connect('athletes.db', check_same_thread=False)
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

def query_df(conn_mode, sql, params=None):
    params = params or ()
    if conn_mode == "snowflake":
        import snowflake.connector
        sf = st.secrets["snowflake"]
        conn = snowflake.connector.connect(
            user=sf["user"],
            password=sf["password"],
            account=sf["account"],
            warehouse=sf["warehouse"],
            database=sf["database"],
            schema=sf["schema"],
            role=sf.get("role")
        )
        try:
            df = pd.read_sql(sql, conn, params=params)
            return df
        finally:
            conn.close()
    else:
        conn = sqlite3.connect('athletes.db', check_same_thread=False)
        try:
            df = pd.read_sql_query(sql, conn, params=params)
            return df
        finally:
            conn.close()

def execute(conn_mode, sql, params=None):
    params = params or ()
    if conn_mode == "snowflake":
        import snowflake.connector
        sf = st.secrets["snowflake"]
        conn = snowflake.connector.connect(
            user=sf["user"],
            password=sf["password"],
            account=sf["account"],
            warehouse=sf["warehouse"],
            database=sf["database"],
            schema=sf["schema"],
            role=sf.get("role")
        )
        cs = conn.cursor()
        try:
            cs.execute(sql, params)
            conn.commit()
        finally:
            cs.close()
            conn.close()
    else:
        conn = sqlite3.connect('athletes.db', check_same_thread=False)
        cs = conn.cursor()
        try:
            cs.execute(sql, params)
            conn.commit()
        finally:
            cs.close()
            conn.close()

# ---------- Seed mock data ----------
def seed_if_empty(conn_mode):
    ensure_tables(conn_mode)
    if conn_mode == "snowflake":
        df = query_df(conn_mode, "SELECT COUNT(*) AS CNT FROM USERS")
        if int(df.CNT.iloc[0]) > 0:
            return
    else:
        df = query_df(conn_mode, "SELECT COUNT(*) AS CNT FROM users")
        if int(df.CNT.iloc[0]) > 0:
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
        if conn_mode == "snowflake":
            execute(conn_mode, "INSERT INTO USERS (USERNAME, ROLE, SPORT, PASSWORD) VALUES (%s,%s,%s,%s)",
                    (uname, role, sport, pwd))
        else:
            execute(conn_mode, "INSERT INTO users (username, role, sport, password) VALUES (?,?,?,?)",
                    (uname, role, sport, pwd))

    birth_years = [2007,2008,2009,2010,2006]
    for sport_key in SPORTS.keys():
        for _ in range(15):
            first = fake.first_name()
            last = fake.last_name()
            by = random.choice(birth_years)
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
            created_at = datetime.utcnow().isoformat()
            if conn_mode == "snowflake":
                execute(conn_mode,
                        "INSERT INTO ATHLETES (FIRST_NAME,LAST_NAME,BIRTH_YEAR,SEX,SPORT,REGION,BEST_RESULT,VO2MAX,MAX_STRENGTH,LEAN_MASS,PANO,HR_REST,HR_MAX,STROKE_VOLUME,CREATED_AT) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (first,last,by,sex,sport_key,region,best,vo2,ms,lm,pano,hr_rest,hr_max,sv,created_at))
            else:
                execute(conn_mode,
                        "INSERT INTO athletes (first_name,last_name,birth_year,sex,sport,region,best_result,vo2max,max_strength,lean_mass,pano,hr_rest,hr_max,stroke_volume,created_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (first,last,by,sex,sport_key,region,best,vo2,ms,lm,pano,hr_rest,hr_max,sv,created_at))

# ---------- Authentication ----------
def authenticate(conn_mode, username, password):
    if conn_mode == "snowflake":
        df = query_df(conn_mode, "SELECT * FROM USERS WHERE USERNAME=%s AND PASSWORD=%s", (username, password))
        if df.shape[0] == 0:
            return None
        row = df.iloc[0]
        return {'id': int(row.ID), 'username': row.USERNAME, 'role': row.ROLE, 'sport': row.SPORT}
    else:
        df = query_df(conn_mode, "SELECT * FROM users WHERE username=? AND password=?", (username, password))
        if df.shape[0] == 0:
            return None
        row = df.iloc[0]
        return {'id': int(row.id), 'username': row.username, 'role': row.role, 'sport': row.sport}

# ---------- PDF ----------
def generate_pdf_bytes(sport_key, df_athletes):
    buf = io.BytesIO()
    p = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 40
    p.setFont("Helvetica-Bold", 14)
    p.drawString(40, y, f"Отчёт по виду спорта: {SPORTS.get(sport_key,sport_key)}")
    p.setFont("Helvetica", 10)
    y -= 30
    for _, a in df_athletes.iterrows():
        # normalize column names for both DBs
        last = a.get('LAST_NAME') or a.get('last_name')
        first = a.get('FIRST_NAME') or a.get('first_name')
        by = int(a.get('BIRTH_YEAR') or a.get('birth_year') or 0)
        region = a.get('REGION') or a.get('region') or ''
        vo2 = a.get('VO2MAX') or a.get('vo2max') or ''
        ms = a.get('MAX_STRENGTH') or a.get('max_strength') or ''
        lm = a.get('LEAN_MASS') or a.get('lean_mass') or ''
        pano = a.get('PANO') or a.get('pano') or ''
        hr_rest = int(a.get('HR_REST') or a.get('hr_rest') or 0)
        hr_max = int(a.get('HR_MAX') or a.get('hr_max') or 0)
        sv = a.get('STROKE_VOLUME') or a.get('stroke_volume') or ''
        line = f"{last} {first}, {by}, Регион: {region}"
        p.drawString(40, y, line)
        y -= 14
        sub = f"МПК: {vo2}  Сила: {ms}  БМ: {lm}  ПАНО: {pano}  ЧСС пок: {hr_rest}  ЧСС макс: {hr_max}  УО: {sv}"
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
st.set_page_config(page_title="Реестр резервов", layout="wide")

mode = "snowflake" if USE_SNOWFLAKE else "sqlite"
st.sidebar.info(f"Режим хранения: {mode}")

# Инициализация данных
with st.spinner("Проверка и инициализация хранилища..."):
    try:
        seed_if_empty(mode)
    except Exception as e:
        st.error("Ошибка инициализации: " + str(e))
        st.stop()

# Сессия пользователя
if 'user' not in st.session_state:
    st.session_state['user'] = None

# Авторизация
if st.session_state['user'] is None:
    st.sidebar.header("Вход")
    username = st.sidebar.text_input("Логин")
    password = st.sidebar.text_input("Пароль", type="password")
    if st.sidebar.button("Войти"):
        user = authenticate(mode, username.strip(), password.strip())
        if user:
            st.session_state['user'] = user
            st.experimental_rerun()
        else:
            st.sidebar.error("Неверные учётные данные")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Тестовые учётки (демо):")
    st.sidebar.write("- leader / leaderpass")
    st.sidebar.write("- curator_ski / curpass1")
    st.sidebar.write("- curator_biathlon / curpass2")
    st.sidebar.write("- curator_rowing / curpass3")
    st.sidebar.write("- coach_ski / coach1")
    st.stop()

user = st.session_state['user']
st.sidebar.success(f"Выполнен вход: {user['username']} ({user['role']})")
if st.sidebar.button("Выйти"):
    st.session_state['user'] = None
    st.experimental_rerun()

page = st.sidebar.radio("Раздел", ["Dashboard", "Список спортсменов", "Паспорт спортсмена", "Генерация отчёта"])

def load_athletes_for_user(u):
    if u['role'] == 'leader':
        if mode == "snowflake":
            return query_df(mode, "SELECT * FROM ATHLETES ORDER BY SPORT, LAST_NAME")
        else:
            return query_df(mode, "SELECT * FROM athletes ORDER BY sport, last_name")
    elif u['role'] == 'curator':
        if mode == "snowflake":
            return query_df(mode, "SELECT * FROM ATHLETES WHERE SPORT=%s ORDER BY LAST_NAME", (u['sport'],))
        else:
            return query_df(mode, "SELECT * FROM athletes WHERE sport=? ORDER BY last_name", (u['sport'],))
    else:
        if mode == "snowflake":
            return query_df(mode, "SELECT ID, FIRST_NAME, LAST_NAME, BIRTH_YEAR, SPORT FROM ATHLETES WHERE SPORT=%s ORDER BY LAST_NAME", (u['sport'],))
        else:
            return query_df(mode, "SELECT id, first_name, last_name, birth_year, sport FROM athletes WHERE sport=? ORDER BY last_name", (u['sport'],))

# Dashboard
if page == "Dashboard":
    st.header("Панель управления")
    if user['role'] == 'leader':
        counts = {}
        for s in SPORTS.keys():
            if mode == "snowflake":
                dfc = query_df(mode, "SELECT COUNT(*) AS CNT FROM ATHLETES WHERE SPORT=%s", (s,))
                counts[s] = int(dfc.CNT.iloc[0])
            else:
                dfc = query_df(mode, "SELECT COUNT(*) AS CNT FROM athletes WHERE sport=?", (s,))
                counts[s] = int(dfc.CNT.iloc[0])
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
        st.dataframe(df[['id' if mode=="sqlite" else 'ID','last_name' if mode=="sqlite" else 'LAST_NAME','first_name' if mode=="sqlite" else 'FIRST_NAME','birth_year' if mode=="sqlite" else 'BIRTH_YEAR','region' if mode=="sqlite" else 'REGION']])
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
            # отображаем урезанную информацию
            if mode == "snowflake":
                st.dataframe(df[['ID','LAST_NAME','FIRST_NAME','BIRTH_YEAR']])
            else:
                st.dataframe(df[['id','last_name','first_name','birth_year']])
        else:
            st.dataframe(df)

# Паспорт спортсмена
if page == "Паспорт спортсмена":
    st.header("Паспорт спортсмена")
    athlete_id = st.number_input("Введите ID спортсмена", min_value=1, step=1)
    if st.button("Загрузить паспорт"):
        try:
            if mode == "snowflake":
                df = query_df(mode, "SELECT * FROM ATHLETES WHERE ID=%s", (athlete_id,))
            else:
                df = query_df(mode, "SELECT * FROM athletes WHERE id=?", (athlete_id,))
            if df.shape[0] == 0:
                st.error("Спортсмен не найден")
            else:
                a = df.iloc[0].to_dict()
                # normalize keys for mode
                sport_key = a.get('SPORT') or a.get('sport')
                if user['role'] == 'leader' or (user['role'] == 'curator' and user['sport'] == sport_key):
                    st.subheader(f"{a.get('LAST_NAME') or a.get('last_name')} {a.get('FIRST_NAME') or a.get('first_name')}, {int(a.get('BIRTH_YEAR') or a.get('birth_year') or 0)}")
                    st.write("Регион:", a.get('REGION') or a.get('region',''))
                    st.write("Вид спорта:", SPORTS.get(sport_key, sport_key))
                    with st.form("edit_form"):
                        best = st.text_input("Лучший результат", value=a.get('BEST_RESULT') or a.get('best_result',''))
                        vo2 = st.text_input("МПК", value=str(a.get('VO2MAX') or a.get('vo2max','')))
                        maxs = st.text_input("Макс сила", value=str(a.get('MAX_STRENGTH') or a.get('max_strength','')))
                        lm = st.text_input("Безжировая масса", value=str(a.get('LEAN_MASS') or a.get('lean_mass','')))
                        pano = st.text_input("ПАНО", value=str(a.get('PANO') or a.get('pano','')))
                        hr_rest = st.text_input("ЧСС в покое", value=str(a.get('HR_REST') or a.get('hr_rest','')))
                        hr_max = st.text_input("Макс ЧСС", value=str(a.get('HR_MAX') or a.get('hr_max','')))
                        sv = st.text_input("Ударный объём", value=str(a.get('STROKE_VOLUME') or a.get('stroke_volume','')))
                        submitted = st.form_submit_button("Сохранить")
                        if submitted:
                            try:
                                if mode == "snowflake":
                                    execute(mode,
                                            "UPDATE ATHLETES SET BEST_RESULT=%s, VO2MAX=%s, MAX_STRENGTH=%s, LEAN_MASS=%s, PANO=%s, HR_REST=%s, HR_MAX=%s, STROKE_VOLUME=%s WHERE ID=%s",
                                            (best, float(vo2 or 0), float(maxs or 0), float(lm or 0), float(pano or 0),
                                             int(hr_rest or 0), int(hr_max or 0), float(sv or 0), athlete_id))
                                else:
                                    execute(mode,
                                            "UPDATE athletes SET best_result=?, vo2max=?, max_strength=?, lean_mass=?, pano=?, hr_rest=?, hr_max=?, stroke_volume=? WHERE id=?",
                                            (best, float(vo2 or 0), float(maxs or 0), float(lm or 0), float(pano or 0),
                                             int(hr_rest or 0), int(hr_max or 0), float(sv or 0), athlete_id))
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
                if mode == "snowflake":
                    df = query_df(mode, "SELECT * FROM ATHLETES WHERE SPORT=%s ORDER BY LAST_NAME", (sport_choice,))
                else:
                    df = query_df(mode, "SELECT * FROM athletes WHERE sport=? ORDER BY last_name", (sport_choice,))
                pdf_bytes = generate_pdf_bytes(sport_choice, df)
                st.success("Отчёт сгенерирован")
                st.download_button("Скачать PDF", data=pdf_bytes, file_name=f"report_{sport_choice}.pdf", mime="application/pdf")
            except Exception as e:
                st.error("Ошибка генерации отчёта: " + str(e))
