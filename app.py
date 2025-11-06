# app.py
import streamlit as st
import pandas as pd
import io
import random
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
# safe_snowflake_check (вставьте в top app.py перед import snowflake.connector usage)
import streamlit as st

USE_SNOWFLAKE = False
if "snowflake" in st.secrets:
    try:
        import snowflake.connector  # попытка импортировать
        # попытка открыть тестовое соединение (без парсинга ошибок подробно)
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
        conn.close()
        USE_SNOWFLAKE = True
    except Exception as e:
        # не падаем — включаем режим sqlite и показываем уведомление в UI
        USE_SNOWFLAKE = False
        st.warning("Snowflake недоступен (ошибка соединения или неправильные креды). Переходим в локальный режим SQLite.")
else:
    # секретов нет — переключаемся на SQLite
    USE_SNOWFLAKE = False
    # НЕ используйте st.error/st.stop здесь — иначе импорт завершится с ошибкой до UI

# ---------- Snowflake helpers ----------
def get_snowflake_connection():
    # Ожидается st.secrets["snowflake"] с keys user,password,account,warehouse,database,schema,role
    if "snowflake" not in st.secrets:
        raise RuntimeError("Snowflake credentials not found in st.secrets. Проверьте настройки секретов.")
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
    return conn

def ensure_schema_and_tables():
    conn = get_snowflake_connection()
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

def query_df(sql, params=None):
    conn = get_snowflake_connection()
    try:
        df = pd.read_sql(sql, conn, params=params)
        return df
    finally:
        conn.close()

def execute(sql, params=None):
    conn = get_snowflake_connection()
    cs = conn.cursor()
    try:
        cs.execute(sql, params or {})
        conn.commit()
    finally:
        cs.close()
        conn.close()

# ---------- Seed mock data (если пусто) ----------
def seed_if_empty():
    ensure_schema_and_tables()
    users_df = query_df("SELECT COUNT(*) AS CNT FROM USERS")
    if int(users_df.CNT.iloc[0]) > 0:
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
        execute("INSERT INTO USERS (USERNAME, ROLE, SPORT, PASSWORD) VALUES (%s,%s,%s,%s)",
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
            created_at = datetime.utcnow()
            execute(
                "INSERT INTO ATHLETES (FIRST_NAME,LAST_NAME,BIRTH_YEAR,SEX,SPORT,REGION,BEST_RESULT,"
                "VO2MAX,MAX_STRENGTH,LEAN_MASS,PANO,HR_REST,HR_MAX,STROKE_VOLUME,CREATED_AT) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (first,last,by,sex,sport_key,region,best,vo2,ms,lm,pano,hr_rest,hr_max,sv,created_at)
            )

# ---------- Authentication (demo) ----------
def authenticate(username, password):
    df = query_df("SELECT * FROM USERS WHERE USERNAME=%s AND PASSWORD=%s", (username, password))
    if df.shape[0] == 0:
        return None
    row = df.iloc[0]
    return {'id': int(row.ID), 'username': row.USERNAME, 'role': row.ROLE, 'sport': row.SPORT}

# ---------- PDF report ----------
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
        line = f"{a['LAST_NAME']} {a['FIRST_NAME']}, {int(a['BIRTH_YEAR'])}, Регион: {a.get('REGION','')}"
        p.drawString(40, y, line)
        y -= 14
        sub = (f"МПК: {a.get('VO2MAX','')}  Сила: {a.get('MAX_STRENGTH','')}  БМ: {a.get('LEAN_MASS','')}  "
               f"ПАНО: {a.get('PANO','')}  ЧСС пок: {int(a.get('HR_REST',0))}  ЧСС макс: {int(a.get('HR_MAX',0))}  УО: {a.get('STROKE_VOLUME','')}")
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
st.set_page_config(page_title="Реестр олимпийского резерва", layout="wide")
st.title("Цифровой реестр / Паспорт спортсмена — Streamlit + Snowflake (прототип)")

# Инициализация
with st.spinner("Проверка и инициализация базы..."):
    try:
        seed_if_empty()
    except Exception as e:
        st.error("Ошибка инициализации базы данных: " + str(e))
        st.stop()

# Сессия пользователя
if 'user' not in st.session_state:
    st.session_state['user'] = None

# Вход
if st.session_state['user'] is None:
    st.sidebar.header("Вход")
    username = st.sidebar.text_input("Логин")
    password = st.sidebar.text_input("Пароль", type="password")
    if st.sidebar.button("Войти"):
        user = authenticate(username.strip(), password.strip())
        if user:
            st.session_state['user'] = user
            st.experimental_rerun()
        else:
            st.sidebar.error("Неверные учетные данные")
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
        df = query_df("SELECT * FROM ATHLETES ORDER BY SPORT, LAST_NAME")
    elif u['role'] == 'curator':
        df = query_df("SELECT * FROM ATHLETES WHERE SPORT=%s ORDER BY LAST_NAME", (u['sport'],))
    else:
        df = query_df("SELECT ID, FIRST_NAME, LAST_NAME, BIRTH_YEAR, SPORT FROM ATHLETES WHERE SPORT=%s ORDER BY LAST_NAME", (u['sport'],))
    return df

if page == "Dashboard":
    st.header("Панель управления")
    if user['role'] == 'leader':
        counts = {}
        for s in SPORTS.keys():
            dfc = query_df("SELECT COUNT(*) AS CNT FROM ATHLETES WHERE SPORT=%s", (s,))
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
        st.dataframe(df[['ID','LAST_NAME','FIRST_NAME','BIRTH_YEAR','REGION']])
    else:
        st.write(f"Тренер вида: {SPORTS.get(user['sport'])}")
        st.info("Тренерам прямой доступ к персональным данным закрыт. Для получения отчёта обратитесь к куратору или руководителю.")

if page == "Список спортсменов":
    st.header("Список спортсменов")
    df = load_athletes_for_user(user)
    if df.empty:
        st.info("Нет данных для отображения в рамках вашей роли.")
    else:
        if user['role'] == 'coach':
            st.dataframe(df[['ID','LAST_NAME','FIRST_NAME','BIRTH_YEAR']])
        else:
            st.dataframe(df)

if page == "Паспорт спортсмена":
    st.header("Паспорт спортсмена")
    athlete_id = st.number_input("Введите ID спортсмена", min_value=1, step=1)
    if st.button("Загрузить паспорт"):
        try:
            df = query_df("SELECT * FROM ATHLETES WHERE ID=%s", (athlete_id,))
            if df.shape[0] == 0:
                st.error("Спортсмен не найден")
            else:
                a = df.iloc[0].to_dict()
                if user['role'] == 'leader' or (user['role'] == 'curator' and user['sport'] == a['SPORT']):
                    st.subheader(f"{a['LAST_NAME']} {a['FIRST_NAME']}, {int(a['BIRTH_YEAR'])}")
                    st.write("Регион:", a.get('REGION',''))
                    st.write("Вид спорта:", SPORTS.get(a['SPORT'], a['SPORT']))
                    with st.form("edit_form"):
                        best = st.text_input("Лучший результат", value=a.get('BEST_RESULT',''))
                        vo2 = st.text_input("МПК", value=str(a.get('VO2MAX','')))
                        maxs = st.text_input("Макс сила", value=str(a.get('MAX_STRENGTH','')))
                        lm = st.text_input("Безжировая масса", value=str(a.get('LEAN_MASS','')))
                        pano = st.text_input("ПАНО", value=str(a.get('PANO','')))
                        hr_rest = st.text_input("ЧСС в покое", value=str(a.get('HR_REST','')))
                        hr_max = st.text_input("Макс ЧСС", value=str(a.get('HR_MAX','')))
                        sv = st.text_input("Ударный объём", value=str(a.get('STROKE_VOLUME','')))
                        submitted = st.form_submit_button("Сохранить")
                        if submitted:
                            execute(
                                "UPDATE ATHLETES SET BEST_RESULT=%s, VO2MAX=%s, MAX_STRENGTH=%s, LEAN_MASS=%s, PANO=%s, HR_REST=%s, HR_MAX=%s, STROKE_VOLUME=%s WHERE ID=%s",
                                (best, float(vo2 or 0), float(maxs or 0), float(lm or 0), float(pano or 0),
                                 int(hr_rest or 0), int(hr_max or 0), float(sv or 0), athlete_id)
                            )
                            st.success("Паспорт обновлён")
                else:
                    st.error("У вас нет права просматривать/редактировать паспорт этого спортсмена")
        except Exception as e:
            st.error("Ошибка: " + str(e))

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
                df = query_df("SELECT * FROM ATHLETES WHERE SPORT=%s ORDER BY LAST_NAME", (sport_choice,))
                pdf_bytes = generate_pdf_bytes(sport_choice, df)
                st.success("Отчёт сгенерирован")
                st.download_button("Скачать PDF", data=pdf_bytes, file_name=f"report_{sport_choice}.pdf", mime="application/pdf")
            except Exception as e:
                st.error("Ошибка генерации отчёта: " + str(e))
