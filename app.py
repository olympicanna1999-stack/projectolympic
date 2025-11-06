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

    birth
