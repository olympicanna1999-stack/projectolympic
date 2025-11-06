# Цифровой реестр / Паспорт спортсмена (Streamlit + Snowflake)

Коротко: прототип веб‑приложения на Streamlit для демонстрации цифрового реестра олимпийского резерва (спортсмены 14–18 лет), интеграция со Snowflake, разграничение прав (leader, curator, coach), генерация PDF‑отчётов.

Содержимое репозитория:
- app.py — основной код Streamlit
- requirements.txt — зависимости
- Dockerfile — (опционально)

Тестовые учётные записи (demo):
- leader / leaderpass
- curator_ski / curpass1
- curator_biathlon / curpass2
- curator_rowing / curpass3
- coach_ski / coach1
- coach_biathlon / coach2
- coach_rowing / coach3

Развёртывание в Streamlit Cloud (быстрое):
1. Создайте новый репозиторий на GitHub и загрузите файлы из этого проекта.
2. В Streamlit Cloud выберите "New app" → подключите репозиторий → укажите ветку и `app.py`.
3. В Settings → Secrets приложения добавьте секрет `snowflake` (см. .streamlit/secrets.toml.example).
   Пример (в Streamlit Secrets UI задаётся как ключ `snowflake` с JSON полями):
   {
     "user":"MY_USER",
     "password":"MY_PASSWORD",
     "account":"xy12345.eu-central-1",
     "warehouse":"COMPUTE_WH",
     "database":"SPORTS_DB",
     "schema":"PUBLIC",
     "role":"ACCOUNTADMIN"
   }
4. Нажмите Deploy. При первом старте приложение создаст таблицы в Snowflake и заполнит mock‑данными.

Локальный запуск без Snowflake (альтернатива для теста):
- Если не хотите настраивать Snowflake, можно запустить локально, заменив get_snowflake_connection / SQL-функции на SQLite. В случае запроса — подготовлю локальную версию.

Безопасность:
- В прототипе пароли хранятся в БД в открытом виде только для демонстрации. Для продакшена требуется: хеширование паролей, HTTPS, аудит доступа, шифрование PII, юридическая поддержка обработки персональных данных.

Если нужно — подготовлю автоматизированный GitHub Actions workflow или рабочий скрипт для быстрого создания репозитория.
