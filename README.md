# LiquidityAI

Treasury Management System — предиктивное управление ликвидностью для финтех-компаний.

## Быстрый старт

```bash
pip install -r requirements.txt
streamlit run app.py
```

Открыть: **http://localhost:8501**

---

## Возможности

### ML-прогнозирование
- Прогноз cash flow на 1–7 дней (Random Forest + Gradient Boosting)
- Квантильные сценарии: оптимистичный (q90), ожидаемый (q50), пессимистичный (q10)
- Вероятность дефицита P(deficit) для каждого счёта
- Учёт задержек клиринга: SEPA (1д), SWIFT (3д), Card (5д), Local (0д)

### Алерты
| Тип | Описание |
|---|---|
| CURRENT_DEFICIT | Баланс ниже минимума прямо сейчас |
| FORECAST_DEFICIT | Прогнозируемый дефицит через N дней |
| CLEARING_RISK | Высокая зависимость от ожидаемых поступлений |
| EXCESS_IDLE | Избыточные средства не приносят доход |

Уровни: **CRITICAL / HIGH / MEDIUM / LOW**

### Оптимизация ликвидности
- Автоматический поиск профицитных и дефицитных счетов
- Рекомендации переводов с учётом стоимости (bps), времени и FX-конвертации
- Расчёт упущенного дохода на idle-капитале (4.5% годовых)

### What-If симулятор
| Сценарий | Параметр |
|---|---|
| Отключение SWIFT | Все SWIFT-счета теряют 65% баланса |
| Задержка SEPA/Card | +1…+5 дней к клирингу |
| Пик объёмов | +0…+200% к исходящим |
| FX-шок | EUR/GBP ±20% к USD |
| Banking outage | Выбранные счета блокируются |

---

## Структура

```
├── app.py              — Streamlit-приложение (главный файл)
├── core/
│   ├── data.py         — генератор синтетических данных (12 мес., 6 счетов)
│   ├── ml.py           — ML-модели прогнозирования
│   └── engine.py       — движок рисков, оптимизатор, what-if
├── api/
│   └── main.py         — FastAPI REST API
├── models/             — вспомогательные ML-модули
├── data/               — генераторы данных
└── requirements.txt
```

---

## REST API

```bash
python run_api.py
# Документация: http://localhost:8000/docs
```

| Метод | Эндпоинт | Описание |
|---|---|---|
| GET | /api/accounts | Текущие балансы |
| GET | /api/forecast?days=3 | Прогноз cash flow |
| GET | /api/alerts | Активные алерты |
| GET | /api/recommendations | Рекомендации переводов |
| POST | /api/stress-test | Запуск стресс-теста |

---

## Демо-данные

6 ностро-счетов в 3 валютах:

| Счёт | Валюта | Система | Мин. баланс | Целевой |
|---|---|---|---|---|
| Citibank USD | USD | SWIFT | $500K | $2M |
| Stripe USD | USD | CARD | $300K | $1.5M |
| Deutsche Bank EUR | EUR | SEPA | €400K | €1.8M |
| Adyen EUR | EUR | CARD | €250K | €1.2M |
| Barclays GBP | GBP | LOCAL | £200K | £800K |
| HSBC GBP | GBP | SWIFT | £150K | £600K |

---

## Деплой на Streamlit Cloud

1. Форкни репозиторий
2. Зайди на [share.streamlit.io](https://share.streamlit.io)
3. New app → выбери этот репо → `app.py`
4. Deploy

---

FinTech Hackathon 2026 | LiquidityAI Team
