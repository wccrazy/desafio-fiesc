# 🔧 Desafio — Manutenção Prescritiva

Sistema inteligente de diagnóstico e prescrição de manutenção de equipamentos industriais, combinando similaridade semântica, RAG sobre manuais técnicos e integração MQTT.

---

## 📁 Estrutura do Projeto

```
desafio-manutencao-prescritiva/
├── context/                    # Edital e especificação do desafio
├── data/
│   ├── banner.csv              # Dataset histórico de ordens de serviço
│   └── manuals/                # Manuais técnicos dos equipamentos (PDF/TXT)
├── artifacts/
│   ├── app.db                  # Banco SQLite
│   ├── similarity.joblib       # Índice de similaridade serializado
│   └── chroma/                 # Base vetorial ChromaDB
├── tests/                      # Testes automatizados (pytest)
├── similarity_engine.py        # Motor de similaridade semântica
├── rag_engine.py               # Motor RAG (retrieval + geração)
├── api.py                      # API REST (FastAPI)
├── mqtt_bridge.py              # Consumidor de alertas MQTT
├── mqtt_simulator.py           # Simulador de alertas MQTT
├── streamlit_app.py            # Dashboard interativo
├── Dockerfile
├── docker-compose.yml
├── mosquitto.conf
├── requirements.txt
└── README.md
```

## 🚀 Início Rápido

### Com Docker Compose

```bash
docker-compose up --build
```

| Serviço    | URL                    |
|------------|------------------------|
| API        | http://localhost:8000  |
| Docs API   | http://localhost:8000/docs |
| Dashboard  | http://localhost:8501  |
| MQTT       | localhost:1883         |

### Localmente

```bash
pip install -r requirements.txt

# API
uvicorn api:app --reload

# Dashboard
streamlit run streamlit_app.py

# Bridge MQTT
python mqtt_bridge.py

# Simulador MQTT
python mqtt_simulator.py
```

## 🧪 Testes

```bash
pytest tests/ -v
```

## 📄 Licença

A definir.
