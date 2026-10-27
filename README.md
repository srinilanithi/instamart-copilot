# instamart-copilot
# Instamart Customer Intelligence & Support Copilot

An end-to-end Data Science and AI Engineering capstone pipeline for e-commerce customer intelligence, churn prediction, automated ticket triage with PII privacy guardrails, and autonomous policy retrieval & refund handling.

---

## 📌 Architecture & Features

This project integrates four main components into a single automated pipeline:

1. **Data Engineering & Relational DB (`generate_data.py`, `clean_data.py`)**
   - Ingests raw customer, product, and transaction data.
   - Cleans missing values, standardizes schemas, and persists structured tables into a local SQLite database (`data/instamart.db`).

2. **Customer Segmentation & Churn Modeling (`segmentation_lapse.py`)**
   - Performs **Recency, Frequency, Monetary (RFM)** feature engineering.
   - Uses **K-Means Clustering** to segment customers into actionable behavioural cohorts.
   - Trains a **Random Forest Classifier** to predict customer lapse/churn risk.

3. **Support Ticket Triage & PII Masking (`support_triage.py`)**
   - Implements regex-based **PII privacy guardrails** to mask sensitive personal data (emails, phone numbers) before processing.
   - Categorizes customer support tickets and assigns urgency levels using LLM structured outputs with built-in retry logic.

4. **Policy RAG & Autonomous ReAct Refund Agent (`policy_agent.py`)**
   - **Policy RAG:** Answers customer policy questions based on context-grounding documents.
   - **ReAct Refund Agent:** Executes multi-step tool calls (checking database order status, verifying policy rules, updating customer ledgers) to handle refund requests autonomously.

---

## 📁 Repository Structure

```text
instamart-copilot/
├── data/
│   ├── raw/                 # Raw datasets & policy documents
│   └── instamart.db         # Processed SQLite database
├── outputs/                 # Execution results, models, & output CSVs/JSONLs
├── sql/                     # Analytical SQL scripts
├── generate_data.py         # Synthetic data generator script
├── clean_data.py            # Data cleaning & SQLite ingestion pipeline
├── segmentation_lapse.py    # RFM clustering & Random Forest lapse model
├── support_triage.py        # PII masking & support triage pipeline
├── policy_agent.py          # Policy RAG & ReAct refund agent implementation
├── run_pipeline.py          # Master orchestrator script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation