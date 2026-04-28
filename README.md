# ⚙️ NexusPart | Local RAG System for Industrial Supply Chains


**NexusPart** is a smart tool for people who work with industrial parts. It helps you find the right electrical components (like fuses) from a big warehouse list. It uses **AI (Llama 3)** to make sure the parts are safe to use.

---

## 🚀 What can it do?

* **Smart Search:** You don't need exact part numbers. You can type "2A Fuse for high voltage," and it will understand what you need.
* **AI Advice:** It talks like a friendly expert. It tells you *why* a part is a good match and warns you if there is a safety risk.
* **Modern Design:** The dashboard is dark-themed and very easy to read on a computer screen.
* **Fast Results:** It finds the best matches in less than a second using a fast search engine (FAISS).

---

## 🛠️ How it was built

* **Python:** The main language used for coding.
* **Streamlit:** Used to create the website-like dashboard.
* **Llama 3:** The "brain" of the AI that gives the advice.
* **FAISS:** A special tool to search through thousands of parts quickly.
* **Sentence-Transformers:** Helps the AI understand the meaning of your words.

---

## 📦 How to set it up

1.  **Clone the code:** Download this project to your computer.
2.  **Install Libraries:** Open your terminal and type:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Start the AI:** Make sure you have **Ollama** installed and running with Llama 3.
4.  **Run the App:** Type this in your terminal:
    ```bash
    streamlit run app.py
    ```

---

## 📊 Files in this project

* `app.py` – The main code for the dashboard and UI.
* `processed_parts_for_rag.csv` – The cleaned list of all industrial parts.
* `parts_vector_db.index` – The database file used for fast searching.
* `.gitignore` – Tells GitHub to ignore hidden or heavy folders like `.venv`.
* `README.md` – This description file.

---

## 🛡️ Safety Warning
Always double-check the AI's advice. Before buying or using a part, make sure it meets your circuit's safety rules and official datasheets.

---
**Created by [Laiva Noor]** *Making industrial work easier with AI.*