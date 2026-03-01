# 🤖 AI-Driven App Development — OIM 3641

**Babson College | Spring 2026 | Prof. Matthew Macarty**

Classwork repository for OIM 3641: AI-Driven App Development. This repo contains in-class activities, assignments, and experiments built throughout the semester using Python, AI APIs, and modern development tools.

---

## 👤 About Me

I'm a senior at Babson College studying business with a concentration in AI and entrepreneurship. I founded **PING!** — a smart NFC ring for professional networking — and lead AI workshops at The Generator, Babson's AI innovation lab. I'm passionate about building real products at the intersection of AI and business.

---

## 🛠️ Skills & Tools

**Languages**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![Markdown](https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white)

**Libraries & Frameworks**

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**Tools & Platforms**

![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![PyCharm](https://img.shields.io/badge/PyCharm-000000?style=for-the-badge&logo=pycharm&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-6B21A8?style=for-the-badge&logoColor=white)

---

## 📁 Directory Structure

```
ai-app-dev-class1/
│
├── data/                        # Datasets used in assignments
│
├── 02-python_concepts.ipynb     # Python concepts review (Class 2)
├── 06-inclass-activity.py       # Stock analyzer with comparison tab (Class 6)
├── llm_test.py                  # LLM API call testing
├── main.py                      # General scratchpad / entry point
├── stock.py                     # Stock class — Assignment 1
│
└── README.md                    # You are here
```

---

## ⚙️ Install Instructions

**Clone the repo**
```bash
git clone https://github.com/TheRealPress1/ai-app-dev-class1.git
cd ai-app-dev-class1
```

**Create and activate a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

**Install dependencies**
```bash
pip install yfinance pandas numpy matplotlib seaborn streamlit plotly
```

**Run the stock analyzer app**
```bash
streamlit run 06-inclass-activity.py
```

**Use the Stock class**
```python
from stock import Stock

aapl = Stock("AAPL")
print(aapl.data.head())
aapl.plot_performance()
aapl.add_technical_indicators()
```

---

## 🔗 Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gaspard-seuge)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/TheRealPress1)
