"""
In-class Activity 14 - Hugging Face Sentiment Analysis & Zero-Shot Classification
OIM 3641 | AI-Driven App Development
"""

from transformers import pipeline

"""
STUDENT CHANGE LOG & AI DISCLOSURE:
----------------------------------
1. Did you use an LLM (ChatGPT/Claude/etc.)? No
2. If yes, what was your primary prompt? N/A
----------------------------------
"""

# --- PART 2: Default Sentiment Pipeline ---
print("=" * 60)
print("PART 2: Default Sentiment Pipeline")
print("=" * 60)

model = pipeline("sentiment-analysis")
text = "This is the most straightforward and effective method I have ever learned"
result = model(text)
print(result)

# --- PART 3: Custom Finance Model ---
print("\n" + "=" * 60)
print("PART 3: Custom Finance Model (yiyanghkfinbert-tone)")
print("=" * 60)

financial_classifier = pipeline("text-classification", model="ProsusAI/finbert")
finance_text = "The stock market rally continued, suggesting strong long-term growth."
finance_result = financial_classifier(finance_text)
print(finance_result)

# --- PART 4: Bulk Analysis ---
print("\n" + "=" * 60)
print("PART 4: Bulk Sentence Analysis")
print("=" * 60)

sentences = [
    "The quarterly earnings report was surprisingly weak, causing investor concern.",
    "Despite market volatility, the company announced record profits.",
    "I'm not sure if I should invest in tech stocks this quarter."
]

bulk_results = financial_classifier(sentences)
for sentence, res in zip(sentences, bulk_results):
    print(f"\nText:  {sentence}")
    print(f"Label: {res['label']}  |  Score: {round(res['score'] * 100, 2)}%")

# --- PART 5: Zero-Shot Classification ---
print("\n" + "=" * 60)
print("PART 5: Zero-Shot Classification - Original Labels")
print("=" * 60)

model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
text = ("The latest press release details the company's new policy on remote work, "
        "including guidelines for team communication and hardware allocation for employees worldwide.")

labels = ["Employee Relations", "Financial News", "Product Announcement", "Technical Support"]
result = model(text, labels)

print(f"Input Text: {result['sequence']}")
print("Classification Scores:")
for i, (label, score) in enumerate(zip(result['labels'], result['scores'])):
    print(f"  {i+1}. {label}: {round(score * 100, 2)}%")

# --- PART 5 EXTENDED: Add more labels ---
print("\n" + "=" * 60)
print("PART 5: Zero-Shot Classification - Extended Labels")
print("=" * 60)

extended_labels = [
    "Employee Relations", "Financial News", "Product Announcement", "Technical Support",
    "Sales", "HR Policy", "Legal Compliance"
]
result2 = model(text, extended_labels)

print(f"Input Text: {result2['sequence']}")
print("Classification Scores:")
for i, (label, score) in enumerate(zip(result2['labels'], result2['scores'])):
    print(f"  {i+1}. {label}: {round(score * 100, 2)}%")
