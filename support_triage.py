import os
import re
import json
import pandas as pd
from mock_llm import chat, TransientLLMError

os.makedirs("outputs", exist_ok=True)

print("Loading tickets...")
tickets_df = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

def mask_pii(text):
    if not text:
        return ""
    # Mask email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Mask phone numbers (10 digits)
    text = re.sub(r'\b\d{10}\b', '[PHONE]', text)
    return text

results = []
print("Classifying tickets with privacy masking and error recovery...")

for _, row in tickets_df.iterrows():
    ticket_id = row['ticket_id']
    customer_id = row['customer_id']
    order_id = row['order_id']
    true_cat = row['true_category']
    raw_text = row['text']
    
    # 1. Privacy Shield: Mask PII
    masked_text = mask_pii(raw_text)
    
    # 2. Classification via mock_llm with retry for TransientLLMError
    messages = [
        {"role": "system", "content": "CLASSIFY: Categorize the support ticket into billing, delivery, technical, or fraud."},
        {"role": "user", "content": masked_text}
    ]
    
    predicted_cat = "unknown"
    success = False
    while not success:
        try:
            response = chat(messages)
            content = response.choices[0].message.content
            if content.startswith("{"):
                parsed = json.loads(content)
                predicted_cat = parsed.get("category", "unknown")
            else:
                predicted_cat = content
            success = True
        except TransientLLMError:
            # Retry automatically on server busy
            continue
        except Exception:
            predicted_cat = "error"
            success = True

    results.append({
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "order_id": order_id,
        "masked_text": masked_text,
        "true_category": true_cat,
        "predicted_category": predicted_cat
    })

triage_results_df = pd.DataFrame(results)
triage_results_df.to_csv("outputs/triage_results.csv", index=False)
triage_results_df.head(50).to_csv("outputs/triage_budget_demo.csv", index=False)
print("Part 3 Triage completed successfully! Saved to outputs/triage_results.csv")