import os
import json
import csv
import pandas as pd
import sqlite3
from mock_llm import chat

os.makedirs("outputs", exist_ok=True)
os.makedirs("state", exist_ok=True)

print("Loading policy documents...")
POL_DIR = "data/policies"
policies = {}
if os.path.exists(POL_DIR):
    for fn in os.listdir(POL_DIR):
        if fn.endswith(".txt"):
            with open(os.path.join(POL_DIR, fn), "r", encoding="utf-8") as f:
                policies[fn] = f.read()

# 1. Policy RAG Evaluation
eval_path = "data/raw/policy_eval.csv"
if os.path.exists(eval_path):
    qa_df = pd.read_csv(eval_path)
    rag_results = []
    for _, row in qa_df.iterrows():
        question = row['question']
        combined_context = " ".join([f"[{k}] {v}" for k, v in policies.items()])
        messages = [
            {"role": "system", "content": "POLICY ASSISTANT: Answer the question based on the provided context."},
            {"role": "user", "content": f"CONTEXT: {combined_context}\nQUESTION: {question}"}
        ]
        res = chat(messages)
        answer = res.choices[0].message.content
        rag_results.append({"question": question, "answer": answer})
    print("Policy RAG evaluation complete.")

# 2. Refund Agent Execution (ReAct loop simulation)
scenarios_path = "data/raw/agent_scenarios.csv"
if os.path.exists(scenarios_path) and os.path.exists("data/instamart.db"):
    scenarios_df = pd.read_csv(scenarios_path)
    conn = sqlite3.connect("data/instamart.db")

    agent_runs = []
    refund_ledger = []

    for _, sc in scenarios_df.iterrows():
        scenario_id = sc['scenario_id']
        cid = sc['customer_id']
        msg = sc['message']
        
        messages = [
            {"role": "system", "content": "AGENT: Process refund requests using available tools."},
            {"role": "user", "content": msg}
        ]
        
        tool_calls_log = []
        final_response = ""
        
        for step in range(6):
            res = chat(messages, tools=["get_order", "search_policy", "calculate_refund", "issue_refund", "escalate_to_human"])
            choice = res.choices[0]
            
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_name = tc.name
                    args = tc.arguments
                    tool_calls_log.append({"tool": tool_name, "arguments": args})
                    
                    tool_result = {}
                    if tool_name == "get_order":
                        order_id = args.get("order_id")
                        query = f"SELECT * FROM orders WHERE order_id = '{order_id}'"
                        order_row = pd.read_sql(query, conn)
                        if order_row.empty:
                            tool_result = {"error": "no such order"}
                        else:
                            tool_result = order_row.iloc[0].to_dict()
                    elif tool_name == "search_policy":
                        tool_result = {"policy_snippet": policies.get("refund_policy.txt", "")}
                    elif tool_name == "calculate_refund":
                        amt = float(args.get("order_amount", 0))
                        deliv = int(args.get("delivery_minutes", 0))
                        refund_amt = amt if deliv > 45 else 0.0
                        tool_result = {"refund_inr": refund_amt}
                    elif tool_name == "issue_refund":
                        refund_amt = float(args.get("refund_inr", 0))
                        tool_result = {"status": "success", "refund_inr": refund_amt}
                        refund_ledger.append({"scenario_id": scenario_id, "customer_id": cid, "amount": refund_amt})
                    elif tool_name == "escalate_to_human":
                        tool_result = {"status": "escalated"}
                    
                    messages.append({"role": "assistant", "tool_calls": [tc]})
                    messages.append({"role": "tool", "name": tool_name, "content": json.dumps(tool_result)})
            else:
                final_response = choice.message.content
                break

        agent_runs.append({
            "scenario_id": scenario_id,
            "customer_id": cid,
            "message": msg,
            "tool_calls": tool_calls_log,
            "final_output": final_response
        })

    conn.close()

    with open("outputs/agent_runs.jsonl", "w", encoding="utf-8") as f:
        for run in agent_runs:
            f.write(json.dumps(run) + "\n")

    with open("outputs/refund_ledger.jsonl", "w", encoding="utf-8") as f:
        for entry in refund_ledger:
            f.write(json.dumps(entry) + "\n")

    print("Part 4 Refund Agent completed successfully! Saved logs to outputs/")