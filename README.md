# ⚖️ LLMS-DEBATE  

Welcome to **LLMS-DEBATE** — a flexible framework that lets multiple **Large Language Models (LLMs) debate** a given topic and evaluates them using an **AI judge**.  

> ✨ **Key Features:**  
> ✅ Multi-LLM debate simulation  
> ✅ AI-powered judgment & evaluation  
> ✅ Automated post-processing & analysis  

---

## 🚀 **Quick Overview**  

### 📁 **Data & Files Structure**  

📌 **Debate Topics** (`data/debate_topics.txt`)  

> You can store your debate prompts here, separated by blank lines:  
>
> ```
> Ancient Athens or Sparta: Which was the better place to live?  
> Are "debt-for-nature swaps" a good strategy for rainforest conservation?  
> Are apology videos effective?  
> ```

📌 **Experiment Scripts** (`experiments/`)  
🔹 `debate_experiment.py` → Runs LLM debates  
🔹 `postprocess_records.py` → Cleans & organizes debate logs  
🔹 `evaluation_experiment.py` → Judges debates using AI  

📌 **Data Logs**  
📂 `records/` → Stores raw debate records (`debates_*.json`)  
📂 `results/` → Saves evaluation results (`analysis_final.json`)  

---

## 🛠 **Setup & Installation**  

### 1️⃣ **Install Dependencies**  

Check `requirements.txt` for required Python packages.  

### 2️⃣ **Set Up Environment**  

Create or edit `.env` in the root folder:  

```ini
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
MISTRAL_API_KEY=your_mistral_key
GENAI_API_KEY=your_genai_key
```

> **Note:** Only fill in the keys for the models you want to use. We also support the local models by Ollama!

---

## 🧠 **Supported Models**  

This framework supports multiple LLMs out of the box:  

| Model Name             | Provider       |
| ---------------------- | -------------- |
| `gpt-4o`               | OpenAI         |
| `claude-3.5-haiku`     | Anthropic      |
| `mistral-small-latest` | Mistral        |
| `llama-3.2-3b`         | Local (Ollama) |
| `gemini-2.0-flash`     | Google GenAI   |

Ensure you have the correct API keys and installed libraries for each.

---

## 🎯 **Usage**  

### 💬 1) Run Debates  

Run an experiment with LLMs debating topics from `data/debate_topics.txt`:  

```bash
python experiments/debate_experiment.py   --start 0   --end 4   --rounds 3   --models gpt-4o claude-3.5-haiku   --max_tokens 200
```

📌 **Breakdown:**  

- Picks topics **[0–4]** from `data/debate_topics.txt`  
- Runs **3 rebuttal rounds**  
- Uses **GPT-4o vs Claude-3.5-haiku**  (you could use more than 2 models, they will debate in a round-robin manner)
- Saves logs in `records/`

---

### 🧹 2) Postprocess Records  

Clean and organize debate logs:  

```bash
python experiments/postprocess_records.py --n_models 2
```

📌 **Outputs:**  

- Process all the files inside the `records/` folder, distinguish completed vs. incomplete debates  
- Summary report in `records/postprocess_records_{timestamp}/`

---

### 🔄 3) Fill Missing Debates  

If the summary shows missing debates, complete them:  

```bash
python experiments/debate_experiment.py   --complete_from records/postprocess_records_{timestamp}/postprocess_summarization_{timestamp}.json
```

📌 **Only missing pairs will be debated again!**

---

### ✅ 4) Evaluate Debates  

Use AI to judge the debates and determine winners:  

```bash
python experiments/evaluation_experiment.py   --postprocess_folder records/postprocess_records_{timestamp}   --judge_model gpt-4o   --judge_max_tokens 100
```

📌 **Saves results in `results/evaluation_{timestamp}/`.**  

---

## 📊 **Final Results**  

🎯 Check the final debate rankings and insights in `results/analysis_final.json`.  

---

## 🤝 **Contributing**  

Contributions are welcome! Open a PR or issue to suggest improvements.  

---

## ⚖️ **License**  

MIT License. Feel free to modify and use!  

---

## 🌟 **Enjoy the debating of AI!**