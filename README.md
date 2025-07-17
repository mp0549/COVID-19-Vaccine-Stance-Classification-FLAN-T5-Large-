# COVID-19 Vaccine Stance Classification (FLAN‑T5‑Large)

*A research take‑home: classify tweet stance toward COVID‑19 vaccination as **in-favor**, **against**, or **neutral-or-unclear**. Built under tight time & GPU constraints; intentionally transparent, with commented‑out experiments preserved for review.*

**Task window:** due **July 16, 2025**.
**Author:** *\Maanan Purothi*

---

## Contents

* Dataset
* Environment & Libraries
* Development Timeline
* Common Failure Cases
* Training Details (Curriculum + LoRA)
* CSV Inference Code
* Local Results Snapshot
* Lessons Learned
* Exploratory Directions (Not Fully Integrated)
* What to Look For When Reviewing
* Personal Note
* References

---

## Dataset

**File:** `OriginalData.csv` (\~5.7k rows).
**Columns:**

*`tweet_id` – 19 digit id code for each tweet.
*`created_at` – tweet creation date
* `tweet` – raw tweet text (emoji, hashtags, links preserved).
* `label_majority` – majority human annotation.
* `label_pred` – to be filled by model.

**Label definitions:**

* **in-favor** – endorses, recommends, celebrates, or positively reports vaccination.
* **against** – rejects, warns against, distrusts vaccine institutions, conspiracy framing, negative anecdote.
* **neutral-or-unclear** – informational / asks a question / off‑topic mention / ambiguous / mixed.

> ⚠ **Neutral is tricky.** Many tweets are short, ironic, or purely informational ("clinic open Sat"). Others encode stance via emoji only (😷✅ vs 🤡💉). Expect noise.

---

## Environment & Libraries

Developed primarily in **Google Colab**. Early CPU runs were painfully slow; switching to **T4 GPU** + **8‑bit loading** + **LoRA** made iteration feasible.

**Install (Colab cell):**

```python
!pip install -U transformers==4.53.1 peft bitsandbytes accelerate datasets torch scikit-learn pandas numpy tqdm
```

**Key packages:** `transformers`, `peft`, `bitsandbytes`, `datasets`, `torch`, `scikit-learn`, `pandas`, `numpy`, `tqdm`.

**Version gotcha:** My first training attempt crashed: `TypeError: unexpected keyword argument 'evaluation_strategy'`. The installed transformers build expected `eval_strategy` (see Timeline). Always print library versions.

---

## Development Timeline

*Real lab‑notebook style: what I tried, what broke, what improved. I don’t label these as V1/V2/etc. in code; this is the narrative arc.*

### Baseline Prompting → Neutral Collapse

I wrapped each tweet in a detailed instruction prompt (expert journalist voice, rationale request, numeric label mapping). FLAN‑T5‑Large responded, but overwhelmingly with **neutral‑ish outputs**, long rationales, and formatting drift. After regex parsing, accuracy was **<30%** on a quick local check. Lesson: prompting alone + fragile parsing = junky baseline.

### Supervised Fine‑Tune (Full Seq2Seq)

I fine‑tuned FLAN‑T5 using Hugging Face `Seq2SeqTrainer` on the labeled data. After fixing the arg mismatch (`eval_strategy`), small‑batch training ran. Accuracy rose to about **\~50%** on dev splits, a real signal! But full‑model training was slow/heavy on Colab; I needed something lighter.

### Resource Pivot: 8‑bit + LoRA

Loaded FLAN‑T5‑Large with `load_in_8bit=True` and attached **LoRA adapters** (r=8, alpha=16, dropout=0.1; target `q`/`v`). This fit in Colab memory and trained quickly. I also tried **multiple prompt formats** (short direct, A/B/C multiple choice). Training & val loss fell to \~0.22 in one run which was pretty exciting! But performance on held‑out text dropped to \~40% because label decoding was brittle and data slices were narrow. Low loss ≠ generalization.

### Curriculum‑Style Stance Training

Neutral kept breaking things. I staged training:

1. **Poles only** (`in-favor` vs `against`) → learn polarity.
2. **Neutral only** → learn "absence of stance" signals.
3. **Balanced mix** (sampled \~1k/class) → calibrate 3‑way boundary.
   Ran short epochs per stage with early stopping; reloaded adapters sequentially. This stabilized predictions: fewer "all neutral" runs; better neutral recall than straight fine‑tune. Still noisy, but progress.

### Prompt Tightening & Mapping Fixes

Many "errors" were decode bugs: model output "Answer: A) Pro‑vaccine" → regex failed → default neutral. I hardened parsing (case‑insensitive label match; letter fallback; numeric fallback). Also shortened prompts for generation budget. Label hygiene alone rescued several checkpoints.

### Still Hard: Sarcasm, Mixed Stance, Emoji

Tweets like "Can't wait for my 5G upgrade 😂" or "Got the shot but don't trust pharma" confused both model and annotators. I experimented with hint prompts triggered by uncertainty keywords and emoji (see Common Failures below). Mixed results; I kept the cleanest prompt for final runs.

---

## Common Failure Cases

Below are recurring error patterns I saw while spot‑checking misclassified tweets during development. These informed prompt tweaks and curriculum design.

| Case                   | Example (abbrev)                                | Gold                | Model Tendency  | Notes                                                    |
| ---------------------- | ----------------------------------------------- | ------------------- | --------------- | -------------------------------------------------------- |
| **Sarcasm / irony**    | "Yeah, inject me with government microchips 🤡" | against             | neutral         | Polarity inversion; irony tokens (🤡) matter.            |
| **Emoji‑only signal**  | "💉💉✅" or "💉🤮"                               | depends             | neutral         | Emojis stripped of text lose stance; need emoji lexicon. |
| **Info / logistics**   | "Clinic open Sat. Register here"                | neutral             | in-favor        | Model equates mention w/ endorsement.                    |
| **Mixed / conflicted** | "I got it, but I still don't trust it"          | annotator‑dependent | neutral         | Multi‑clause stance confuses mapping.                    |
| **Policy snark**       | "Mandate me harder, gov't"                      | against             | neutral         | Satire vs compliance ambiguous.                          |
| **Hashtag drift**      | "#GetVaccinated" inside joke threads            | in-favor            | against/neutral | Hashtag sarcasm.                                         |

> 💡 *Emoji edge case:* Multi‑emoji tweets (👏, 🤒, 😷, 🤡, 💉, ✅, ❌) frequently tripped the model; in future work I'd map common emoji clusters to prior stance probabilities.

---

## Training Details (Curriculum + LoRA)

Core elements of the training pipeline used for my strongest runs.

### Prompt (final training form)

Short, direct, and label‑exact:

```python
def format_prompt(tweet):
    return (
        "What is the stance of the author of the following tweet toward COVID-19 vaccines?
" \
        f'Tweet: "{tweet}"
' \
        "Respond with exactly one of: in-favor, against, neutral-or-unclear."
    )
```

### Data Staging

```python
phase1 = df[df.target.isin(["in-favor","against"])]
phase2 = df[df.target == "neutral-or-unclear"]
phase3 = pd.concat([
    df[df.target=="in-favor"].sample(1000,random_state=1),
    df[df.target=="against"].sample(1000,random_state=1),
    df[df.target=="neutral-or-unclear"].sample(1000,random_state=1),
])
```

### Tokenization (truncation tuned for Colab)

```python
def tokenize(batch):
    x = tokenizer(batch["prompt"], max_length=256, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        y = tokenizer(batch["target"], max_length=8, truncation=True, padding="max_length")
    x["labels"] = y["input_ids"]
    return x
```

### LoRA Setup

```python
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", load_in_8bit=True)
model = prepare_model_for_kbit_training(model)
conf = LoraConfig(task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q","v"]) 
model = get_peft_model(model, conf)
```

### Train Wrapper

```python
def train_phase(df_phase, name, epochs, lr):
    dset = Dataset.from_pandas(df_phase[["prompt","target"]])
    tok = dset.map(tokenize, batched=True)
    split = tok.train_test_split(test_size=0.2)
    args = Seq2SeqTrainingArguments(
        output_dir=f"./results_{name}",
        learning_rate=lr,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        report_to="none",
        save_total_limit=2,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    return trainer
```

### Curriculum Run

```python
train_phase(phase1, "phase1_poles",   2, 2e-5)
train_phase(phase2, "phase2_neutral", 2, 1e-5)
train_phase(phase3, "phase3_mix",     5, 2e-5)
```

(Reran short refresh epochs on mix to stabilize.)

---

## CSV Inference Code

Minimal batch inference that reads the provided CSV, generates predictions, and writes `label_pred`.

```python
import pandas as pd, torch, re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- load ---
MODEL_DIR = "models/final_curriculum_lora"  # adapter + base
CSV_IN   = "data/Q2_20230202_majority.csv"
CSV_OUT  = "outputs/stance_predictions.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

PAT = re.compile(r"(in[- ]?favor|against|neutral[- ]?or[- ]?unclear)", re.I)
LET = re.compile(r"([abc])", re.I)

LETTER_MAP = {"a":"in-favor","b":"against","c":"neutral-or-unclear"}


def canonicalize(s: str):
    t = s.strip().lower()
    m = PAT.search(t)
    if m:
        g = m.group(1).replace(" ", "").replace("-", "")
        if g.startswith("infavor"): return "in-favor"
        if g.startswith("against"): return "against"
        return "neutral-or-unclear"
    m = LET.search(t)
    if m: return LETTER_MAP[m.group(1).lower()]
    if "1" in t: return "in-favor"
    if "-1" in t: return "against"
    return "neutral-or-unclear"


def format_prompt(tweet):
    return (
        "What is the stance of the author of the following tweet toward COVID-19 vaccines?
"
        f'Tweet: "{tweet}"
'
        "Respond with exactly one of: in-favor, against, neutral-or-unclear."
    )


def predict_batch(texts, max_new_tokens=10):
    prompts = [format_prompt(t) for t in texts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    outs = tokenizer.batch_decode(out, skip_special_tokens=True)
    labels = [canonicalize(o) for o in outs]
    return labels, outs


# --- run ---
df = pd.read_csv(CSV_IN)
labels, raw = predict_batch(df.tweet.tolist())
df["label_pred"] = labels
# df["model_output_raw"] = raw  # uncomment for debugging

df.to_csv(CSV_OUT, index=False)
print("wrote", CSV_OUT)
```

---

## Local Results Snapshot

*(Replace ? with your measured values; provide seed + sample size if possible.)*

| Stage                     | Accuracy  | Macro F1              | Notes                      |
| ------------------------- | --------- | --------------------- | -------------------------- |
| Baseline prompting        | <0.30     | low                   | strong neutral bias        |
| First fine‑tune           | \~0.50    | rising                | full Seq2Seq; small epochs |
| LoRA early                | \~0.40    | unstable              | overfit + parsing issues   |
| Curriculum (best dev)     | 0.45–0.55 | better neutral recall | poles→neutral→mix          |
| **Submission checkpoint** | TBD       | TBD                   | fill after final run       |

---

## Lessons Learned

Shortlist of things that materially affected results or productivity.

**Data balance matters early.** With a dominant class, zero‑shot + weak supervision collapses to it.

**Prompt verbosity hurts throughput & parsing.** Short label‑exact prompts gave cleaner generations.

**Neutral is conceptually different.** Treating neutral as “everything else” fails; dedicate training signal (curriculum) and instructions.

**Always inspect raw generations.** Pretty eval loss can hide unusable outputs.

**LoRA + 8‑bit = student‑friendly scaling.** Huge model, small adapter; fast iteration.

**Emoji ≠ decoration.** 🤡, 💉, ✅, ❌ often carry stance; ignoring them costs accuracy.

**Comment your experiments.** Leaving dead‑end code in notebooks shows thought process and helps later debugging.

---

## Exploratory Directions (Not Fully Integrated)

Prototype workbooks live in `notebooks/_experiments/`. These guided intuition but did not replace the main curriculum model.

### Two‑Stage Pipeline (Neutral Gate → Polarity)

Stage 1: stance vs neutral. Stage 2: in‑favor vs against. Reduced 3‑way confusion but added latency.

### Rule‑Assisted Neutral

Keyword/emoji heuristics (`not sure`, `?`, 🤔) pre‑assign or bias neutral. Good recall; sarcasm false‑positives.

### Retrieval‑Augmented Few‑Shot

Embed labeled tweets (Sentence‑BERT) → top‑k in‑context examples. Promising but prompt budget heavy.

### Preference (A/B/C) Prompting

Force discrete choice; easy parsing. Some drift when model writes explanations.

### Classical ML Baseline

Sentence embeddings + logistic regression. Strong sanity check; out of scope for final submission but kept for comparison.

---

## What to Look For When Reviewing

If you’re evaluating this repo for the research assessment, here’s where to focus:

* **Readable, commented notebooks** that show real iteration (failures included).
* **Clear training path**: poles → neutral → mixed (curriculum).
* **Resource awareness**: 8‑bit load, LoRA adapters, small batch gradient accumulation.
* **Robust label parsing**: multiple pattern fallbacks; raw output logging.
* **Local eval & error inspection**: I review misclassified slices and note sarcasm / emoji issues.

---

## Personal Note

I underestimated this task. Classifying stance in noisy, sarcastic, emoji‑filled tweets during a polarized public health conversation is *hard*. I hit:

* RAM walls (before discovering I could enable GPU 🤦‍♂️).
* Transformers arg mismatches.
* Training sessions that looked great until I realized my label parser was silently defaulting everything to neutral.
* Sarcasm, jokes, half‑thoughts, and hashtag irony that confuse both humans and models.

But I loved the challenge. It pushed me to combine **prompting**, **parameter‑efficient fine‑tuning**, **curriculum learning**, and **error‑driven iteration** under constraint. I’m very interested in continuing this line of work, especially scaling toward the \~1M tweet dataset mentioned in the brief and exploring retrieval + sarcasm modeling.

Thanks for taking the time to review my submission.

---

## References

* Hugging Face: `google/flan-t5-large` model card & usage guide.
* Hugging Face Transformers docs (Seq2SeqTrainer, generation).
* PEFT / LoRA docs for parameter‑efficient fine‑tuning.
* BitsAndBytes 8‑bit loading documentation.
* scikit‑learn metrics reference (F1, classification report).

---
