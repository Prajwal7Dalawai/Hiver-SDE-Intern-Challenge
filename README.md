# Hiver SDE Intern Take-Home — AmazonHelp AI Support Agent

## 1. Overview

This repository contains a brand-specific AI customer-support agent built from the **Customer Support on Twitter** dataset (`thoughtvector/customer-support-on-twitter`).

The system is intentionally designed as a **conservative support copilot**, rather than a generic autonomous chatbot. It:

1. classifies an incoming customer message into an operational intent;
2. retrieves historically similar AmazonHelp customer-support cases;
3. uses those historical cases as grounding evidence for a response draft;
4. applies a risk-aware routing policy;
5. either auto-handles the request or escalates it to a human with an explicit reason.

The central design principle is:

> **The model should not invent a support resolution when the historical evidence is weak or the issue is risky.**

The project therefore treats evaluation, escalation, evidence quality, and failure analysis as first-class components.

---

## 2. Why AmazonHelp?

The dataset contains conversations between customers and many brands. I selected **AmazonHelp** because it provided the largest support corpus among the compared brands and substantial conversation depth.

Observed dataset statistics for AmazonHelp:

- Support tweets: **169,840**
- Unique customers: **71,049**
- Average support replies per customer: approximately **2.38**
- Customers receiving 5+ support replies: **7,102**
- Approximately **99.67%** of AmazonHelp support tweets had a parent tweet

This made AmazonHelp particularly useful for learning from historical support behavior rather than attempting to build a generic FAQ bot.

---

## 3. Problem Framing

### What "good" means

For this prototype, a good support agent should:

- identify the customer's operational problem;
- distinguish closely related support states;
- retrieve relevant historical support behavior;
- avoid unsupported promises or claims;
- produce a concise and customer-appropriate response;
- avoid leaking Twitter handles, URLs, tweet IDs, agent signatures, or dataset artifacts;
- escalate cases where autonomous handling is not sufficiently trustworthy.

### What I deliberately did not build

I did **not** attempt to build:

- a general-purpose Amazon chatbot;
- an end-to-end order-management system;
- live access to Amazon order/account information;
- a production-grade policy engine;
- a fully autonomous financial/account-security agent;
- a massive fine-tuned language model.

The historical dataset does not contain live account state, so the system cannot legitimately claim that a refund, cancellation, replacement, or account action has actually been performed.

---

## 4. Architecture

```text
Customer message
       |
       v
Conversation context
       |
       v
Casual-message routing
       |
       +---- greeting / acknowledgement ----> safe response
       |
       v
LLM intent classification
       |
       v
Multilingual semantic embedding
       |
       v
Historical AmazonHelp retrieval
       |
       v
Risk-aware routing
       |
       +---- AUTO_HANDLE
       |
       +---- ESCALATE
       |
       v
Grounded response generation
       |
       v
Deterministic output sanitization
       |
       v
Customer-facing response
```

The Streamlit interface also exposes internal diagnostics such as detected intent, routing decision, historical similarity, decision reason, and retrieved historical evidence.

---

## 5. Dataset Processing

The raw dataset contains approximately 2.8M tweets and the following columns:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

### Conversation reconstruction

The dataset contains both response lists and parent pointers. I used `in_response_to_tweet_id` as the primary mechanism for reconstructing conversation context because it directly identifies the parent message.

A terminal support tweet was **not** interpreted as proof that a case was resolved. A missing `response_tweet_id` only means that another response was not recorded in the dataset.

### Customer-support pair construction

For AmazonHelp, support tweets with a parent pointer were joined to their parent customer tweet to create immediate customer → support pairs.

After cleaning and filtering low-information records, the processed historical corpus contains roughly **167k usable customer-support pairs**. The exact count can vary slightly with preprocessing version; the deployed retrieval artifact should be treated as the source of truth.

### Cleaning

Customer messages were normalized by:

- removing Twitter-style handles;
- removing URLs;
- decoding `&amp;`;
- normalizing whitespace.

I deliberately did **not** impose an aggressive character-length cutoff. Short multilingual messages can still carry meaningful support intent.

---

## 6. Intent Taxonomy

The final operational taxonomy contains **15 supported intents + OTHER**.

| Intent | Definition |
|---|---|
| `DELIVERY_DELAY` | Expected/promised delivery date has passed |
| `DELIVERY_NOT_RECEIVED` | Customer does not have the shipment/package |
| `DELIVERY_TRACKING` | Tracking/status is missing, stuck, or unclear |
| `DELIVERY_DRIVER_ISSUE` | Carrier/driver/delivery attempt/instructions issue |
| `ORDER_STATUS` | General order state/status |
| `ORDER_CANCELLATION` | Customer wants cancellation or reports cancellation issue |
| `MISSING_ITEM` | Package arrived but expected item(s) are missing |
| `WRONG_ITEM_RECEIVED` | Incorrect product was received |
| `RETURN_REPLACEMENT` | Explicit return or replacement request |
| `REFUND_STATUS` | Refund request/status |
| `ACCOUNT_ACCESS` | Login, lockout, password, verification access |
| `ACCOUNT_SECURITY` | Hacked, suspicious, fraudulent, unauthorized activity |
| `UNEXPECTED_CHARGE` | Unrecognized, duplicate, or unexpected charge |
| `CASHBACK` | Expected cashback/reward not received |
| `PRODUCT_ISSUE` | Damaged, defective, broken, faulty, poor-quality, incompatible product |
| `OTHER` | Unsupported, ambiguous, casual, or insufficient-evidence request |

### Important boundary rules

- A missed promised date → `DELIVERY_DELAY`
- A package simply absent → `DELIVERY_NOT_RECEIVED`
- Tracking/status uncertainty → `DELIVERY_TRACKING`
- Driver/carrier/attempt/instructions → `DELIVERY_DRIVER_ISSUE`
- Missing item after package arrival remains `MISSING_ITEM`
- Broken product + explicit return/replace request → `RETURN_REPLACEMENT`
- Broken/defective product without explicit return request → `PRODUCT_ISSUE`
- Prime is treated as context, not an intent
- `OTHER` is an explicit rejection class rather than a fallback to force every message into a supported category

---

## 7. Intent Discovery

Intent discovery began with exploratory TF-IDF analysis and clustering.

English MiniLM embeddings produced clusters that were partly driven by language rather than operational support behavior. I therefore moved to a multilingual embedding model:

**`paraphrase-multilingual-MiniLM-L12-v2`**

K-Means was used only as a discovery aid. The clusters were inspected manually and translated into operational categories.

Silhouette scores were low (approximately 0.035–0.047 depending on K), so clustering was **not** treated as ground truth.

This distinction is important: the final taxonomy reflects operationally meaningful boundaries, not an arbitrary clustering output.

---

## 8. Golden Evaluation Set

A frozen golden set of **213 human-labelled examples** was created.

Candidate generation used a 10,000-message sample with semantic retrieval to find examples around each proposed intent, plus random examples. Prototype retrieval was used only to discover candidates.

The final human labels are the source of truth.

### Golden-set distribution

| Intent | Count |
|---|---:|
| OTHER | 37 |
| ACCOUNT_SECURITY | 12 |
| DELIVERY_DELAY | 11 |
| ORDER_CANCELLATION | 19 |
| DELIVERY_NOT_RECEIVED | 16 |
| DELIVERY_TRACKING | 16 |
| PRODUCT_ISSUE | 13 |
| REFUND_STATUS | 17 |
| ACCOUNT_ACCESS | 15 |
| ORDER_STATUS | 12 |
| UNEXPECTED_CHARGE | 9 |
| RETURN_REPLACEMENT | 9 |
| CASHBACK | 8 |
| DELIVERY_DRIVER_ISSUE | 8 |
| MISSING_ITEM | 6 |
| WRONG_ITEM_RECEIVED | 5 |
| **Total** | **213** |

The set was frozen before final evaluation; it was not repeatedly tuned to improve the reported result.

---

## 9. Baselines and Evaluation

### Baseline 1 — Majority class

Always predict `OTHER`.

- Accuracy: **17.37%**
- Macro F1: **0.0185**
- Weighted F1: **0.0514**

This is intentionally trivial and establishes a lower bound.

### Baseline 2 — TF-IDF + Logistic Regression

A conventional lexical classifier was evaluated using stratified 5-fold cross-validation.

- Accuracy: **35.21%**
- Macro F1: **0.3344**
- Weighted F1: **0.3437**

### Semantic prototype classifier

A prototype classifier using 15 intents × 4 canonical examples was also evaluated.

- Accuracy: **42.72%**
- Macro F1: **0.4306**
- Weighted F1: **0.4003**

This improved over the simple lexical baseline but remained weak on fine-grained operational boundaries and `OTHER`.

### LLM intent classifier

The final LLM classifier uses Groq with:

**`openai/gpt-oss-120b`**

On the frozen 213-example golden set:

- Accuracy: **61.0%**
- Macro F1: **0.61**
- Weighted F1: **0.61**

The strongest areas included delivery-driver issues and several concrete delivery/item intents. The weakest areas included delivery delay, order status, product issue, return/replacement, and OTHER.

---

## 10. Retrieval

Historical customer-support pairs are embedded using:

**`paraphrase-multilingual-MiniLM-L12-v2`**

The normalized query embedding is compared with normalized historical embeddings using cosine similarity, implemented as a dot product.

Top-k historical cases are passed to the generation step as evidence.

Observed retrieval statistics on the 213-case golden set:

- top-1 similarity mean: **0.897**
- top-1 similarity median: **0.888**
- top-5 mean similarity: **0.832**

High similarity does not imply correct intent. Closely related delivery cases can have high lexical/semantic similarity while requiring different operational responses.

An attempted silver-intent agreement signal was rejected because even unanimous silver labels matched the human intent only about **56.3%** of the time.

---

## 11. Routing / Escalation

The system uses conservative routing.

### Always escalate

- `ACCOUNT_SECURITY`
- `UNEXPECTED_CHARGE`
- `OTHER`
- weak historical evidence
- low-confidence intent categories

The current UI policy also treats several historically difficult intents conservatively, including:

- `DELIVERY_DELAY`
- `CASHBACK`
- `MISSING_ITEM`
- `PRODUCT_ISSUE`
- `REFUND_STATUS`
- `ORDER_STATUS`

The evaluation experiment that combined intent, similarity, and retrieval-agreement signals produced:

- **77 AUTO_HANDLE**
- **136 ESCALATE**
- **36.2% automation coverage**
- **70.1% safe automation precision**

This is an evaluation-policy result, not a claim that 36.2% of real-world Amazon support traffic can safely be automated.

---

## 12. Response Generation

Generation is grounded in retrieved historical customer/support pairs.

The generation prompt explicitly instructs the model to:

- use historical support behavior as evidence;
- avoid inventing policies, refunds, guarantees, or actions;
- never claim an action happened unless supported;
- avoid copying historical responses verbatim;
- remove handles, tweet IDs, agent identifiers and URLs;
- ask for information when evidence is insufficient;
- respond concisely and professionally.

A deterministic sanitization layer removes common artifacts such as:

- URLs;
- Twitter handles;
- trailing agent-style signatures;
- unresolved `__placeholder__` tokens;
- whitespace/punctuation artifacts.

---

## 13. LLM-as-Judge

A six-dimensional rubric was used:

1. Relevance
2. Groundedness
3. No hallucination
4. Customer appropriateness
5. Resolution alignment
6. Overall quality

Scores are 1–5.

A **39-case diagnostic sample** produced:

- Relevance: **4.95**
- Groundedness: **4.85**
- No hallucination: **4.97**
- Customer appropriateness: **4.87**
- Resolution alignment: **4.92**
- Overall: **4.92 / 5**

However, the judge used the same provider/model family as generation, so it is not an independent evaluator.

More importantly, a separate **15-case human audit** found substantial disagreement: only **1/15** responses met the strict human acceptability criteria, while the LLM judge accepted all 15. The audit caught URLs, handles, placeholders, and unsupported claims that the judge overlooked.

Therefore, the LLM judge is treated as a **diagnostic signal, not proof of safety**.

---

## 14. Top Failure Modes

### 1. Fine-grained delivery confusion

`DELIVERY_DELAY`, `DELIVERY_NOT_RECEIVED`, `DELIVERY_TRACKING`, and `DELIVERY_DRIVER_ISSUE` share vocabulary such as "package", "delivery", "tracking", and "late".

**Hypothesis:** semantic similarity collapses operationally distinct states.

**Mitigation:** explicit boundary rules and conservative escalation.

### 2. Refund vs cashback ambiguity

Financial-status language can look similar even though cashback and refunds are different workflows.

**Mitigation:** route conservatively when the transaction type is unclear.

### 3. OTHER rejection weakness

Messages containing Amazon/support vocabulary can be forced into a supported intent even when they are generic, casual, or unsupported.

**Mitigation:** preserve OTHER as an explicit rejection class and use conversational routing.

### 4. Underlying issue vs requested resolution

A single message can mention a product issue, missing item, cancellation, and refund.

**Mitigation:** prioritize the underlying operational issue unless the user explicitly requests a return/replacement.

### 5. Grounded-generation artifact leakage

Historical Twitter responses contain URLs, handles, signatures, and dataset-specific artifacts.

**Mitigation:** explicit generation constraints plus deterministic sanitization.

---

## 15. What Is Misleading About My Headline Number?

The most tempting headline is the **36.2% automation coverage** or **70.1% safe automation precision**.

Neither number should be interpreted as production-ready automation capability.

The evaluation set contains 213 human-labelled examples and was deliberately constructed to cover supported intents, including rare categories. It therefore does not represent the natural distribution of incoming support traffic.

In addition, the LLM judge produced a 4.92/5 average on a 39-case sample, but a separate 15-case human audit found substantial disagreement. Responses containing historical URLs, handles, placeholders, or unsupported claims were often rated highly by the LLM judge.

This demonstrates that judge scores can overestimate safety. The headline metrics should therefore be interpreted as **prototype evaluation results under a deliberately constructed test set**, not a guarantee of autonomous resolution.

The strongest claim this project supports is that a conservative pipeline can combine historical retrieval, intent classification, evidence-based drafting, and explicit escalation while making its failure modes measurable.

---

## 16. Decision Log

1. **Selected AmazonHelp** because it offered the largest and deepest support corpus among the compared brands.
2. **Used parent pointers** rather than relying primarily on `response_tweet_id` lists for conversation reconstruction.
3. **Did not equate terminal support tweets with resolution**, because absence of another recorded response is not proof of resolution.
4. **Removed handles and URLs from customer text** to reduce identity noise and retrieval contamination.
5. **Kept short multilingual messages** rather than applying a blanket 15-character cutoff.
6. **Moved to multilingual embeddings** after English-only clustering showed language-driven structure.
7. **Used clustering only for discovery**, not as ground-truth intent definition.
8. **Defined 15 operational intents + OTHER** to capture actionable support boundaries.
9. **Used human annotation as the golden-set source of truth**; prototype retrieval was only candidate discovery.
10. **Froze the 213-example golden set** before final evaluation.
11. **Kept a semantic prototype classifier as a baseline**, rather than presenting it as the final production classifier.
12. **Did not use silver intent labels as a hard retrieval filter**, because their agreement with human labels was weak.
13. **Rejected silver-intent unanimous agreement as an escalation signal** after only ~56.3% agreement with human intent.
14. **Always escalate account-security and unexpected-charge issues** because the cost of a wrong autonomous action is high.
15. **Added deterministic response sanitization** because prompt-only controls were insufficient to prevent historical artifacts.
16. **Treat the LLM judge as diagnostic** because the human audit demonstrated that high judge scores can miss obvious grounding defects.
17. **Kept conversation/session state in the Streamlit application** so follow-up questions can use the ongoing conversation rather than treating every message as independent.

---

## 17. Repository Structure

```text
Hiver-SDE-Intern-Challenge/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── .env                    # local only; never commit
├── data/
│   ├── raw/                # local only; not required for deployment
│   ├── processed/
│   │   ├── amazon_support_pairs.parquet
│   │   ├── amazon_customer_embeddings.npy
│   │   └── ...
│   └── golden/
│       └── golden_set.csv
├── notebooks/
│   ├── dataset_exploration.ipynb
│   ├── 02_intent_discovery.ipynb
│   └── 03_evaluation.ipynb
├── scripts/
├── src/
└── tests/
```

Large deployment artifacts should be stored with **Git LFS** rather than normal Git objects.

Do not commit:

- `.env`
- `.venv/`
- raw multi-million-row dataset
- caches
- temporary files
- API keys

---

## 18. Local Setup

### Windows PowerShell

```powershell
cd D:\Projects\Hiver-SDE-Intern-Challenge

python -m venv .venv
.venv\Scripts\activate

python -m pip install -r requirements.txt
```

Create `.env`:

```env
GROQ_API_KEY=your_actual_key
```

Then run:

```powershell
python -m streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

---

## 19. Reproducing the Evaluation

The project intentionally separates expensive dataset preparation from evaluation.

The processed retrieval artifacts are precomputed so that the evaluation does not require re-embedding the entire historical corpus.

Typical evaluation flow:

```text
1. Install requirements
2. Load frozen golden_set.csv
3. Load precomputed historical embeddings
4. Load processed AmazonHelp pairs
5. Run classifier evaluation
6. Run routing evaluation
7. Optionally run the LLM response/judge experiment
```

The LLM-based benchmark requires a valid Groq API key.

The headline deterministic metrics are:

```text
Majority:
Accuracy       17.37%
Macro F1        0.0185
Weighted F1     0.0514

TF-IDF + LR:
Accuracy       35.21%
Macro F1        0.3344
Weighted F1     0.3437

Semantic prototype:
Accuracy       42.72%
Macro F1        0.4306
Weighted F1     0.4003

LLM intent classifier:
Accuracy       ~61.0%
Macro F1       ~0.61
Weighted F1    ~0.61
```

The LLM judge and generation calls are API-dependent and may vary slightly across runs.

---

## 20. Deployment

Recommended deployment target: **Streamlit Community Cloud**.

Before deployment:

1. Ensure `.env` is ignored.
2. Put required large artifacts under Git LFS.
3. Push the repository to GitHub.
4. Configure `GROQ_API_KEY` using Streamlit Cloud Secrets.
5. Deploy `app.py` as the entry point.

Cloud secret:

```toml
GROQ_API_KEY = "your_actual_key"
```

Never commit the secret to GitHub.

---

## 21. Limitations

- The source data is historical Twitter support data rather than live Amazon operational data.
- Intent boundaries are human-defined and imperfect.
- The golden set is only 213 examples.
- Rare classes have small sample sizes.
- Retrieval similarity can be high for the wrong operational state.
- LLM classification can confuse neighboring intents.
- Generation quality depends on retrieved evidence and the generation model.
- The LLM judge is not independent and can overrate unsafe responses.
- The current automation policy is intentionally conservative and should not be interpreted as production authorization.
- The system cannot actually inspect orders, accounts, payments, or shipments.

---

## 22. One-Week Next Steps

If given one additional week, I would prioritize:

### 1. Better intent calibration
Build a held-out validation split and calibrate confidence/margins instead of using fixed similarity thresholds alone.

### 2. Context-aware classification
Use multi-turn conversation state explicitly when resolving ambiguous follow-ups such as "still not received" or "what about my refund?"

### 3. Retrieval reranking
Add a cross-encoder or lightweight reranker to distinguish semantically close but operationally different cases.

### 4. Better evidence selection
Retrieve cases jointly using intent + query rather than query similarity alone.

### 5. Human-reviewed generation set
Expand the human audit substantially and measure groundedness and artifact leakage directly.

### 6. Better automation objective
Optimize for a risk-weighted utility such as:

```text
expected benefit of automation
-
expected cost of an incorrect autonomous response
```

rather than maximizing raw automation coverage.

### 7. Production observability
Track intent distribution, escalation rate, retrieval confidence, response latency, and human override rate.

---

## 23. Summary

This project is intentionally a **measured prototype**.

The main contribution is not a large language model or a generic chatbot. It is the end-to-end engineering of:

- a brand-specific support corpus;
- an operational intent taxonomy;
- historical-resolution retrieval;
- grounded response drafting;
- risk-aware escalation;
- a frozen human-labelled evaluation set;
- baseline comparisons;
- LLM-judge diagnostics;
- human-vs-judge validation;
- explicit failure analysis.

The system is designed to make a support automation decision **only when the evidence is sufficiently trustworthy**, and to make uncertainty visible when it is not.
