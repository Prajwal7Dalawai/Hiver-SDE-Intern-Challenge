import os
import re
import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="AmazonHelp AI Support Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 5rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.18);
        }

        .hero {
            padding: 0.25rem 0 1.4rem 0;
        }

        .hero-title {
            font-size: 2.45rem;
            font-weight: 750;
            letter-spacing: -0.04em;
            margin-bottom: 0.25rem;
        }

        .hero-subtitle {
            color: #8d93a1;
            font-size: 1rem;
            margin-bottom: 0;
        }

        .status-card {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin: 0.5rem 0 1rem 0;
            background: rgba(128, 128, 128, 0.045);
        }

        .status-title {
            font-weight: 650;
            margin-bottom: 0.15rem;
        }

        .status-text {
            color: #8d93a1;
            font-size: 0.88rem;
        }

        .example-label {
            color: #8d93a1;
            font-size: 0.88rem;
            margin-bottom: 0.35rem;
        }

        .metric-note {
            color: #8d93a1;
            font-size: 0.76rem;
            line-height: 1.35;
        }

        .footer-note {
            text-align: center;
            color: #777d8a;
            font-size: 0.75rem;
            padding-top: 1.5rem;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 14px;
        }

        .evidence-box {
            border-left: 3px solid rgba(255, 153, 0, 0.65);
            padding-left: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Configuration
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

GENERATION_MODEL = "openai/gpt-oss-120b"

SUPPORTED_INTENTS = [
    "DELIVERY_DELAY",
    "DELIVERY_NOT_RECEIVED",
    "DELIVERY_TRACKING",
    "DELIVERY_DRIVER_ISSUE",
    "ORDER_STATUS",
    "ORDER_CANCELLATION",
    "MISSING_ITEM",
    "WRONG_ITEM_RECEIVED",
    "RETURN_REPLACEMENT",
    "REFUND_STATUS",
    "ACCOUNT_ACCESS",
    "ACCOUNT_SECURITY",
    "UNEXPECTED_CHARGE",
    "CASHBACK",
    "PRODUCT_ISSUE",
    "OTHER",
]

RISK_INTENTS = {
    "ACCOUNT_SECURITY",
    "UNEXPECTED_CHARGE",
}

LOW_CONFIDENCE_INTENTS = {
    "DELIVERY_DELAY",
    "CASHBACK",
    "MISSING_ITEM",
    "PRODUCT_ISSUE",
    "REFUND_STATUS",
    "ORDER_STATUS",
}


# ============================================================
# Conversation state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ============================================================
# Load model and historical data
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )


@st.cache_data
def load_historical_data():
    pairs = pd.read_parquet(
        "data/processed/amazon_support_pairs.parquet"
    )

    embeddings = np.load(
        "data/processed/amazon_customer_embeddings.npy"
    )

    return pairs, embeddings


model = load_embedding_model()
amazon_pairs, historical_embeddings = load_historical_data()


# ============================================================
# Intent classification
# ============================================================

def build_intent_prompt(query):
    return f"""
You are an intent classifier for an Amazon customer-support agent.

Classify the customer conversation into exactly ONE intent.

Supported intents:

DELIVERY_DELAY
Expected/promised delivery date has passed.

DELIVERY_NOT_RECEIVED
Customer does not have the package/order.

DELIVERY_TRACKING
Customer primarily needs tracking/status information or tracking
has not updated.

DELIVERY_DRIVER_ISSUE
Problem involving driver, carrier, delivery attempt, safe place,
delivery instructions, or delivery behavior.

ORDER_STATUS
General order status, especially before/around fulfillment.

ORDER_CANCELLATION
Customer wants an order cancelled or reports an unexpected cancellation.

MISSING_ITEM
Order/package arrived but one or more expected items are missing.

WRONG_ITEM_RECEIVED
Customer received an incorrect or duplicate product.

RETURN_REPLACEMENT
Customer explicitly wants to return or replace an item.

REFUND_STATUS
Customer asks about a refund or refund status.

ACCOUNT_ACCESS
Login, password, locked account, or account access problem.

ACCOUNT_SECURITY
Hacked account, unauthorized access, fraud, or suspicious account activity.

UNEXPECTED_CHARGE
Unrecognized, duplicate, or unexpected charge.

CASHBACK
Expected cashback/reward was not received.

PRODUCT_ISSUE
Product is damaged, defective, broken, faulty, incompatible,
or not working, without an explicit return/replace request.

OTHER
Does not clearly fit the supported taxonomy.

Rules:
- A known missed delivery date -> DELIVERY_DELAY.
- Package simply absent -> DELIVERY_NOT_RECEIVED.
- Tracking/status uncertainty -> DELIVERY_TRACKING.
- Explicit driver/carrier/attempt/instruction issue -> DELIVERY_DRIVER_ISSUE.
- Explicit return/replace request -> RETURN_REPLACEMENT.
- Broken/defective product without explicit return/replace -> PRODUCT_ISSUE.
- Prime membership is context, not an intent.
- If the message is insufficiently specific, use OTHER.
- Use the conversation context when the latest message is a follow-up.
- Do not force a casual greeting or generic complaint into a supported intent.

CONVERSATION:
{query}

Return ONLY the intent name.
"""


def classify_intent(query):
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "user",
                "content": build_intent_prompt(query),
            }
        ],
        temperature=0,
    )

    intent = response.choices[0].message.content.strip()

    if intent not in SUPPORTED_INTENTS:
        intent = "OTHER"

    return intent


# ============================================================
# Retrieval
# ============================================================

def retrieve_similar_cases(query, top_k=5):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    scores = historical_embeddings @ query_embedding
    indices = np.argsort(scores)[::-1][:top_k]

    results = amazon_pairs.iloc[indices].copy()
    results["similarity"] = scores[indices]

    return results


# ============================================================
# Conversational routing
# ============================================================

CASUAL_PATTERNS = [
    r"^\s*(hi|hello|hey|hiya|howdy)\s*[!.?]*\s*$",
    r"^\s*(hi|hello|hey)[,\s]+how\s+are\s+you\s*[!.?]*\s*$",
    r"^\s*how\s+are\s+you(\s+doing)?\s*[!.?]*\s*$",
    r"^\s*(good\s+morning|good\s+afternoon|good\s+evening)\s*[!.?]*\s*$",
    r"^\s*(thanks|thank\s+you|thx|ty)\s*[!.?]*\s*$",
    r"^\s*(ok|okay|got\s+it|great|perfect|sounds\s+good)\s*[!.?]*\s*$",
]


def is_casual_message(text):
    text = str(text).strip().lower()
    if not text:
        return True
    return any(re.match(pattern, text) for pattern in CASUAL_PATTERNS)


def casual_response(text):
    text = str(text).strip().lower()

    if re.match(r"^\s*(thanks|thank\s+you|thx|ty)\b", text):
        return (
            "You're welcome! If you need help with an Amazon order "
            "or another issue, just let me know."
        )

    if re.match(r"^\s*(ok|okay|got\s+it|great|perfect|sounds\s+good)\b", text):
        return "You're welcome! If anything else comes up, I'm happy to help."

    return (
        "Hi! I'm here to help with Amazon orders, deliveries, refunds, "
        "accounts, and product issues. What can I help you with?"
    )


# ============================================================
# Decision engine
# ============================================================

def make_decision(intent, top_similarity, casual=False):
    if casual:
        return (
            "AUTO_HANDLE",
            "Casual conversation does not require support escalation.",
        )

    if intent in RISK_INTENTS:
        return (
            "ESCALATE",
            f"{intent} is a high-risk issue and requires human review.",
        )

    if intent == "OTHER":
        return (
            "ESCALATE",
            "The message does not clearly fit a supported intent.",
        )

    if top_similarity < 0.85:
        return (
            "ESCALATE",
            "Historical evidence is not sufficiently similar.",
        )

    if intent in LOW_CONFIDENCE_INTENTS:
        return (
            "ESCALATE",
            f"{intent} is treated conservatively because "
            "classification reliability is lower.",
        )

    return (
        "AUTO_HANDLE",
        "Supported intent with sufficiently strong historical evidence.",
    )


# ============================================================
# Response generation
# ============================================================

def build_generation_prompt(query, results, decision):
    evidence = ""

    for i, (_, row) in enumerate(results.iterrows(), 1):
        evidence += f"""
CASE {i}

Customer:
{row["customer_text"]}

Historical AmazonHelp response:
{row["support_text"]}
"""

    escalation_instruction = ""
    if decision == "ESCALATE":
        escalation_instruction = """
This case is being routed for human review. Still provide a useful,
customer-facing response. Acknowledge the issue, explain the next
reasonable step, and ask only for information that would genuinely
help the support team investigate. Do not mention internal routing,
classification, confidence, retrieval, or escalation machinery.
"""

    return f"""
You are an Amazon customer-support response assistant.

Draft a concise, natural customer-facing response using historical
AmazonHelp cases as evidence.

CUSTOMER CONVERSATION:
{query}

HISTORICAL SUPPORT CASES:
{evidence}

Rules:
- Ground the response in the historical support behavior.
- Respond in the same language as the customer's latest message unless
  the customer clearly asks for another language.
- Never switch languages merely because retrieved historical evidence
  is written in another language.
- Do not invent policies, refunds, guarantees, timelines, or actions.
- Do not claim that an action has been taken unless the evidence supports
  that exact action.
- Do not copy historical responses verbatim.
- Do not reproduce Twitter usernames, handles, tweet IDs, agent
  identifiers, URLs, placeholders, or dataset artifacts.
- Never expose historical links or internal contact details.
- Adapt the response naturally to the customer's actual situation.
- Be concise, professional, and empathetic.
- If the evidence is insufficient, ask for the appropriate information
  rather than inventing an answer.
- For casual greetings, thanks, or acknowledgements, respond naturally.
- Do not force casual conversation into a support intent.
{escalation_instruction}

Return ONLY the customer-facing response.
"""


def generate_response(query, results, decision):
    prompt = build_generation_prompt(
        query,
        results,
        decision,
    )

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# Output safety
# ============================================================

def sanitize_response(response):
    response = re.sub(r"https?://\S+", "", response)
    response = re.sub(r"(?<!\w)@\w+", "", response)
    response = re.sub(r"\s*\^?[A-Z]{1,4}\s*$", "", response)
    response = re.sub(r"__[^_]+__", "", response)
    response = re.sub(r"\s+", " ", response)
    response = re.sub(r"\s+([,.!?])", r"\1", response)

    return response.strip()


def sanitize_evidence(text):
    text = re.sub(r"https?://\S+", "", str(text))
    text = re.sub(r"(?<!\w)@\w+", "", text)
    text = re.sub(r"__[^_]+__", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fallback_response(intent):
    fallbacks = {
        "ACCOUNT_SECURITY": (
            "I’m sorry you’re dealing with this. Because this involves "
            "account security, a support specialist should review it. "
            "Please avoid sharing passwords or other sensitive information here."
        ),
        "UNEXPECTED_CHARGE": (
            "I’m sorry about the unexpected charge. A support specialist "
            "should review the transaction with you. Please avoid sharing "
            "full payment details in this chat."
        ),
        "OTHER": (
            "Hi! I’m happy to help. Please tell me a little more about "
            "the order, delivery, account, or product issue you’re experiencing."
        ),
    }

    return fallbacks.get(
        intent,
        "I’m sorry you’re having trouble. Please share a little more detail "
        "about the issue so we can point you in the right direction.",
    )


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🤖 AmazonHelp AI Support Agent</div>
        <div class="hero-subtitle">
            Brand-specific support grounded in historical AmazonHelp resolutions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Agent")

    st.markdown(
        """
        **How it works**

        1. Understand the conversation
        2. Classify the support intent
        3. Retrieve similar historical cases
        4. Apply risk-aware routing
        5. Draft a grounded response
        """
    )

    st.divider()

    st.subheader("Evaluation snapshot")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Coverage", "36.2%")
        st.metric("Golden set", "213")

    with col2:
        st.metric("Safe precision", "70.1%")
        st.metric("Intents", "15 + OTHER")

    st.caption(
        "Coverage and safe precision are measured on the frozen "
        "213-case human-labelled evaluation set."
    )

    st.divider()

    st.subheader("LLM judge")

    st.metric("Sample score", "4.92 / 5")

    st.markdown(
        '<div class="metric-note">'
        "39-case diagnostic sample only. A separate human audit found "
        "the judge could miss grounding/artifact problems."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("Session")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.caption(
        f"{len(st.session_state.messages)} messages in this session"
    )


# ============================================================
# Empty-state / examples
# ============================================================

if not st.session_state.messages:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-title">How can I help?</div>
            <div class="status-text">
                Ask about a delivery, order, refund, account, product,
                or another Amazon support issue. You can continue with
                follow-up questions in the same conversation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="example-label">Try an example</div>',
        unsafe_allow_html=True,
    )

    examples = [
        "My package was supposed to arrive yesterday but it still hasn't arrived.",
        "My order was delivered but one of the two items is missing.",
        "The tracking hasn't updated in three days.",
    ]

    example_cols = st.columns(3)

    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(
                example,
                key=f"example_{i}",
                use_container_width=True,
            ):
                st.session_state.pending_query = example
                st.rerun()


# ============================================================
# Conversation history
# ============================================================

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("metadata"):
            metadata = message["metadata"]

            with st.expander("Agent details"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.caption("Detected intent")
                    st.code(metadata["intent"])

                with col2:
                    st.caption("Routing")
                    if metadata["decision"] == "AUTO_HANDLE":
                        st.success("AUTO-HANDLE")
                    else:
                        st.warning("HUMAN REVIEW")

                with col3:
                    st.caption("Top evidence similarity")
                    st.write(f'{metadata["similarity"]:.3f}')

                st.caption(metadata["reason"])

                with st.expander("Historical evidence"):
                    for i, case in enumerate(metadata["evidence"], 1):
                        st.markdown(
                            f"**Case {i} · similarity {case['similarity']:.3f}**"
                        )

                        st.markdown("**Customer**")
                        st.write(sanitize_evidence(case["customer"]))

                        st.markdown("**Historical response**")
                        st.write(sanitize_evidence(case["response"]))

                        if i < len(metadata["evidence"]):
                            st.divider()


# ============================================================
# Input
# ============================================================

typed_query = st.chat_input(
    "Describe your issue or ask a follow-up..."
)

query = typed_query or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    recent_messages = st.session_state.messages[-6:]

    conversation_context = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in recent_messages
    )

    with st.chat_message("assistant", avatar="🤖"):
        error_details = None

        with st.status("Working on your request...", expanded=False) as status:
            try:
                casual = is_casual_message(query)

                if casual:
                    status.write("Handling conversational message...")
                    intent = "OTHER"
                    results = pd.DataFrame(
                        columns=["customer_text", "support_text", "similarity"]
                    )
                    top_similarity = 1.0
                    decision, reason = make_decision(
                        intent,
                        top_similarity,
                        casual=True,
                    )
                    response = casual_response(query)

                else:
                    status.write("Understanding the support issue...")
                    intent = classify_intent(conversation_context)

                    status.write("Finding similar historical cases...")
                    results = retrieve_similar_cases(
                        conversation_context,
                        top_k=5,
                    )

                    top_similarity = float(
                        results.iloc[0]["similarity"]
                    )

                    status.write("Applying risk-aware routing...")
                    decision, reason = make_decision(
                        intent,
                        top_similarity,
                    )

                    status.write("Drafting a grounded response...")
                    response = generate_response(
                        conversation_context,
                        results,
                        decision,
                    )
                    response = sanitize_response(response)

                    if not response:
                        response = fallback_response(intent)

                status.update(
                    label="Response ready",
                    state="complete",
                    expanded=False,
                )

            except Exception as exc:
                # Keep backend/API details out of the customer-facing chat.
                error_details = f"{type(exc).__name__}: {str(exc)}"

                intent = "SYSTEM_ERROR"
                decision = "ESCALATE"
                reason = "The support agent encountered an internal error."
                top_similarity = 0.0
                results = pd.DataFrame(
                    columns=["customer_text", "support_text", "similarity"]
                )

                response = (
                    "There was a temporary problem processing your request. "
                    "Please try again in a moment. If the problem continues, "
                    "a support specialist can help you."
                )

                status.update(
                    label="Something went wrong",
                    state="error",
                    expanded=False,
                )

        st.markdown(response)

        if error_details:
            st.warning(
                "There has been a temporary problem processing this request. "
                "Please try again. If it continues, a support specialist can help."
            )

            with st.expander("View technical details"):
                st.code(error_details, language="text")

        else:
            with st.expander("Agent details"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.caption("Detected intent")
                    st.code(intent)

                with col2:
                    st.caption("Routing")
                    if decision == "AUTO_HANDLE":
                        st.success("AUTO-HANDLE")
                    else:
                        st.warning("HUMAN REVIEW")

                with col3:
                    st.caption("Top evidence similarity")
                    st.write(f"{top_similarity:.3f}")

                st.caption(reason)

                if not results.empty:
                    with st.expander("Historical evidence"):
                        for i, (_, case) in enumerate(
                            results.head(3).iterrows(),
                            1,
                        ):
                            st.markdown(
                                f"**Case {i} · similarity {case['similarity']:.3f}**"
                            )

                            st.markdown("**Customer**")
                            st.write(
                                sanitize_evidence(case["customer_text"])
                            )

                            st.markdown("**Historical response**")
                            st.write(
                                sanitize_evidence(case["support_text"])
                            )

                            if i < 3:
                                st.divider()

    evidence = []

    for _, case in results.head(3).iterrows():
        evidence.append(
            {
                "customer": case["customer_text"],
                "response": case["support_text"],
                "similarity": float(case["similarity"]),
            }
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "metadata": {
                "intent": intent,
                "decision": decision,
                "similarity": top_similarity,
                "reason": reason,
                "evidence": evidence,
            },
        }
    )

    st.rerun()


st.markdown(
    '<div class="footer-note">'
    "Prototype · Historical support data is used as evidence, not as a guarantee."
    "</div>",
    unsafe_allow_html=True,
)
