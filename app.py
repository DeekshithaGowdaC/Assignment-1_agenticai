
import io
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# 1. STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Finance Agent",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Statement-to-Insights Finance Agent")

st.caption(
    "AI-Powered Personal Finance Assistant using "
    "Ollama, Qwen 2.5, LangChain, ChromaDB and RAG"
)


# ==========================================
# 2. SIDEBAR CONFIGURATION
# ==========================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.success("Local AI Model: Qwen 2.5 (1.5B)")
    st.caption("No Gemini API key required.")

    show_debug = st.checkbox(
        "Show Agent Debug Information",
        value=False
    )

    st.divider()

    st.header("📁 Upload Bank Statement")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )


# ==========================================
# 3. INITIALIZE LOCAL LLM
# ==========================================

@st.cache_resource
def initialize_llm():

    return ChatOllama(
        model="qwen2.5:1.5b",
        temperature=0.2
    )


try:

    llm = initialize_llm()

except Exception as e:

    st.error(f"Ollama initialization failed: {e}")
    st.info("Check that Ollama is running and Qwen is installed.")
    st.stop()


# ==========================================
# 4. HUGGING FACE EMBEDDINGS
# ==========================================

@st.cache_resource
def initialize_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


try:

    embeddings = initialize_embeddings()

except Exception as e:

    st.error(f"Embedding initialization failed: {e}")
    st.stop()


# ==========================================
# 5. CHROMADB VECTOR DATABASE
# ==========================================

@st.cache_resource
def initialize_vector_db():

    db = Chroma(
        collection_name="finance_guidelines",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    rules = [

        Document(
            page_content=(
                "Dining and food delivery expenses should "
                "ideally remain below 15% of monthly spending."
            ),
            metadata={"category": "Dining"}
        ),

        Document(
            page_content=(
                "Entertainment and subscriptions should "
                "ideally remain below 10% of monthly spending."
            ),
            metadata={"category": "Entertainment"}
        ),

        Document(
            page_content=(
                "Groceries and household essentials generally "
                "require around 25-30% of the monthly budget."
            ),
            metadata={"category": "Groceries"}
        ),

        Document(
            page_content=(
                "Utility bills and rent are fixed expenses. "
                "Pay them on time to avoid late fees."
            ),
            metadata={"category": "Utilities"}
        ),

        Document(
            page_content=(
                "Save a portion of monthly income and maintain "
                "an emergency fund."
            ),
            metadata={"category": "Savings"}
        ),

        Document(
            page_content=(
                "Review unnecessary subscriptions and recurring "
                "payments regularly."
            ),
            metadata={"category": "Subscriptions"}
        )
    ]

    if db._collection.count() == 0:
        db.add_documents(rules)

    return db


try:

    vector_db = initialize_vector_db()

except Exception as e:

    st.error(f"ChromaDB initialization failed: {e}")
    st.stop()


# ==========================================
# 6. SAMPLE BANK STATEMENT
# ==========================================

def load_sample_data():

    sample_csv = """date,description,amount,type
2026-10-01,Zomato Food Delivery,450,DEBIT
2026-10-02,Supermarket Groceries,2100,DEBIT
2026-10-03,Salary Credit,55000,CREDIT
2026-10-05,Netflix Subscription,649,DEBIT
2026-10-08,Swiggy Gourmet,820,DEBIT
2026-10-10,Electricity Bill,1850,DEBIT
2026-10-12,Uber Ride,340,DEBIT
2026-10-15,Blinkit Groceries,950,DEBIT
2026-10-18,Starbucks Coffee,410,DEBIT
2026-10-22,Amazon Shopping,3400,DEBIT
2026-10-25,PVR Movie Tickets,900,DEBIT
"""

    return pd.read_csv(io.StringIO(sample_csv))


# ==========================================
# 7. LOAD CSV FILE
# ==========================================

try:

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
    else:
        st.sidebar.info("Using sample bank statement.")
        df_raw = load_sample_data()

except Exception as e:

    st.error(f"CSV reading error: {e}")
    st.stop()


# ==========================================
# 8. VALIDATE TRANSACTION DATA
# ==========================================

def validate_data(data):

    data = data.copy()

    data.columns = data.columns.str.strip().str.lower()

    required_columns = [
        "date", "description", "amount", "type"
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    data["amount"] = pd.to_numeric(
        data["amount"], errors="coerce"
    )

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    )

    data["description"] = (
        data["description"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    data["type"] = (
        data["type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    valid = (
        data["amount"].notna()
        & data["date"].notna()
        & data["description"].ne("")
        & data["type"].isin(["DEBIT", "CREDIT"])
    )

    invalid_count = int((~valid).sum())

    data = data.loc[valid].copy()
    data["amount"] = data["amount"].abs()

    if data.empty:
        raise ValueError("No valid transactions found.")

    return data, invalid_count


try:

    df_raw, invalid_count = validate_data(df_raw)

    if invalid_count > 0:
        st.warning(
            f"{invalid_count} invalid transactions skipped."
        )

except ValueError as e:

    st.error(str(e))
    st.stop()


# ==========================================
# 9. TRANSACTION CATEGORIZATION
# ==========================================

VALID_CATEGORIES = [
    "Groceries",
    "Dining",
    "Utilities",
    "Entertainment",
    "Transport",
    "Shopping",
    "Income",
    "Miscellaneous"
]


def fallback_category(description, transaction_type):

    if transaction_type == "CREDIT":
        return "Income"

    text = description.lower()

    if any(word in text for word in [
        "zomato", "swiggy", "restaurant",
        "food", "cafe", "starbucks"
    ]):
        return "Dining"

    if any(word in text for word in [
        "grocery", "supermarket", "blinkit",
        "vegetables", "groceries"
    ]):
        return "Groceries"

    if any(word in text for word in [
        "electricity", "water", "gas",
        "rent", "utility"
    ]):
        return "Utilities"

    if any(word in text for word in [
        "netflix", "movie", "pvr",
        "cinema", "entertainment"
    ]):
        return "Entertainment"

    if any(word in text for word in [
        "uber", "ola", "bus", "taxi",
        "train", "metro"
    ]):
        return "Transport"

    if any(word in text for word in [
        "amazon", "shopping", "flipkart"
    ]):
        return "Shopping"

    return "Miscellaneous"


def extract_json(text):

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text
    ).strip()

    match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)

    if not match:
        raise ValueError("No valid JSON found in model response.")

    return json.loads(match.group())


def categorize_transactions(data):

    descriptions = data["description"].tolist()
    types = data["type"].tolist()

    prompt = PromptTemplate(
        template="""
You are a bank transaction categorization agent.

Available categories:
Groceries, Dining, Utilities, Entertainment,
Transport, Shopping, Income, Miscellaneous.

Examples:
Zomato Food Delivery = Dining
Supermarket = Groceries
Salary Credit = Income
Electricity Bill = Utilities
Uber Ride = Transport

Transactions:
{transactions}

Return ONLY a JSON array containing one category
for each transaction, in the same order.

Example:
["Dining", "Groceries", "Income"]
""",
        input_variables=["transactions"]
    )

    chain = prompt | llm | StrOutputParser()

    transaction_list = [
        {
            "description": descriptions[i],
            "type": types[i]
        }
        for i in range(len(descriptions))
    ]

    try:

        response = chain.invoke({
            "transactions": json.dumps(transaction_list)
        })

        categories = extract_json(response)

        if (
            not isinstance(categories, list)
            or len(categories) != len(descriptions)
        ):
            raise ValueError("Invalid category response.")

        final_categories = []

        for i, category in enumerate(categories):

            category = str(category).strip().title()

            if category in VALID_CATEGORIES:
                final_categories.append(category)
            else:
                final_categories.append(
                    fallback_category(
                        descriptions[i], types[i]
                    )
                )

        return final_categories

    except Exception as e:

        st.warning(
            f"AI categorization used rule-based fallback: {e}"
        )

        return [
            fallback_category(descriptions[i], types[i])
            for i in range(len(descriptions))
        ]


# ==========================================
# 10. PROCESS TRANSACTIONS
# ==========================================

st.subheader("🔄 Transaction Processing")

with st.spinner("Qwen AI is categorizing transactions..."):

    df = df_raw.copy()
    df["category"] = categorize_transactions(df)

st.success("Transaction processing completed!")


# ==========================================
# 11. FINANCIAL CALCULATIONS
# ==========================================

debit_df = df[df["type"] == "DEBIT"].copy()
credit_df = df[df["type"] == "CREDIT"].copy()

total_expenses = debit_df["amount"].sum()
total_income = credit_df["amount"].sum()

net_savings = total_income - total_expenses

savings_percentage = (
    net_savings / total_income * 100
    if total_income > 0 else 0
)

cat_totals = (
    debit_df.groupby("category")["amount"]
    .sum()
    .sort_values(ascending=False)
)


# ==========================================
# 12. FINANCIAL DASHBOARD
# ==========================================

st.divider()
st.header("📊 Financial Overview")

col1, col2, col3 = st.columns(3)

col1.metric("💵 Total Income", f"₹{total_income:,.2f}")
col2.metric("💸 Total Expenses", f"₹{total_expenses:,.2f}")
col3.metric(
    "💰 Net Savings",
    f"₹{net_savings:,.2f}",
    delta=f"{savings_percentage:.1f}% of income"
)


# ==========================================
# 13. TRANSACTIONS AND PIE CHART
# ==========================================

left, right = st.columns(2)

with left:

    st.subheader("📄 Processed Transactions")

    display_df = df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


with right:

    st.subheader("🍩 Expense Distribution")

    if not cat_totals.empty:

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.pie(
            cat_totals.values,
            labels=cat_totals.index,
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops={
                "width": 0.4,
                "edgecolor": "white"
            }
        )

        ax.set_title("Category-wise Expenses")
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("No debit transactions available.")


# ==========================================
# 14. SPENDING INSIGHTS
# ==========================================

st.divider()
st.header("🏆 Spending Insights")

if not cat_totals.empty:

    top_category = cat_totals.idxmax()
    top_amount = cat_totals.max()

    percentage = (
        top_amount / total_expenses * 100
        if total_expenses > 0 else 0
    )

    c1, c2 = st.columns(2)

    c1.metric("Highest Spending Category", top_category)

    c2.metric(
        "Category Expense",
        f"₹{top_amount:,.2f}",
        delta=f"{percentage:.1f}% of expenses"
    )


if not debit_df.empty:

    highest_transaction = debit_df.loc[
        debit_df["amount"].idxmax()
    ]

    st.write(
        f"**Highest Individual Expense:** "
        f"{highest_transaction['description']} "
        f"(₹{highest_transaction['amount']:,.2f})"
    )


# ==========================================
# 15. INTENT ROUTER AGENT
# ==========================================

router_prompt = PromptTemplate(
    template="""
You are an Intent Router Agent.

ANALYSIS:
Questions asking for financial calculations,
income, expenses, savings, transaction details
or category breakdown.

ADVISOR:
Questions asking for suggestions, saving tips,
budget planning, overspending or recommendations.

Examples:
"What is my total income?" = ANALYSIS
"Which category has the highest spending?" = ANALYSIS
"How can I save more money?" = ADVISOR
"How can I reduce food expenses?" = ADVISOR

User Question:
{question}

Return ONLY JSON:
{{
    "target_agent": "ANALYSIS",
    "reason": "Short explanation"
}}
""",
    input_variables=["question"]
)

router_chain = router_prompt | llm | StrOutputParser()


# ==========================================
# 16. AGENTIC RAG DECISION AGENT
# ==========================================

rag_decision_prompt = PromptTemplate(
    template="""
You are an Agentic RAG decision agent.

Decide if the question needs financial guidelines
from the vector database.

RETRIEVE:
Use when budgeting rules, spending limits,
saving principles or financial guidelines are useful.

DIRECT:
Use when stored guidelines are not required.

Question:
{question}

Return ONLY JSON:
{{
    "action": "RETRIEVE",
    "reason": "Short explanation"
}}
""",
    input_variables=["question"]
)

rag_decision_chain = (
    rag_decision_prompt | llm | StrOutputParser()
)


# ==========================================
# 17. FINANCIAL ADVISOR AGENT
# ==========================================

advisor_prompt = PromptTemplate(
    template="""
You are an AI Personal Finance Advisor.

Use the user's actual financial information.
Answer the exact question asked.
Do not repeat a generic answer for different questions.
Do not invent figures or transactions.
Use simple English.

FINANCIAL SUMMARY:
{spending_summary}

RETRIEVED GUIDELINES:
{rag_context}

USER QUESTION:
{question}

Give exactly 3 recommendations when the user asks for advice.
Make each recommendation specific to the question.
Use actual figures where relevant.
""",
    input_variables=[
        "spending_summary",
        "rag_context",
        "question"
    ]
)

advisor_chain = advisor_prompt | llm | StrOutputParser()


# ==========================================
# 18. QUESTION-SPECIFIC ANALYSIS
# ==========================================

def analyze_question(question):

    q = question.lower()

    # Individual transaction question
    if any(word in q for word in [
        "biggest transaction",
        "largest transaction",
        "highest individual expense",
        "most expensive transaction"
    ]):

        if debit_df.empty:
            return "No expense transactions found."

        transaction = debit_df.loc[
            debit_df["amount"].idxmax()
        ]

        return (
            "### 💳 Biggest Transaction\n\n"
            f"- **Description:** {transaction['description']}\n"
            f"- **Category:** {transaction['category']}\n"
            f"- **Amount:** ₹{transaction['amount']:,.2f}"
        )

    # Highest spending category
    if any(word in q for word in [
        "highest spending category",
        "highest expense category",
        "top spending category",
        "most spending category",
        "which category"
    ]):

        if cat_totals.empty:
            return "No expense data is available."

        category = cat_totals.idxmax()
        amount = cat_totals.max()

        percentage = (
            amount / total_expenses * 100
            if total_expenses > 0 else 0
        )

        return (
            "### 🏆 Highest Spending Category\n\n"
            f"- **Category:** {category}\n"
            f"- **Amount:** ₹{amount:,.2f}\n"
            f"- **Percentage of expenses:** {percentage:.2f}%"
        )

    # Category-specific expense
    category_keywords = {
        "food": "Dining",
        "dining": "Dining",
        "restaurant": "Dining",
        "grocery": "Groceries",
        "groceries": "Groceries",
        "shopping": "Shopping",
        "transport": "Transport",
        "travel": "Transport",
        "entertainment": "Entertainment",
        "subscription": "Entertainment",
        "utilities": "Utilities",
        "electricity": "Utilities"
    }

    for keyword, category in category_keywords.items():

        if keyword in q and any(word in q for word in [
            "expense", "spend", "spent", "amount", "how much"
        ]):

            amount = cat_totals.get(category, 0.0)

            return (
                f"### {category} Expense Analysis\n\n"
                f"- **Category:** {category}\n"
                f"- **Total amount:** ₹{amount:,.2f}"
            )

    # Savings percentage
    if "percentage" in q and "saving" in q:

        return (
            "### 📈 Savings Percentage\n\n"
            f"Your savings percentage is "
            f"**{savings_percentage:.2f}%**."
        )

    # Income
    if any(word in q for word in [
        "income", "salary", "earned", "earnings"
    ]):

        return (
            "### 💵 Income Analysis\n\n"
            f"Your total income is **₹{total_income:,.2f}**."
        )

    # Expenses
    if any(word in q for word in [
        "total expense", "total spending",
        "how much did i spend", "spent in total"
    ]):

        return (
            "### 💸 Expense Analysis\n\n"
            f"Your total expenses are **₹{total_expenses:,.2f}**."
        )

    # Savings
    if any(word in q for word in [
        "saving", "savings", "remaining balance",
        "net balance"
    ]):

        return (
            "### 💰 Savings Analysis\n\n"
            f"- **Net Savings:** ₹{net_savings:,.2f}\n"
            f"- **Savings Percentage:** {savings_percentage:.2f}%"
        )

    # Complete financial summary
    return (
        "### 📊 Complete Financial Summary\n\n"
        f"- **Total Income:** ₹{total_income:,.2f}\n"
        f"- **Total Expenses:** ₹{total_expenses:,.2f}\n"
        f"- **Net Savings:** ₹{net_savings:,.2f}\n"
        f"- **Savings Percentage:** {savings_percentage:.2f}%"
    )


# ==========================================
# 19. FINANCE AGENT CHAT INTERFACE
# ==========================================

st.divider()
st.header("🤖 Ask Your Finance Agents")

user_query = st.text_input(
    "Ask a question about your finances:",
    placeholder="Example: How can I reduce my food expenses?"
)

if st.button("Analyze My Question", type="primary"):

    if not user_query.strip():

        st.warning("Please enter a question.")

    else:

        try:

            # Detect numerical questions directly
            q = user_query.lower()

            advice_words = [
                "how can i", "how do i", "suggest",
                "recommend", "advice", "tips",
                "improve", "reduce", "manage",
                "budget", "should i"
            ]

            is_advice_question = any(
                word in q for word in advice_words
            )

            # Ask the router when it is not clearly advice
            if is_advice_question:

                selected_agent = "ADVISOR"
                router_result = {
                    "target_agent": "ADVISOR",
                    "reason": "Question requests financial advice."
                }

            else:

                with st.spinner("Intent Router analyzing..."):

                    router_response = router_chain.invoke({
                        "question": user_query
                    })

                    router_result = extract_json(router_response)

                    selected_agent = str(
                        router_result.get(
                            "target_agent", "ANALYSIS"
                        )
                    ).upper()

                    if selected_agent not in [
                        "ANALYSIS", "ADVISOR"
                    ]:
                        selected_agent = "ANALYSIS"

            if show_debug:

                with st.expander("🛠️ Router Agent Debug"):

                    st.json(router_result)
                    st.code(router_prompt.template)

            # ------------------------------
            # ANALYSIS AGENT
            # ------------------------------

            if selected_agent == "ANALYSIS":

                st.success("📊 Analysis Agent Activated")

                answer = analyze_question(user_query)

                st.markdown(answer)

            # ------------------------------
            # ADVISOR AGENT + AGENTIC RAG
            # ------------------------------

            elif selected_agent == "ADVISOR":

                st.success("🧠 Financial Advisor Activated")

                with st.spinner("RAG Agent deciding whether retrieval is needed..."):

                    rag_response = rag_decision_chain.invoke({
                        "question": user_query
                    })

                    rag_decision = extract_json(rag_response)

                    action = str(
                        rag_decision.get("action", "RETRIEVE")
                    ).upper()

                    if action not in ["RETRIEVE", "DIRECT"]:
                        action = "RETRIEVE"

                rag_context = "No additional guidelines retrieved."
                retrieved_docs = []

                if action == "RETRIEVE":

                    with st.spinner("Searching ChromaDB..."):

                        retrieved_docs = vector_db.similarity_search(
                            user_query,
                            k=3
                        )

                        if retrieved_docs:

                            rag_context = "\n".join(
                                [
                                    "- " + doc.page_content
                                    for doc in retrieved_docs
                                ]
                            )

                        else:

                            rag_context = "No matching guidelines found."

                else:

                    st.info(
                        "RAG Agent selected DIRECT response. "
                        "No vector search was required."
                    )

                spending_summary = json.dumps(
                    {
                        "total_income": float(total_income),
                        "total_expenses": float(total_expenses),
                        "net_savings": float(net_savings),
                        "savings_percentage": float(savings_percentage),
                        "category_wise_expenses": {
                            str(k): float(v)
                            for k, v in cat_totals.items()
                        }
                    },
                    indent=2
                )

                with st.spinner("Advisor preparing your personalized answer..."):

                    advice = advisor_chain.invoke({
                        "spending_summary": spending_summary,
                        "rag_context": rag_context,
                        "question": user_query
                    })

                st.subheader("💡 Personalized Financial Advice")
                st.markdown(advice)

                if show_debug:

                    with st.expander("🔍 Agentic RAG Debug"):

                        st.write("RAG Decision:")
                        st.json(rag_decision)

                        st.write("Retrieved Guidelines:")
                        st.write(rag_context)

                        st.write("RAG Prompt:")
                        st.code(rag_decision_prompt.template)

                        st.write("Advisor Prompt:")
                        st.code(advisor_prompt.template)


        except Exception as e:

            st.error(f"Agent execution failed: {e}")


# ==========================================
# 20. PROMPT ENGINEERING DEMONSTRATION
# ==========================================

st.divider()

with st.expander("📝 Prompt Engineering and LangChain Details"):

    st.markdown("""
    **LLM:** Qwen 2.5 (1.5B) using Ollama.

    **Prompt Engineering Techniques:**

    1. Zero-shot prompting – transaction categorization instructions.
    2. Few-shot prompting – example questions for intent routing.
    3. Role-based prompting – Router Agent and Finance Advisor roles.
    4. Structured output prompting – JSON responses.
    5. ReAct-style tool selection – RAG agent decides whether to retrieve guidelines.

    **LangChain + LCEL:**

    PromptTemplate → ChatOllama → StrOutputParser

    **Agentic RAG:**

    User Question → Retrieval Decision → ChromaDB Similarity Search
    → Retrieved Context → Finance Advisor.
    """)


# ==========================================
# 21. DOWNLOAD PROCESSED STATEMENT
# ==========================================

st.divider()
st.subheader("📥 Download Processed Statement")

csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Categorized CSV",
    data=csv_data,
    file_name="processed_finance_statement.csv",
    mime="text/csv"
)


# ==========================================
# 22. FOOTER
# ==========================================

st.divider()

st.caption(
    "💰 Statement-to-Insights Finance Agent | "
    "Powered by Ollama, Qwen 2.5, LangChain, "
    "ChromaDB and Hugging Face"
)
