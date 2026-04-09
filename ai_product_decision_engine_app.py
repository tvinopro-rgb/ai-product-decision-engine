import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


st.set_page_config(page_title="AI Product Decision Engine", layout="wide")


# -----------------------------
# Prompt templates
# -----------------------------
INSIGHT_CLUSTER_PROMPT = """
You are a senior product manager.

Analyze the following customer feedback and:
1. Group it into the most important themes
2. Name each theme clearly
3. Provide for each theme:
   - description
   - number_of_occurrences
   - example_user_quotes
4. Use only evidence grounded in the provided feedback
5. Ensure number_of_occurrences is a realistic count based on the input

Return valid JSON in this shape:
{{
  "themes": [
    {{
      "theme_name": "...",
      "description": "...",
      "number_of_occurrences": 0,
      "example_user_quotes": ["...", "..."]
    }}
  ]
}}

Feedback:
{feedback_data}
""".strip()

OPPORTUNITY_PROMPT = """
You are a product strategy expert.

Based on these customer insight themes:
{themes_output}

Identify the top product opportunities.
Return valid JSON in this shape:
{{
  "opportunities": [
    {{
      "name": "...",
      "problem_statement": "...",
      "business_impact": "...",
      "suggested_solution_direction": "..."
    }}
  ]
}}
""".strip()

PRIORITIZATION_PROMPT = """
You are a product leader.

Rank the following opportunities based on:
- customer impact
- business impact
- estimated effort

Return valid JSON in this shape:
{{
  "prioritized_opportunities": [
    {{
      "name": "...",
      "customer_impact": 1,
      "business_impact": 1,
      "estimated_effort": 1,
      "overall_score": 1,
      "reasoning": "..."
    }}
  ]
}}

Opportunities:
{opportunities}
""".strip()

PRD_PROMPT = """
Create a structured product requirements document for the following feature:
{selected_opportunity}

Return valid JSON in this shape:
{{
  "title": "...",
  "problem_statement": "...",
  "goals": ["..."],
  "user_stories": ["..."],
  "functional_requirements": ["..."],
  "success_metrics": ["..."]
}}
""".strip()

JIRA_PROMPT = """
Convert the following PRD into Jira-ready stories.

Return valid JSON in this shape:
{{
  "epic": "...",
  "stories": [
    {{
      "title": "...",
      "description": "...",
      "acceptance_criteria": ["...", "..."]
    }}
  ]
}}

PRD:
{prd_output}
""".strip()


# -----------------------------
# Sample data
# -----------------------------
def get_sample_data() -> pd.DataFrame:
    data = [
        {"id": 1, "source": "app_review", "feedback": "The onboarding is confusing, I couldn’t understand what to do next"},
        {"id": 2, "source": "support_ticket", "feedback": "Login keeps failing even after resetting password"},
        {"id": 3, "source": "app_review", "feedback": "Too many steps to create an account, I gave up halfway"},
        {"id": 4, "source": "sales_call", "feedback": "Customers are asking for better reporting dashboards"},
        {"id": 5, "source": "app_review", "feedback": "App is slow when loading dashboard"},
        {"id": 6, "source": "support_ticket", "feedback": "I didn’t receive OTP during signup"},
        {"id": 7, "source": "app_review", "feedback": "Navigation is not intuitive"},
        {"id": 8, "source": "sales_call", "feedback": "Enterprise clients want role-based access"},
        {"id": 9, "source": "app_review", "feedback": "Too many bugs in latest update"},
        {"id": 10, "source": "support_ticket", "feedback": "Payment failed but money got deducted"},
    ]
    return pd.DataFrame(data)


# -----------------------------
# Parsing and validation helpers
# -----------------------------
def safe_json_loads(content: str) -> Dict[str, Any]:
    content = content.strip()

    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response")

    return json.loads(content[start : end + 1])


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def guess_feedback_column(columns: List[str]) -> str | None:
    preferred = [
        "feedback",
        "comment",
        "comments",
        "review",
        "reviews",
        "ticket",
        "tickets",
        "issue",
        "issues",
        "voice_of_customer",
        "customer_feedback",
        "text",
        "message",
        "description",
        "summary",
    ]
    lowered = {col.lower(): col for col in columns}
    for item in preferred:
        if item in lowered:
            return lowered[item]
    for col in columns:
        if any(word in col.lower() for word in ["feedback", "review", "comment", "issue", "message", "description"]):
            return col
    return None


def guess_source_column(columns: List[str]) -> str | None:
    for col in columns:
        if any(word in col.lower() for word in ["source", "channel", "type", "origin"]):
            return col
    return None


def prepare_feedback_df(df: pd.DataFrame, feedback_col: str, source_col: str | None) -> pd.DataFrame:
    prepared = pd.DataFrame()
    prepared["feedback"] = df[feedback_col].fillna("").astype(str).str.strip()
    prepared = prepared[prepared["feedback"] != ""].copy()
    if source_col and source_col in df.columns:
        prepared["source"] = df.loc[prepared.index, source_col].fillna("uploaded_csv").astype(str)
    else:
        prepared["source"] = "uploaded_csv"
    prepared.insert(0, "id", range(1, len(prepared) + 1))
    return prepared.reset_index(drop=True)


def dataframe_to_feedback_text(df: pd.DataFrame) -> str:
    return "\n".join(f"[{row['source']}] {row['feedback']}" for _, row in df.iterrows())


# -----------------------------
# Fallback heuristic engine
# -----------------------------
STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "have", "after", "into", "your", "they", "them",
    "their", "there", "about", "when", "what", "where", "which", "while", "would", "could", "should", "been",
    "being", "were", "was", "are", "is", "our", "too", "not", "can", "cant", "couldnt", "didnt", "doesnt",
    "very", "more", "less", "some", "many", "much", "next", "then", "than", "just", "also", "still", "user",
    "users", "customer", "customers", "app", "product", "please", "need", "want", "like", "dont", "did", "got",
    "get", "has", "had", "use", "using", "used", "how", "why", "all", "any", "its", "it", "to", "of", "on",
    "in", "at", "a", "an", "as", "be", "or", "if", "by", "my", "we", "i", "me", "you", "he", "she", "they",
}

THEME_LIBRARY: List[Tuple[str, List[str], str, str, str]] = [
    (
        "Onboarding and signup friction",
        ["onboarding", "signup", "sign up", "register", "registration", "account", "otp", "verify", "verification"],
        "Users face friction while creating accounts or completing initial setup.",
        "Improves activation and conversion rates.",
        "Simplify signup, reduce steps, and improve verification reliability.",
    ),
    (
        "Authentication and access issues",
        ["login", "log in", "password", "authentication", "access", "sign in", "signin", "unable to login"],
        "Users struggle to access the product reliably.",
        "Reduces support volume and improves trust.",
        "Improve authentication recovery, session handling, and error messaging.",
    ),
    (
        "Performance and speed",
        ["slow", "loading", "load", "lag", "latency", "performance", "freeze", "crash"],
        "Users perceive the product as slow or unstable.",
        "Improves retention and task completion.",
        "Reduce latency, optimize heavy screens, and address stability bottlenecks.",
    ),
    (
        "Reporting and analytics gaps",
        ["dashboard", "report", "reporting", "analytics", "insights", "metrics", "export"],
        "Users need better visibility into data, metrics, and reporting.",
        "Improves stickiness and enterprise value.",
        "Expand reporting, improve dashboard speed, and add exportable insights.",
    ),
    (
        "Usability and navigation",
        ["navigation", "confusing", "intuitive", "workflow", "hard", "difficult", "steps", "find"],
        "Users struggle to understand the interface or complete tasks smoothly.",
        "Improves activation, productivity, and overall satisfaction.",
        "Simplify key journeys, improve labels, and add in-product guidance.",
    ),
    (
        "Payments and billing issues",
        ["payment", "billing", "charged", "deducted", "invoice", "refund", "subscription"],
        "Users encounter payment, billing, or subscription problems.",
        "Protects revenue and reduces escalations.",
        "Improve transaction reliability, reconciliation, and billing transparency.",
    ),
    (
        "Permissions and enterprise controls",
        ["role", "access control", "permission", "admin", "enterprise", "rbac"],
        "Larger customers need stronger permissions and administrative controls.",
        "Supports expansion into enterprise accounts.",
        "Add role-based access controls and admin workflows.",
    ),
    (
        "Bugs and reliability",
        ["bug", "bugs", "error", "issue", "broken", "failed", "failure"],
        "Users report broken flows or inconsistent behavior.",
        "Improves trust and lowers support burden.",
        "Tighten release quality gates and prioritize top recurring defects.",
    ),
]


def extract_keywords(texts: List[str], top_n: int = 12) -> List[str]:
    tokens: List[str] = []
    for text in texts:
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        tokens.extend(word for word in words if word not in STOPWORDS)
    return [word for word, _ in Counter(tokens).most_common(top_n)]


def heuristic_themes(df: pd.DataFrame) -> Dict[str, Any]:
    feedbacks = df["feedback"].tolist()
    lowered_feedbacks = [item.lower() for item in feedbacks]
    themes: List[Dict[str, Any]] = []
    used_indexes: set[int] = set()

    for theme_name, keywords, description, _, _ in THEME_LIBRARY:
        matched_indexes = []
        for idx, fb in enumerate(lowered_feedbacks):
            if any(keyword in fb for keyword in keywords):
                matched_indexes.append(idx)
        if matched_indexes:
            used_indexes.update(matched_indexes)
            themes.append(
                {
                    "theme_name": theme_name,
                    "description": description,
                    "number_of_occurrences": len(matched_indexes),
                    "example_user_quotes": [feedbacks[idx] for idx in matched_indexes[:3]],
                    "matched_keywords": keywords[:4],
                }
            )

    unmatched = [feedbacks[idx] for idx in range(len(feedbacks)) if idx not in used_indexes]
    if unmatched:
        top_keywords = extract_keywords(unmatched)
        if top_keywords:
            themes.append(
                {
                    "theme_name": "Other emerging feedback themes",
                    "description": f"Additional feedback patterns appear around: {', '.join(top_keywords[:5])}.",
                    "number_of_occurrences": len(unmatched),
                    "example_user_quotes": unmatched[:3],
                    "matched_keywords": top_keywords[:5],
                }
            )

    if not themes:
        themes.append(
            {
                "theme_name": "General customer feedback",
                "description": "Uploaded feedback did not match predefined patterns, so the app grouped it into a general feedback bucket.",
                "number_of_occurrences": len(feedbacks),
                "example_user_quotes": feedbacks[:3],
                "matched_keywords": extract_keywords(feedbacks)[:5],
            }
        )

    return {"themes": themes}


def heuristic_opportunities(themes_output: Dict[str, Any]) -> Dict[str, Any]:
    opportunities = []
    for theme in themes_output.get("themes", [])[:4]:
        theme_name = theme["theme_name"]
        matched = next((item for item in THEME_LIBRARY if item[0] == theme_name), None)
        if matched:
            _, _, _, business_impact, solution_direction = matched
        else:
            business_impact = "Improves customer experience and reduces avoidable friction."
            solution_direction = "Investigate the pattern, validate severity, and define the smallest effective improvement."

        opportunities.append(
            {
                "name": f"Address {theme_name.lower()}",
                "problem_statement": theme["description"],
                "business_impact": business_impact,
                "suggested_solution_direction": solution_direction,
            }
        )

    return {"opportunities": opportunities}


def heuristic_prioritization(opportunities: Dict[str, Any], themes_output: Dict[str, Any]) -> Dict[str, Any]:
    theme_lookup = {
        f"Address {theme['theme_name'].lower()}": theme["number_of_occurrences"]
        for theme in themes_output.get("themes", [])
    }
    prioritized = []
    for opp in opportunities.get("opportunities", []):
        occurrences = theme_lookup.get(opp["name"], 1)
        customer_impact = min(10, max(4, occurrences + 4))
        business_impact = min(10, customer_impact + 1)
        effort = 6 if "enterprise" not in opp["name"].lower() else 8
        overall = round((customer_impact * 0.4) + (business_impact * 0.4) + ((11 - effort) * 0.2))
        prioritized.append(
            {
                "name": opp["name"],
                "customer_impact": customer_impact,
                "business_impact": business_impact,
                "estimated_effort": effort,
                "overall_score": overall,
                "reasoning": f"Prioritized from {occurrences} matching feedback items, expected business relevance, and estimated implementation effort.",
            }
        )

    prioritized.sort(key=lambda item: item["overall_score"], reverse=True)
    return {"prioritized_opportunities": prioritized}


def heuristic_prd(selected_name: str, top_theme: Dict[str, Any]) -> Dict[str, Any]:
    theme_desc = top_theme.get("description", "This theme represents a recurring customer problem.")
    examples = top_theme.get("example_user_quotes", [])
    success_signal = top_theme.get("matched_keywords", ["adoption"])

    return {
        "title": selected_name,
        "problem_statement": theme_desc,
        "goals": [
            "Reduce customer friction in the targeted workflow",
            "Improve user satisfaction on the identified pain point",
            "Create measurable improvement on the most common feedback trend",
        ],
        "user_stories": [
            f"As a user, I want improvements related to {top_theme.get('theme_name', 'this workflow').lower()} so that I can complete my task more easily.",
            "As a product manager, I want measurable before-and-after analytics so that I can validate whether the change improved the experience.",
        ],
        "functional_requirements": [
            "Address the core friction reflected in the uploaded feedback",
            "Add instrumentation to measure baseline and post-launch improvement",
            "Provide clear UX guidance and failure handling where relevant",
            "Ensure the updated flow is testable and observable in production",
        ],
        "success_metrics": [
            f"Reduction in complaints related to {', '.join(success_signal[:2])}",
            "Improvement in completion or success rate for the target flow",
            "Reduction in support tickets for the identified issue",
        ],
        "evidence_quotes": examples[:3],
    }


def heuristic_jira(prd_output: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "epic": f"Epic: {prd_output['title']}",
        "stories": [
            {
                "title": "Define target workflow and baseline metrics",
                "description": "Document the current user flow, pain points, and baseline analytics for the prioritized problem.",
                "acceptance_criteria": [
                    "Current-state workflow is documented",
                    "Baseline metrics are available before solution rollout",
                ],
            },
            {
                "title": "Implement prioritized experience improvements",
                "description": "Build the smallest high-impact improvements that address the top recurring customer problem.",
                "acceptance_criteria": [
                    "Solution addresses the primary friction identified in feedback",
                    "Error handling and edge cases are covered",
                ],
            },
            {
                "title": "Instrument and monitor outcome metrics",
                "description": "Add analytics and reporting to validate whether the release improved the customer experience.",
                "acceptance_criteria": [
                    "Tracking exists for completion, failure, and drop-off signals",
                    "A dashboard or report is available for PM review",
                ],
            },
        ],
    }


# -----------------------------
# Model calls
# -----------------------------
def call_llm(prompt: str, model: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key or OpenAI is None:
        raise RuntimeError("OpenAI client or API key not available")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(model=model, input=prompt)
    return safe_json_loads(response.output_text)


def run_pipeline(df: pd.DataFrame, model: str, use_llm: bool) -> Dict[str, Any]:
    feedback_text = dataframe_to_feedback_text(df)

    if use_llm:
        try:
            themes = call_llm(INSIGHT_CLUSTER_PROMPT.format(feedback_data=feedback_text), model)
            opportunities = call_llm(OPPORTUNITY_PROMPT.format(themes_output=json.dumps(themes, indent=2)), model)
            prioritized = call_llm(PRIORITIZATION_PROMPT.format(opportunities=json.dumps(opportunities, indent=2)), model)
            top_name = prioritized["prioritized_opportunities"][0]["name"]
            prd = call_llm(PRD_PROMPT.format(selected_opportunity=top_name), model)
            jira = call_llm(JIRA_PROMPT.format(prd_output=json.dumps(prd, indent=2)), model)
            return {
                "themes": themes,
                "opportunities": opportunities,
                "prioritized": prioritized,
                "prd": prd,
                "jira": jira,
            }
        except Exception as exc:
            st.warning(f"Live model failed, so the app switched to dynamic fallback mode: {exc}")

    themes = heuristic_themes(df)
    opportunities = heuristic_opportunities(themes)
    prioritized = heuristic_prioritization(opportunities, themes)
    top_name = prioritized["prioritized_opportunities"][0]["name"] if prioritized["prioritized_opportunities"] else "Top product opportunity"
    top_theme = themes["themes"][0] if themes["themes"] else {"theme_name": "General feedback", "description": "Recurring customer feedback pattern."}
    prd = heuristic_prd(top_name, top_theme)
    jira = heuristic_jira(prd)

    return {
        "themes": themes,
        "opportunities": opportunities,
        "prioritized": prioritized,
        "prd": prd,
        "jira": jira,
    }


# -----------------------------
# UI
# -----------------------------
st.title("AI Product Decision Engine")
st.caption("Upload a CSV of customer feedback and turn it into themes, prioritized opportunities, a PRD, and Jira-ready stories.")

with st.sidebar:
    st.header("Settings")
    use_sample = st.checkbox("Use sample dataset", value=False)
    use_llm = st.toggle("Use live LLM", value=False)
    model = st.text_input("Model", value="gpt-5")
    st.markdown("---")
    st.markdown("**What this app expects**")
    st.write("Upload a CSV with at least one text column containing customer feedback.")

uploaded_file = st.file_uploader("Upload feedback CSV", type=["csv"])

raw_df = get_sample_data() if use_sample else None
if not use_sample and uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

if raw_df is not None:
    raw_df = normalize_columns(raw_df)
    guessed_feedback_col = guess_feedback_column(raw_df.columns.tolist())
    guessed_source_col = guess_source_column(raw_df.columns.tolist())

    st.subheader("Map your CSV columns")
    map_col1, map_col2 = st.columns(2)
    with map_col1:
        feedback_col = st.selectbox(
            "Feedback text column",
            options=raw_df.columns.tolist(),
            index=raw_df.columns.tolist().index(guessed_feedback_col) if guessed_feedback_col else 0,
        )
    with map_col2:
        source_options = ["None"] + raw_df.columns.tolist()
        source_default = source_options.index(guessed_source_col) if guessed_source_col else 0
        source_choice = st.selectbox("Source column (optional)", options=source_options, index=source_default)
        source_col = None if source_choice == "None" else source_choice

    df = prepare_feedback_df(raw_df, feedback_col, source_col)
else:
    raw_df = pd.DataFrame()
    df = pd.DataFrame(columns=["id", "source", "feedback"])

col1, col2 = st.columns([1.25, 1])

with col1:
    st.subheader("Prepared input data")
    if df.empty:
        st.info("Upload a CSV to begin. The app will let you map the feedback text column and generate outputs from that file.")
    else:
        st.dataframe(df, use_container_width=True, height=320)

with col2:
    st.subheader("How it works")
    st.markdown(
        """
        1. Upload a CSV
        2. Select the column that contains customer feedback
        3. Run the engine
        4. Review themes, opportunities, PRD, and Jira stories

        The app works in two modes:
        - **Live LLM mode** when an API key is available
        - **Dynamic fallback mode** that still generates outputs from your uploaded CSV
        """
    )
    if not df.empty:
        st.metric("Feedback rows", len(df))
        st.metric("Unique sources", df["source"].nunique())

run_clicked = st.button("Generate outputs", type="primary", disabled=df.empty)

if run_clicked:
    results = run_pipeline(df, model=model, use_llm=use_llm)

    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Themes", "Opportunities", "Prioritization", "PRD", "Jira Stories"])

    with tab1:
        st.subheader("Customer feedback themes")
        for theme in results["themes"].get("themes", []):
            with st.container(border=True):
                st.markdown(f"### {theme['theme_name']}")
                st.write(theme["description"])
                st.metric("Occurrences", theme["number_of_occurrences"])
                st.markdown("**Example quotes**")
                for quote in theme.get("example_user_quotes", []):
                    st.write(f"- {quote}")

    with tab2:
        st.subheader("Product opportunities")
        for opp in results["opportunities"].get("opportunities", []):
            with st.container(border=True):
                st.markdown(f"### {opp['name']}")
                st.write(f"**Problem statement:** {opp['problem_statement']}")
                st.write(f"**Business impact:** {opp['business_impact']}")
                st.write(f"**Suggested solution direction:** {opp['suggested_solution_direction']}")

    with tab3:
        st.subheader("Prioritized opportunities")
        ranked = results["prioritized"].get("prioritized_opportunities", [])
        if ranked:
            st.dataframe(pd.DataFrame(ranked), use_container_width=True)

    with tab4:
        prd = results["prd"]
        st.subheader(prd["title"])
        st.write(f"**Problem statement:** {prd['problem_statement']}")
        st.write("**Goals**")
        for item in prd.get("goals", []):
            st.write(f"- {item}")
        st.write("**User stories**")
        for item in prd.get("user_stories", []):
            st.write(f"- {item}")
        st.write("**Functional requirements**")
        for item in prd.get("functional_requirements", []):
            st.write(f"- {item}")
        st.write("**Success metrics**")
        for item in prd.get("success_metrics", []):
            st.write(f"- {item}")
        if prd.get("evidence_quotes"):
            st.write("**Evidence from uploaded feedback**")
            for quote in prd["evidence_quotes"]:
                st.write(f"- {quote}")

    with tab5:
        jira = results["jira"]
        st.subheader(jira["epic"])
        for story in jira.get("stories", []):
            with st.container(border=True):
                st.markdown(f"### {story['title']}")
                st.write(story["description"])
                st.markdown("**Acceptance criteria**")
                for ac in story.get("acceptance_criteria", []):
                    st.write(f"- {ac}")

    with st.expander("Raw JSON outputs"):
        st.json(results)

st.markdown("---")
st.markdown("### Suggested next improvements")
st.markdown(
    """
1. Add export buttons for PRD and Jira stories.
2. Add weighting sliders for customer impact, revenue impact, and effort.
3. Add evidence highlighting back to the original feedback rows.
4. Connect real systems like Zendesk, app reviews, or support tickets.
    """
)
