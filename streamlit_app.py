import os
import time
import json
import re
import streamlit as st
import pandas as pd
from datasets import Dataset as HFDataset

from config import PDF_LOCAL_PATH, PDF_NAME
from answer_gen import answer_query
from memory import save_message, load_history, new_session_id
from retriever import retrieve

st.set_page_config(page_title="RAG Chatbot — Philips Ingenia R11.1", layout="wide")
st.title("RAG Chatbot — Philips Ingenia R11.1 (Technical Description)")

tab_chat, tab_eval = st.tabs(["Chat", "Evaluation"])

# ---------------- CHAT ----------------
with tab_chat:
    if "session_id" not in st.session_state:
        st.session_state.session_id = new_session_id()
    if "last_citations" not in st.session_state:
        st.session_state.last_citations = []
    if "last_model_used" not in st.session_state:
        st.session_state.last_model_used = None

    sid = st.session_state.session_id
    st.caption(f"Session ID: {sid} (each user has isolated memory)")

    st.markdown("---")
    col_a, col_b, _ = st.columns([1, 1, 6])
    with col_a:
        show_ctx = st.checkbox("Show retrieved context", value=True)
    with col_b:
        top_k = st.slider("Top-K docs", min_value=3, max_value=12, value=5, step=1)
    st.markdown("---")

    with st.form("chat_form", clear_on_submit=True):
        query = st.text_input("Ask a question about Ingenia R11.1:")
        submitted = st.form_submit_button("Send")

    if submitted and query.strip():
        save_message(sid, "user", query.strip())
        answer, citations, model_used = answer_query(query.strip(), sid, top_k=top_k)
        st.session_state.last_citations = citations
        st.session_state.last_model_used = model_used
        save_message(sid, "assistant", answer)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Chat")
        history = load_history(sid, limit=200)
        for role, text in history:
            speaker = "You" if role == "user" else "Assistant"
            st.markdown(f"**{speaker}:** {text}")

    with col2:
        st.subheader("Citations")
        if st.session_state.last_model_used:
            st.caption(f"Model used: {st.session_state.last_model_used}")
        if st.session_state.last_citations:
            st.write("Click a page to open the PDF at that page (works if PDF_URL is set in .env).")
            for c in st.session_state.last_citations:
                page = c["page"]
                url = c.get("pdf_url") or ""
                if url:
                    st.markdown(f"- [{PDF_NAME} — Page {page}]({url}#page={page})")
                else:
                    st.markdown(f"- {PDF_NAME} — Page {page}")
        else:
            st.caption("Ask a question to see citations here.")

        st.divider()
        st.subheader("Retrieved context")
        if show_ctx and st.session_state.last_citations:
            for c in st.session_state.last_citations:
                with st.expander(f"Page {c['page']}"):
                    st.write(c["content"][:2000])

    st.sidebar.header("Utilities")
    if st.sidebar.button("Clear conversation"):
        st.session_state.session_id = new_session_id()
        st.session_state.last_citations = []
        st.session_state.last_model_used = None
        st.rerun()
    try:
        with open(PDF_LOCAL_PATH, "rb") as f:
            st.sidebar.download_button(
                "Download PDF",
                data=f.read(),
                file_name=os.path.basename(PDF_LOCAL_PATH),
            )
    except Exception:
        st.sidebar.caption("PDF file not found for download.")

# ---------------- EVALUATION ----------------
with tab_eval:
    st.subheader("Evaluate with RAGAS (synthetic dataset)")
    dataset_path = st.text_input("Path to synthetic dataset (.jsonl):", "data/synth_eval.jsonl")

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        max_q = st.number_input("Max questions to evaluate", min_value=1, max_value=200, value=8, step=1)
    with col_cfg2:
        delay_sec = st.slider("Delay between items (seconds)", min_value=0.0, max_value=5.0, value=0.8, step=0.1)

    col_cfg3, col_cfg4 = st.columns(2)
    with col_cfg3:
        retrieval_only = st.checkbox("Retrieval-only (no answer generation)", value=True)
    with col_cfg4:
        use_groq_judge = st.checkbox("Use Groq judge for full metrics", value=False)

    run_btn = st.button("Run Evaluation")

    if run_btn:
        # Load dataset
        try:
            rows = []
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line))
            if not rows:
                st.error("Dataset is empty.")
                st.stop()
            st.write(f"Loaded {len(rows)} questions. Evaluating up to {max_q}.")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

        total = min(max_q, len(rows))

        # Local retrieval-only evaluator (no LLM, no API keys)
        if retrieval_only:
            import string
            def _norm(t: str):
                t = (t or "").lower()
                t = t.translate(str.maketrans("", "", string.punctuation))
                return [w for w in t.split() if w]

            records, prog = [], st.progress(0.0)
            for i, r in enumerate(rows[:total]):
                q = r["question"]
                gt = r.get("answer", "")
                hits = retrieve(q, top_k=5)
                ctxs = [h["content"] for h in hits]

                ref_tokens = set(_norm(gt))
                union_ctx = set()
                per_ctx_overlap = []
                for c in ctxs:
                    ct = set(_norm(c))
                    union_ctx |= ct
                    inter = len(ref_tokens & ct)
                    per_ctx_overlap.append(inter / max(1, len(ref_tokens)) if ref_tokens else 0.0)

                # Precision: fraction of contexts that overlap a little with reference
                precision = sum(1 for x in per_ctx_overlap if x >= 0.05) / max(1, len(per_ctx_overlap)) if per_ctx_overlap else 0.0
                # Recall: recall of reference tokens in union of contexts
                recall = len(ref_tokens & union_ctx) / max(1, len(ref_tokens)) if ref_tokens else 0.0

                records.append({
                    "question": q,
                    "context_precision": round(precision, 4),
                    "context_recall": round(recall, 4),
                })
                prog.progress((i + 1) / total)
                time.sleep(delay_sec)

            scores_df = pd.DataFrame(records)
            st.success("Evaluation complete (local retrieval metrics).")
            st.dataframe(scores_df)

            numeric = list(scores_df.select_dtypes(include="number").columns)
            avg = {c: round(float(scores_df[c].mean()), 4) for c in numeric}
            st.markdown("### Averages (retrieval metrics)")
            st.json(avg)

            csv = scores_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download scores.csv", data=csv, file_name="retrieval_scores.csv")

        else:
            # Optional RAGAS path with Groq judge (requires ragas + langchain-openai)
            try:
                from ragas import evaluate
                from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
                from ragas.llms import LangchainLLM
                from langchain_openai import ChatOpenAI

                groq_key = os.getenv("GROQ_API_KEY")
                judge_model = os.getenv("GROQ_JUDGE_MODEL", "llama-3.1-8b-instant")

                def call_answer(q: str):
                    last = ("", [])
                    for attempt in range(3):
                        ans, cites, _ = answer_query(q, new_session_id(), top_k=5)
                        if ans and "Generation error" not in ans and "429" not in ans:
                            return ans, cites
                        last = (ans or "", cites or [])
                        time.sleep(max(0.8, delay_sec) * (attempt + 1))
                    return last

                preds, prog = [], st.progress(0.0)
                for i, r in enumerate(rows[:total]):
                    q = r["question"]
                    gt = r.get("answer", "")
                    ans, cites = call_answer(q)
                    ctxs = [c["content"] for c in cites]
                    preds.append({"question": q, "answer": ans or "", "contexts": ctxs, "reference": gt})
                    prog.progress((i + 1) / total)
                    time.sleep(delay_sec)

                df = pd.DataFrame(preds, columns=["question", "answer", "contexts", "reference"])
                usable = df[df["answer"].fillna("").str.len() >= 3].copy()
                st.caption(f"Usable rows for scoring: {len(usable)}/{len(df)}")
                if usable.empty:
                    st.error("All answers were empty. Try again or adjust settings.")
                    st.stop()

                hf_ds = HFDataset.from_pandas(usable)

                metrics = [context_precision, context_recall]
                kwargs = {}
                if use_groq_judge:
                    judge = ChatOpenAI(
                        model=judge_model,
                        api_key=groq_key,
                        base_url="https://api.groq.com/openai/v1",
                        temperature=0.0,
                    )
                    kwargs["llm"] = LangchainLLM(judge)
                    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

                with st.spinner("Scoring with RAGAS..."):
                    scores = evaluate(hf_ds, metrics=metrics, **kwargs)

                scores_df = scores if isinstance(scores, pd.DataFrame) else scores.to_pandas()
                st.success("Evaluation complete.")
                st.dataframe(scores_df)

                numeric = list(scores_df.select_dtypes(include="number").columns)
                avg = {c: round(float(scores_df[c].mean()), 4) for c in numeric}
                st.markdown("### Averages (numeric metrics)")
                st.json(avg)

                csv = scores_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download scores.csv", data=csv, file_name="ragas_scores.csv")

            except Exception as e:
                st.error(f"RAGAS path failed: {e}. Use retrieval-only mode or install ragas + langchain-openai.")
