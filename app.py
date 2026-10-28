
import config  # noqa: F401  (must stay first: keeps all downloads inside ./hf_cache)

import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

from src import demo_plots as dp

st.set_page_config(page_title="RAG Quality-Latency Lab", layout="wide")


def _stretch_kwargs() -> dict:
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
    except ValueError:
        return {"use_container_width": True}
    return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}


STRETCH = _stretch_kwargs()

TIER_LABEL = {"Small": "Small", "Medium": "Medium", "Large": "Large"}
RETRIEVERS = {"dense": "Dense FAISS", "bm25": "BM25", "hybrid_rrf": "Hybrid-RRF"}
MODE_FIXED, MODE_ADAPT, MODE_COMPARE = "Fixed Configuration", "Adaptive Pipeline (Auto-Router)", "Side-by-Side Live Compare"
LIVE_NOTE = ("")


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def ms(x):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:,.0f} ms"


# Session state
st.session_state.setdefault("session_id", uuid.uuid4().hex[:8])
st.session_state.setdefault("query_log", [])
st.session_state.setdefault("last_run", None)
st.session_state.setdefault("mini", None)
st.session_state.setdefault("hw_actual", None)


# Heavy resources: created lazily on the first run, then cached for the server's lifetime
@st.cache_resource(show_spinner=False)
def get_hw():
    from src import cpu_generators as cg

    hw = cg.detect_hardware()
    cg.configure_threads()
    return hw


@st.cache_resource(show_spinner=False)
def get_models():
    from src import cpu_generators as cg

    return cg.ModelCache(get_hw())


@st.cache_resource(show_spinner=False)
def get_embedder():
    from src.retrievers import Embedder

    return Embedder(config.EMBED_MODEL, get_hw().embedder_device)


@st.cache_resource(show_spinner=False)
def get_resources(dataset: str):
    from src.demo_pipeline import DatasetResources

    res = DatasetResources(dataset, get_embedder())
    from src.demo_pipeline import laptop_calibration

    cal, source = laptop_calibration(res)
    return res, cal, source


@st.cache_data(show_spinner=False)
def test_queries(dataset: str):
    from src.data_corpus import load_queries

    return load_queries(dataset, "test")


@st.cache_data(show_spinner=False)
def tables_cached(signature):
    return dp.load_tables()


def tables_signature():
    d = dp.TABLES_DIR
    return tuple(sorted((p.name, p.stat().st_mtime) for p in d.glob("*"))) if d.exists() else ()


def build_pipeline(dataset: str, tier_set: str, max_new: int, budget: int, threads: int):
    from src import cpu_generators as cg
    from src.demo_pipeline import DemoPipeline, Settings

    cg.configure_threads(threads)
    res, cal, source = get_resources(dataset)
    return DemoPipeline(res, get_models(), Settings(tier_set, max_new, budget), cal, source)


# Sidebar
quick = dp.quick_hardware()
hw_info = st.session_state.hw_actual or quick

with st.sidebar:
    st.title(" RAG Quality-Latency Lab")
    dataset = st.selectbox("Dataset", list(dp.DOMAIN_LABEL), format_func=lambda d: dp.DOMAIN_LABEL[d])
    st.markdown(f"**Execution engine:** `{hw_info['badge']}`")
    st.caption(f"{hw_info['ram_gb']:.0f} GB RAM · {hw_info['cores']} CPU cores")

    from src.tier_sets import DEFAULT_MAX_NEW_TOKENS_CPU, DEFAULT_PROMPT_BUDGET, FULL_TIER_MODELS, LAPTOP_TIER_MODELS, TIER_SET_LABELS

    can_full = hw_info["device"] == "cuda" or hw_info["ram_gb"] >= 24
    options = ["laptop", "full"] if can_full else ["laptop"]
    default_idx = options.index("full") if (hw_info["device"] == "cuda" and "full" in options) else 0
    tier_set = st.radio("Generator tier set", options, index=default_idx, format_func=lambda k: TIER_SET_LABELS[k])
    if not can_full:
        st.caption("The full Qwen2.5 1.5B/3B/7B tier set needs a GPU or ≥ 24 GB RAM, so it is hidden on this machine.")
    max_new = st.slider("Max output tokens", 48, 160, DEFAULT_MAX_NEW_TOKENS_CPU, step=8,
                        help="Shorter answers finish faster. Answers that hit the limit are cut off.")
    with st.expander("Advanced (optional)"):
        budget = st.slider("Prompt budget (tokens shared by the k passages)", 512, 2048, DEFAULT_PROMPT_BUDGET, step=128,
                           help="Retrieved passages are trimmed evenly to fit. A smaller budget makes prefill much faster on a laptop.")
        threads = st.slider("CPU threads", 1, max(2, hw_info["cores"] * 2), min(hw_info["cores"], 8))
        st.caption("")
    st.divider()
    stt = dp.index_status(dataset) if dp.prep_ready(dataset) else None
    if stt is None:
        st.error("Prepared data for this dataset was not found in `results/prep/`. Copy that folder from the project, "
                 "or run `python download_laptop_models.py`.")
    elif not stt["dense"]:
        st.warning("The dense index is not cached in `results/indices/`. The first run will build it once "
                   "(a few minutes on CPU). `python download_laptop_models.py` does this up front.")
    else:
        st.success("Index and corpus found: ready to run.")

st.title("Quality–Latency Trade-offs in RAG over Technical Documentation")
tab1, tab2, tab3, tab4 = st.tabs(["① Live QA Prototype", "② Live Mini-Benchmark", "③ Full Benchmark Results", "④ Deployment Guidelines & Theory"])


# Rendering helpers
def kpi_cards(row: dict):
    a = st.columns(5)
    a[0].metric("Total latency", ms(row["total_latency_ms"]))
    a[1].metric("Retrieval", ms(row["retrieval_latency_ms"]))
    a[2].metric("TTFT (prefill)", ms(row["ttft_ms"]), help="Time to first token: the forward pass over the whole prompt.")
    a[3].metric("Decode", ms(row["decode_latency_ms"]), help="Token-by-token generation after the first token.")
    a[4].metric("Output tokens", f"{row['output_tokens']}")
    b = st.columns(5)
    b[0].metric("Composite quality", fmt(row["composite_quality"]), help="0.35 evidence F1 + 0.25 ROUGE-L + 0.25 semantic similarity + 0.15 faithfulness")
    b[1].metric("Evidence token F1", fmt(row["evidence_token_f1"]))
    b[2].metric("ROUGE-L", fmt(row["rouge_l_f1"]))
    b[3].metric("Semantic similarity", fmt(row["semantic_similarity"]))
    b[4].metric("Context faithfulness", fmt(row["context_faithfulness"]), help="How well each answer sentence is grounded in the retrieved passages.")
    if row["composite_quality"] != row["composite_quality"]:
        st.caption("Custom question: there is no reference answer, so only context faithfulness can be scored.")


def router_card(row: dict, calibration: dict, source: str):
    st.markdown("##### Router decision")
    c = st.columns(6)
    c[0].metric("Top-1 score", fmt(row["top1_score"]))
    c[1].metric("Score margin", fmt(row["score_margin"], 4))
    c[2].metric("Query complexity", fmt(row["query_complexity"]))
    c[3].metric("Difficulty", fmt(row["difficulty"], 2), help="Weighted z-score of the four features; higher means harder.")
    c[4].metric("Selected tier", f"{row['router_tier'].title()} · {row['config_id']}")
    c[5].metric("Router overhead", f"{row['router_latency_ms']:.3f} ms")
    tiers = " · ".join(f"{n}: `{cfg}`" for n, cfg in [(t["name"].title(), t["config_id"]) for t in calibration["tiers"]])
    st.caption(f"Tiers in use: {tiers}. Source: {source}.")


def passages_expander(row: dict):
    with st.expander(f"Top-{row['k']} retrieved passages with relevance scores"):
        for p in row["passages"]:
            tag = " · ✅ annotated relevant" if p["relevant"] else ""
            st.markdown(f"**#{p['rank']}** · doc `{p['doc_id'][:8]}` · score **{p['score']:.4f}**{tag}")
            txt = p["text"]
            st.text(txt[:1200] + (" …" if len(txt) > 1200 else ""))


def result_block(row: dict, title: str, calibration=None, source="", show_router=False):
    st.markdown(f"#### {title}")
    st.caption(f"{row['retriever'].replace('_', '-')} · k={row['k']} · generator **{row['tier']}** (`{row['model_id'].split('/')[-1]}`)"
               + ("  ·  prompt trimmed to fit the budget" if row["prompt_truncated"] else "")
               + ("  ·  answer hit the token limit" if row.get("hit_max_tokens") else ""))
    if show_router and calibration is not None:
        router_card(row, calibration, source)
    st.markdown("**Generated answer**")
    st.info(row["answer"] or "(empty answer)")
    kpi_cards(row)
    passages_expander(row)


def log_row(row: dict):
    from src.demo_pipeline import LOG_FIELDS, append_log

    append_log(row, st.session_state.session_id)
    st.session_state.query_log.append({k: row.get(k, "") for k in LOG_FIELDS} | {"session_id": st.session_state.session_id})


# Tab 1: live prototype
with tab1:
    st.subheader("Live QA prototype and query logger")
    st.caption("Retrieve → (route) → generate → score. Every run records retrieval, TTFT, decode and total latency plus five quality metrics. " + LIVE_NOTE)
    qs = test_queries(dataset) if dp.prep_ready(dataset) else []
    src_choice = st.radio("Question source", ["Test-split question", "Custom question"], horizontal=True)
    q_text, q_idx = "", 0
    if src_choice == "Test-split question" and qs:
        q_idx = st.selectbox("Question", range(len(qs)), format_func=lambda i: f"{qs[i].qid} · {qs[i].question[:90]}{'…' if len(qs[i].question) > 90 else ''}")
        st.caption(qs[q_idx].question)
    else:
        q_text = st.text_area("Your technical question", placeholder="e.g. How do I fix a timeout when the server starts?", height=90)

    mode = st.radio("Pipeline mode", [MODE_FIXED, MODE_ADAPT, MODE_COMPARE], horizontal=True)
    retr, k_sel, tier_sel = "dense", 3, "Small"
    if mode == MODE_FIXED:
        c1, c2, c3 = st.columns(3)
        retr = c1.selectbox("Retriever", list(RETRIEVERS), format_func=lambda r: RETRIEVERS[r])
        k_sel = c2.select_slider("Retrieved passages k", options=[1, 3, 5, 8], value=3)
        names = LAPTOP_TIER_MODELS if tier_set == "laptop" else FULL_TIER_MODELS
        tier_sel = c3.selectbox("Generator size", list(names), format_func=lambda t: f"{t} · {names[t].split('/')[-1]}")
    elif mode == MODE_COMPARE:
        st.caption("Runs **Fixed Heavy (k=8, Large generator)** and the **Adaptive Pipeline** on the same question and reports the difference.")

    run = st.button("▶ Run pipeline", type="primary", disabled=(not qs and src_choice == "Test-split question"))
    if run:
        if src_choice == "Custom question" and not q_text.strip():
            st.warning("Type a question first.")
        else:
            try:
                with st.spinner("Loading retrieval index and models (first run only; load time is not counted in latency)…"):
                    pipe = build_pipeline(dataset, tier_set, max_new, budget, threads)
                    st.session_state.hw_actual = {"device": get_hw().device, "badge": get_hw().badge, "ram_gb": get_hw().ram_gb, "cores": get_hw().cpu_cores}
                from src.demo_pipeline import make_custom_query

                q = qs[q_idx] if src_choice == "Test-split question" else make_custom_query(dataset, q_text, len(st.session_state.query_log))
                with st.spinner("Running the pipeline…"):
                    if mode == MODE_FIXED:
                        rows = [pipe.run_fixed(q, retr, k_sel, tier_sel)]
                    elif mode == MODE_ADAPT:
                        rows = [pipe.run_adaptive(q)]
                    else:
                        rows = list(pipe.run_compare(q))
                for r in rows:
                    log_row(r)
                st.session_state.last_run = {"mode": mode, "rows": rows, "q": q, "cal": pipe.calibration, "source": pipe.cal_source}
            except Exception as exc:  # noqa: BLE001
                st.error(f"Run failed: {exc}")
                st.caption("If a model could not be loaded, run `python download_laptop_models.py` once (needs internet).")

    lr = st.session_state.last_run
    if lr:
        q, rows = lr["q"], lr["rows"]
        st.divider()
        st.markdown(f"**Question** · `{q.qid}`")
        st.write(q.question)
        if q.reference_response:
            with st.expander("Ground-truth reference answer and evidence"):
                st.markdown("**Reference answer**")
                st.write(q.reference_response)
                st.markdown(f"**Annotated evidence** ({q.evidence_source.replace('_', ' ')})")
                for s in q.evidence_sentences:
                    st.write(f"- {s}")
        if lr["mode"] == MODE_COMPARE:
            heavy, ad = rows
            saved, pct = heavy["total_latency_ms"] - ad["total_latency_ms"], 100 * (heavy["total_latency_ms"] - ad["total_latency_ms"]) / heavy["total_latency_ms"]
            dq = (ad["composite_quality"] - heavy["composite_quality"]) if heavy["composite_quality"] == heavy["composite_quality"] else None
            msg = f"Adaptive pipeline {'saved' if saved >= 0 else 'cost an extra'} **{abs(saved):,.0f} ms ({abs(pct):.1f}%)** versus Fixed Heavy"
            msg += f" · quality change **{dq:+.3f}** (adaptive − heavy)" if dq is not None else " · quality comparison needs a test-split question"
            (st.success if saved > 0 else st.warning)(msg)
            c1, c2 = st.columns(2)
            with c1:
                result_block(heavy, "Fixed Heavy (k=8 · Large)")
            with c2:
                result_block(ad, "Adaptive Pipeline", lr["cal"], lr["source"], show_router=True)
            st.caption("")
        else:
            result_block(rows[0], "Result", lr["cal"], lr["source"], show_router=(lr["mode"] == MODE_ADAPT))

    st.divider()
    st.markdown("#### Recorded query log (this session)")
    log = st.session_state.query_log
    if log:
        from src.demo_pipeline import LOG_FIELDS, LOG_NUMERIC, LOG_PATH

        df = pd.DataFrame(log)[LOG_FIELDS]
        for col in LOG_NUMERIC:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        st.dataframe(df.drop(columns=["answer", "question", "session_id"]), **STRETCH, height=240)
        c1, c2, c3 = st.columns([1, 1, 3])
        c1.download_button("⬇ Download Recorded Queries CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name=f"prototype_query_log_{st.session_state.session_id}.csv", mime="text/csv")
        if c2.button("🧹 Clear Session Log"):
            st.session_state.query_log = []
            st.session_state.last_run = None
            st.rerun()
        c3.caption(f"Every run is also appended to `{LOG_PATH.relative_to(config.PROJECT_ROOT)}` (permanent record). Clearing only resets this session's view.")
    else:
        st.caption("No queries recorded yet. Run one above.")


# Tab 2: live mini-benchmark
with tab2:
    st.subheader("Live mini-benchmark (about 1–2 minutes)")
    st.caption("Runs the fixed configurations you pick, the Fixed Heavy baseline and the Adaptive Pipeline on the same random test questions. " + LIVE_NOTE)
    c1, c2, c3 = st.columns(3)
    n_q = c1.radio("Number of test queries (N)", [5, 10, 15], horizontal=True)
    ks = c2.multiselect("Retrieved passages k (fixed configs)", [1, 3, 5, 8], default=[1, 3, 5])
    names = LAPTOP_TIER_MODELS if tier_set == "laptop" else FULL_TIER_MODELS
    fixed_tier = c3.selectbox("Generator for the fixed configs", list(names), index=0, format_func=lambda t: f"{t} · {names[t].split('/')[-1]}")
    inc_heavy = st.checkbox("Include the Fixed Heavy baseline (k=8, Large): slower, needed for the speed-up chart", value=True)
    n_jobs = n_q * (len(ks) + int(inc_heavy) + 1)
    st.caption(f"{n_jobs} generations in total. Queries are a fixed random sample (seed 42), so a re-run is reproducible.")
    go = st.button("▶ Run Live Mini-Benchmark Now", type="primary", disabled=(not ks or not dp.prep_ready(dataset)))
    if go:
        try:
            with st.spinner("Loading retrieval index and models…"):
                pipe = build_pipeline(dataset, tier_set, max_new, budget, threads)
            from src.demo_pipeline import (MINI_QUERIES_PATH, MINI_SUMMARY_PATH, run_mini_benchmark, sample_queries, summarize_mini)

            queries = sample_queries(pipe.res, n_q)
            bar, status = st.progress(0.0), st.empty()

            def on_progress(i, total, label, elapsed):
                eta = elapsed / i * (total - i)
                bar.progress(i / total)
                status.write(f"Running **{label}** · {i}/{total} · elapsed {elapsed:.0f}s · about {eta:.0f}s left")

            rows = run_mini_benchmark(pipe, queries, sorted(ks), fixed_tier, inc_heavy, on_progress)
            bar.progress(1.0)
            status.write(f"Done: {len(rows)} generations.")
            summary = summarize_mini(rows)
            summary.insert(0, "dataset", dataset)
            summary.insert(1, "timestamp", datetime.now().isoformat(timespec="seconds"))
            summary.insert(2, "tier_set", tier_set)
            MINI_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(MINI_SUMMARY_PATH, index=False)
            pd.DataFrame([{k: v for k, v in r.items() if k != "passages"} for r in rows]).to_csv(MINI_QUERIES_PATH, index=False)
            st.session_state.mini = {"summary": summary, "rows": rows, "n": n_q}
        except Exception as exc:  # noqa: BLE001
            st.error(f"Mini-benchmark failed: {exc}")
    mini = st.session_state.mini
    if mini:
        s = mini["summary"]
        ad_row = s[s.label == "Adaptive Pipeline"]
        hv_row = s[s.label.str.startswith("Fixed Heavy")]
        if len(ad_row) and len(hv_row):
            sp, dq = float(ad_row.speedup_vs_fixed_heavy_pct.iloc[0]), float(ad_row.mean_composite_quality.iloc[0] - hv_row.mean_composite_quality.iloc[0])
            k1, k2, k3 = st.columns(3)
            k1.metric("Adaptive speed-up vs Fixed Heavy", f"{sp:.1f}%")
            k2.metric("Mean latency saved", ms(float(hv_row.mean_total_latency_ms.iloc[0] - ad_row.mean_total_latency_ms.iloc[0])))
            k3.metric("Quality change (adaptive − heavy)", f"{dq:+.3f}")
            (st.success if sp > 0 else st.warning)(f"On these {mini['n']} queries the adaptive pipeline was {'faster' if sp > 0 else 'not faster'}. N this small shows the mechanism, not a statistical result; Tab 3 has the full test-split result.")
        st.plotly_chart(dp.fig_mini(s), **STRETCH)
        show = s.drop(columns=["dataset", "timestamp", "tier_set"]).round(3)
        st.dataframe(show, **STRETCH)
        from src.demo_pipeline import MINI_SUMMARY_PATH as _P

        st.caption(f"Saved to `{_P.relative_to(config.PROJECT_ROOT)}`.")
        adr = [r for r in mini["rows"] if r["label"] == "Adaptive Pipeline"]
        if adr:
            mix = pd.Series([r["router_tier"] for r in adr]).value_counts()
            st.caption("Adaptive tier mix on these queries: " + ", ".join(f"{t} {n}" for t, n in mix.items()))


# Tab 3: pre-computed benchmark
with tab3:
    T = tables_cached(tables_signature())
    n_ds = {d: int(T.cs[T.cs.dataset == d].n.max()) for d in dp.PRETTY} if T.ok and "n" in T.cs else {}
    n_txt = f"{n_ds.get('techqa', 314)} TechQA + {n_ds.get('delucionqa', 184)} DelucionQA queries"
    st.subheader(f"Pre-computed Full Test-Split Benchmark ({n_txt})")
    st.caption("12 fixed configurations (k ∈ {1, 3, 5, 8} × Qwen2.5 1.5B / 3B / 7B generators) plus the adaptive pipeline, measured once on the full held-out test split "
               "(greedy decoding, one query at a time). These are stored results, not recomputed here; latencies were measured on the benchmark machine.")
    if not T.ok:
        st.error("Pre-computed tables were not found in `results/tables/`. Copy that folder from the project to see this tab.")
    else:
        if T.missing:
            st.warning("Some optional tables are missing: " + ", ".join(T.missing))
        verd = dp.parse_verdicts(T.verdicts_md)
        pretty = dp.PRETTY[dataset]
        h2row = T.h2[T.h2.dataset == dataset].iloc[0] if T.h2 is not None and (T.h2.dataset == dataset).any() else None

        st.markdown(f"### {dp.DOMAIN_LABEL[dataset]}")
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("#### RQ1 · H1: do k and generator size matter?")
            txt = verd.get((pretty, "H1"), "")
            if txt:
                lab, lvl = dp.verdict_status("H1", txt)
                getattr(st, lvl)(f"**{lab}**")
                st.caption(txt)
            fr = T.friedman.get(dataset)
            if fr is not None:
                st.dataframe(fr.round(4), **STRETCH, hide_index=True)
                st.caption("Friedman omnibus test over the 12 configurations; Kendall's W is the effect size (0 to 1).")
        with v2:
            st.markdown("#### RQ2 · H2: can adaptive routing keep quality and cut latency?")
            if h2row is not None:
                txt = verd.get((pretty, "H2")) or dp.derive_h2_verdict(h2row)
                lab, lvl = dp.verdict_status("H2", txt)
                getattr(st, lvl)(f"**{lab}**")
                m = st.columns(4)
                m[0].metric("Latency", f"{h2row.speedup_pct:.1f}% lower", f"{h2row.saved_mean_ms:,.0f} ms saved")
                m[1].metric("Quality (fixed − adaptive)", f"{h2row.quality_diff_mean:+.4f}", f"95% CI [{h2row.quality_diff_ci95_lo:+.4f}, {h2row.quality_diff_ci95_hi:+.4f}]", delta_color="off")
                m[2].metric("Non-inferiority (δ = 0.02)", "Yes" if bool(h2row.noninferior) else "No", f"p = {h2row.p_noninferiority_ttest:.2g}", delta_color="off")
                m[3].metric("Effect size (Cohen's d)", f"{h2row.cohens_d_z:.2f}")
                st.caption(f"Best fixed configuration on validation: `{h2row.best_fixed_config}`. Tiers chosen from the validation Pareto frontier: "
                           f"Fast `{h2row.fast_cfg}` ({h2row.tier_fast_pct:.0f}%), Balanced `{h2row.balanced_cfg}` ({h2row.tier_balanced_pct:.0f}%), Heavy `{h2row.heavy_cfg}` ({h2row.tier_heavy_pct:.0f}%).")
                dom = str(h2row.get("dominated_by_fixed", "none"))
                if dom != "none":
                    st.info(f"Honest note: the fixed configuration {dom} is at least as good and at least as fast as the adaptive pipeline on this dataset, so a simple fixed choice works too.")
            else:
                st.info("Adaptive results (h2_summary.csv) not found.")

        st.markdown("#### Retriever comparison (Objective 3)")
        if T.retr is not None:
            st.plotly_chart(dp.fig_retriever(T.retr, dataset), **STRETCH)
            r5 = T.retr[(T.retr.dataset == dataset) & (T.retr.k == 5)].set_index("retriever")
            st.caption("At k=5: " + " · ".join(f"{dp.RETR_NAMES[r]} MRR {r5.loc[r, 'mrr_at_k']:.3f}, {r5.loc[r, 'retrieval_latency_ms_mean']:.1f} ms" for r in r5.index))
        st.markdown("#### RQ1 · answer quality vs. retrieved passages")
        st.plotly_chart(dp.fig_quality_vs_k(T.cs, dataset), **STRETCH)
        g13, g38 = dp.gains_by_k(T.cs, dataset)
        st.caption(f"Average quality gain from k=1 to k=3: **{g13:+.3f}**; from k=3 to k=8: **{g38:+.3f}** (across the three generator sizes). "
                   "Most of the gain arrives by k=3; later passages add little.")
        with st.expander("Adjacent-step Wilcoxon tests (Holm-corrected)"):
            if dataset in T.steps:
                st.dataframe(T.steps[dataset].round(4), **STRETCH, hide_index=True)
        st.markdown("#### RQ1 · latency breakdown")
        st.plotly_chart(dp.fig_latency_breakdown(T.cs, dataset), **STRETCH)
        st.markdown("#### RQ2 · Pareto frontier and adaptive pipeline")
        st.plotly_chart(dp.fig_pareto(T.cs, h2row, dataset), **STRETCH)
        if h2row is not None:
            st.plotly_chart(dp.fig_tier_mix(h2row), **STRETCH)


# Tab 4: guidelines + theory
with tab4:
    T = tables_cached(tables_signature())
    st.subheader("Practical deployment guidelines")
    if not T.ok:
        st.error("Pre-computed tables were not found in `results/tables/`.")
    else:
        st.markdown("#### SLA configuration recommender")
        c1, c2, c3 = st.columns([2, 3, 2])
        dom = c1.selectbox("Domain", list(dp.DOMAIN_LABEL), index=list(dp.DOMAIN_LABEL).index(dataset), format_func=lambda d: dp.DOMAIN_LABEL[d], key="sla_domain")
        budget_ms = c2.slider("Target latency budget (ms)", 1000, 3500, 2000, step=50)
        stat = c3.radio("Budget applies to", ["mean"], horizontal=True, index=0)
        rec = dp.recommend(T.cs, T.retr, T.h2, dom, budget_ms, "mean" if stat == "mean" else "p95")
        feas = rec["feasible"]
        if len(feas):
            best = feas.iloc[0]
            st.success(f"**Recommended: `{best.config_id}`** · retriever **Dense FAISS** unless noted below")
            m = st.columns(4)
            m[0].metric("Quality", f"{best.mean_composite_quality:.3f}", f"{best.mean_composite_quality - rec['best_quality']:+.3f} vs best fixed", delta_color="off")
            m[1].metric("Mean latency", ms(best.total_latency_ms_mean))
            m[2].metric("P95 latency", ms(best.total_latency_ms_p95))
            m[3].metric("Recall@k", f"{best.mean_recall_at_k:.3f}")
            st.caption(f"Highest-quality configuration whose {stat} latency fits {budget_ms:,} ms. On the Pareto frontier: **{'yes' if best.config_id in rec['front'] else 'no'}**.")
            st.dataframe(feas[["config_id", "mean_composite_quality", "total_latency_ms_mean", "total_latency_ms_p95"]].head(5).round(3)
                         .rename(columns={"mean_composite_quality": "quality", "total_latency_ms_mean": "mean ms", "total_latency_ms_p95": "P95 ms"}),
                         **STRETCH, hide_index=True)
        else:
            ch = rec["cheapest"]
            st.error(f"No benchmark configuration meets a {stat} latency of {budget_ms:,} ms. The fastest is `{ch.config_id}` at {ch[rec['col']]:,.0f} ms: raise the budget or shorten answers.")
        r = rec["retriever"]
        if r is not None:
            st.markdown(f"**Retriever:** best MRR@5 on {dp.PRETTY[dom]} is **{dp.RETR_NAMES.get(r.retriever, r.retriever)}** ({r.mrr_at_k:.3f}; {r.retrieval_latency_ms_mean:.1f} ms per query). Retrieval is a small share of total latency, so choose on accuracy.")
        a = rec["adaptive"]
        if a is not None:
            fit = "fits" if a.mean_latency_adaptive_ms <= budget_ms else "does not fit"
            st.markdown(f"**Adaptive pipeline:** mean {a.mean_latency_adaptive_ms:,.0f} ms at quality {a.mean_quality_adaptive:.3f}; it {fit} this budget. "
                        f"Non-inferior to the best fixed configuration: **{'yes' if bool(a.noninferior) else 'no'}**.")
        with st.expander("Guideline text from the benchmark report"):
            for ln in dp.guideline_lines(T.guidelines_md, dom):
                st.markdown(ln)
        

    
