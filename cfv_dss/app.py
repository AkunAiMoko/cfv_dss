"""Streamlit deployment dashboard for CFV-DSS."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from cfv_dss.config import ART, CONSTRUCTS, FEASIBILITY

st.set_page_config(page_title="CFV-DSS", page_icon="📊", layout="wide")
st.markdown("""
<style>
.main {background:#f8fafc}.stMetric {background:white;border:1px solid #e2e8f0;padding:1rem;border-radius:.75rem}
.risk-high{padding:1rem;border-radius:.5rem;background:#fee2e2;color:#991b1b}.risk-med{padding:1rem;border-radius:.5rem;background:#fef3c7;color:#92400e}.risk-low{padding:1rem;border-radius:.5rem;background:#dcfce7;color:#166534}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    return joblib.load(ART / "models" / "best_model_bundle.joblib")

@st.cache_data
def load_tables():
    return {
        "metrics": pd.read_csv(ART / "metrics" / "model_comparison.csv"),
        "importance": pd.read_csv(ART / "metrics" / "predictive_importance.csv"),
        "effects": pd.read_csv(ART / "causal" / "causal_effects_dml.csv"),
        "divergence": pd.read_csv(ART / "causal" / "predictive_vs_causal_divergence.csv"),
        "policies": pd.read_csv(ART / "causal" / "decision_policy_simulation.csv"),
        "threshold_sensitivity": pd.read_csv(ART / "causal" / "graph_threshold_sensitivity.csv"),
        "overlap": pd.read_csv(ART / "causal" / "overlap_diagnostics.csv"),
        "confounding": pd.read_csv(ART / "causal" / "unobserved_confounding_sensitivity.csv"),
        "profile": pd.read_csv(ART / "reports" / "data_profile.csv"),
    }

st.title("CFV-DSS")
st.caption("Causal Financial Vulnerability Decision Support — prediction, diagnosis, and intervention priority")

if not (ART / "models" / "best_model_bundle.joblib").exists():
    st.error("Artefak model belum tersedia. Jalankan: `.venv\\Scripts\\python.exe -m cfv_dss.train`")
    st.stop()

bundle = load_assets(); tables = load_tables()
page = st.sidebar.radio("Navigasi", ["Overview", "Individual Assessment", "Model Evaluation", "Causal Diagnosis", "Policy Simulation", "Governance"])
st.sidebar.info("Decision-support prototype. Bukan keputusan kredit otomatis atau bukti efek intervensi dunia nyata.")

if page == "Overview":
    st.subheader("Ringkasan CFV-DSS")
    m = tables["metrics"].iloc[0]
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Model terbaik", bundle["best_model_name"])
    c2.metric("Test ROC-AUC", f"{m.roc_auc:.3f}")
    c3.metric("Test PR-AUC", f"{m.pr_auc:.3f}")
    c4.metric("Brier score", f"{m.brier_score:.3f}")
    st.markdown("### Empat keluaran keputusan")
    cols=st.columns(4)
    for c,title,body in zip(cols,["1. Calibrated Risk","2. Predictive Explanation","3. Causal Priority","4. Policy Value"],["Siapa berisiko mengalami repayment distress?","Faktor apa yang mendorong prediksi model?","Kandidat faktor mana yang relevan untuk intervensi?","Kebijakan mana memberi utility simulasi tertinggi?"]):
        c.markdown(f"**{title}**\n\n{body}")
    st.image(str(ART / "figures" / "predictive_causal_map.png"), caption="Predictive importance ≠ causal relevance", use_container_width=True)

elif page == "Individual Assessment":
    st.subheader("Penilaian individual")
    st.warning("Input berupa konstruk terstandar finansial. Hasil mendukung diagnosis; tidak boleh digunakan sebagai satu-satunya dasar keputusan kredit.")
    values={}
    cols=st.columns(2)
    defaults={"DBP":.25,"US":.50,"DS":.10,"HAIC":75000,"CE":1.5,"RR":.85,"AAP":3.0,"FHD":5.0,"HB":.5,"CNP":1.0,"EMPSTAB":5.0,"AGE":40.0,"DEBT_RATIO_BUREAU":.3}
    help_text={k:f"{v[0]} — {v[3]}" for k,v in CONSTRUCTS.items()}
    for i,name in enumerate(CONSTRUCTS): values[name]=cols[i%2].number_input(name, value=float(defaults[name]), help=help_text[name])
    if st.button("Hitung risiko dan prioritas", type="primary"):
        row=pd.DataFrame([values]); transformed=bundle["preprocessor"].transform(row[bundle["feature_names"]]); prob=float(bundle["model"].predict_proba(transformed)[0,1])
        level="Tinggi" if prob>=.30 else "Sedang" if prob>=.12 else "Rendah"; css="risk-high" if level=="Tinggi" else "risk-med" if level=="Sedang" else "risk-low"
        st.markdown(f'<div class="{css}"><b>Risiko {level}</b><br>Estimated probability: {prob:.1%}</div>',unsafe_allow_html=True)
        eff=tables["effects"].copy(); eff["individual_priority_score"]=eff.apply(lambda r: abs(r.causal_effect_per_sd)*FEASIBILITY.get(r.exposure,.5)*max(values.get(r.exposure,0),0),axis=1)
        st.markdown("### Prioritas intervensi kandidat")
        st.dataframe(eff.sort_values("individual_priority_score",ascending=False)[["construct_label","causal_effect_per_sd","ci_lower_95","ci_upper_95","actionability","individual_priority_score"]],use_container_width=True,hide_index=True)
        st.caption("Prioritas memakai estimasi observasional berbasis asumsi; bukan rekomendasi klinis, finansial, atau keputusan kredit otomatis.")

elif page == "Model Evaluation":
    st.subheader("Komparasi delapan algoritma")
    st.dataframe(tables["metrics"],use_container_width=True,hide_index=True)
    st.image(str(ART / "figures" / "model_comparison.png"),use_container_width=True)
    st.markdown("### Predictive importance")
    st.info("Permutation importance menjawab: apa yang membantu prediksi model? Bukan: apa yang akan mengubah outcome?")
    st.dataframe(tables["importance"],use_container_width=True,hide_index=True)

elif page == "Causal Diagnosis":
    st.subheader("Domain-constrained causal diagnosis")
    t1,t2,t3=st.tabs(["Consensus graph","Effects and divergence","Sensitivity & overlap"])
    with t1:
        st.image(str(ART / "figures" / "consensus_causal_graph.png"),use_container_width=True)
        st.caption("Edges retained at bootstrap stability ≥60%; orientation constrained by temporal/domain tiers.")
    with t2:
        st.dataframe(tables["divergence"],use_container_width=True,hide_index=True)
        st.image(str(ART / "figures" / "predictive_causal_map.png"),use_container_width=True)
        st.caption("Causal effects identified only under measured-confounding, positivity, consistency, and model assumptions.")
    with t3:
        st.markdown("### Graph-retention threshold sensitivity (60%, 70%, 80%)")
        st.dataframe(tables["threshold_sensitivity"],use_container_width=True,hide_index=True)
        if (ART / "figures" / "threshold_sensitivity.png").exists():
            st.image(str(ART / "figures" / "threshold_sensitivity.png"),use_container_width=True)
        st.markdown("### Generalized propensity overlap & 1%-99% trimming")
        st.dataframe(tables["overlap"],use_container_width=True,hide_index=True)
        if (ART / "figures" / "overlap_diagnostics.png").exists():
            st.image(str(ART / "figures" / "overlap_diagnostics.png"),use_container_width=True)
        st.markdown("### Cinelli–Hazlett unobserved-confounding sensitivity")
        st.dataframe(tables["confounding"],use_container_width=True,hide_index=True)

elif page == "Policy Simulation":
    st.subheader("Model-based counterfactual policy simulation")
    st.dataframe(tables["policies"],use_container_width=True,hide_index=True)
    st.image(str(ART / "figures" / "policy_value.png"),use_container_width=True)
    st.warning("Policy value berasal dari simulasi model. Bukan observed real-world treatment effect dan tidak menggantikan evaluasi prospektif.")

else:
    st.subheader("Model governance dan batas penggunaan")
    st.markdown("""
- Outcome mengukur **credit-related financial vulnerability**, bukan kesejahteraan finansial menyeluruh.
- Data observasional tidak membuktikan kausalitas; estimasi bergantung pada measured confounding dan struktur DAG.
- Protected attributes tidak digunakan dalam konstruk inti, tetapi bias historis tetap mungkin hadir melalui proxy.
- Predictive importance tidak boleh diterjemahkan langsung menjadi intervensi.
- Keputusan berdampak tinggi wajib melibatkan penilaian manusia, proses banding, monitoring drift, dan validasi lokal.
- Policy simulation bukan bukti dampak intervensi dunia nyata.
- Dataset publik lama membatasi transportability.
""")
    st.markdown("### Reproducibility artifacts")
    st.code(str(ART))
    st.dataframe(tables["profile"],use_container_width=True,hide_index=True)
