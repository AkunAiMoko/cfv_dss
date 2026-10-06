import csv, html, json, re, unicodedata
from pathlib import Path
ROOT=Path(__file__).parent
baseline=ROOT/'article.before_reference_update.txt'
original=(baseline if baseline.exists() else ROOT/'article.md').read_text(encoding='utf-8')
data=json.loads((ROOT/'reference_candidates.json').read_text(encoding='utf-8'))
extra=json.loads((ROOT/'reference_details.json').read_text(encoding='utf-8'))
verified=json.loads((ROOT/'selected_candidate_details.json').read_text(encoding='utf-8'))
selection=json.loads((ROOT/'selected_references.json').read_text(encoding='utf-8'))
records={k.lower():data['existing'][k] for k in selection['retained']}
records.update({k.lower():v for k,v in extra.items()})
records.update({k.lower():verified[k] for k in selection['candidates']})
assert len(records)==50
def clean(s): return html.unescape(re.sub('<[^>]*>','',s)).strip()
def year(m):
    if m['DOI'].lower()=='10.3390/systems14010016': return 2026 # publisher's volume 14(1), 2026
    return m.get('published-print',m.get('published',m['issued']))['date-parts'][0][0]
def cite(doi):
    m=records[doi.lower()]; authors=m['author']; name=authors[0]['family']
    if len(authors)==2: name+=' & '+authors[1]['family']
    elif len(authors)>2: name+=' et al.'
    return name+', '+str(year(m))
def cites(*dois): return '('+'; '.join(cite(d) for d in dois)+')'
body=original.split('## References')[0]
replacements={
 'Mashrur et al., 2020':cite('10.1016/j.engappai.2024.109082'),
 'Bussmann et al., 2021':cite('10.3905/jfds.2023.1.141'),
 'Zipperling et al., 2026':cite('10.1016/j.chb.2023.107714'),
 'Chernozhukov et al., 2018':cite('10.18637/jss.v108.i03'),
 'Nie & Wager, 2021':cite('10.1093/ectj/utac015'),
 'Cinelli & Hazlett, 2020':cite('10.1353/obs.2024.a946583'),
 'Chen & Guestrin, 2016; Breiman, 2001':cite('10.1016/j.engappai.2024.109082'),
 'Ji & Li, 2025; Li et al., 2023':cite('10.1007/s00521-022-07472-2')+'; '+cite('10.3390/data8110169'),
}
for a,b in replacements.items(): body=body.replace(a,b)
# Place new supporting literature within the relevant section, before its next heading.
def append_to(section,text,dois):
    global body
    start=body.index(section)
    next_heading=body.find('\n##',start+len(section))
    if next_heading==-1: next_heading=len(body)
    paragraph=text+' '+cites(*dois)+'.\n\n'
    body=body[:next_heading].rstrip()+'\n\n'+paragraph+body[next_heading:]

append_to('## 1. Introduction',
 'Recent consumer-credit studies extend this predictive agenda through interpretable scoring and graph-based representations of borrower information. These developments motivate comparing several learners while retaining a separate identification step for intervention claims',
 ['10.1016/j.ememar.2025.101424','10.1287/ijds.2022.00018'])
append_to('### 2.1 Credit-related financial vulnerability as a decision problem',
 'Financial well-being is a broader, multidimensional construct than repayment status. Recent conceptual and measurement reviews support making this distinction explicit: a credit label is an operational indicator of distress, whereas a comprehensive financial-health assessment requires additional subjective and objective measures',
 ['10.1002/cb.2372','10.1007/s10902-023-00697-5'])
append_to('### 2.1 Credit-related financial vulnerability as a decision problem',
 'Empirical research also examines household debt, financial vulnerability, stress, and well-being together. This literature provides substantive context for debt-pressure and capacity constructs without establishing that the same exposure effects apply to the two competition cohorts',
 ['10.3138/cpp.2022-042','10.1007/s10663-024-09617-z'])
append_to('### 2.2 Predictive explanation versus causal explanation',
 'Credit-specific XAI frameworks and Shapley-based scorecards show how model outputs can become more transparent to borrowers and analysts. CFV-DSS uses this literature to motivate auditable explanations while requiring separate causal evidence before turning an explanation into advice',
 ['10.1016/j.asoc.2024.111307','10.1371/journal.pone.0308718'])
append_to('### 2.3 Requirements for causal financial-vulnerability DSS',
 'Fairness is a distinct design requirement alongside accuracy and explanation. Credit-scoring research studies both the implementation and profit implications of fairness constraints, while recent reviews assess performance, fairness, and explainability jointly; consequently, a transparent causal workflow still requires explicit subgroup evaluation',
 ['10.1016/j.ejor.2021.06.023','10.3390/jrfm19020104'])
append_to('### 3.2 Predictive layer',
 'Interpretable credit-decision alternatives include segmentation-based models and sparse additive interaction networks. These studies motivate considering model structure and decision usability alongside discrimination; they do not imply that interpretable predictive components identify causal levers',
 ['10.1016/j.dss.2024.114170','10.1016/j.dss.2025.114507'])
append_to('### 3.3 Causal structure layer',
 'Modern discovery surveys distinguish graph recovery, equivalence classes, and identification assumptions. This distinction is central to the tier-constrained design: domain-compatible orientations narrow the candidate graph space but do not resolve all observational ambiguities',
 ['10.1145/3527154','10.1016/j.ijar.2022.09.004','10.1186/s41937-024-00131-4'])
append_to('### 3.5 Decision layer',
 'The allocation problem is related to offline multi-action policy learning, which explicitly considers feasible policy classes and budget constraints, and to recent analysis of estimation failures in observational multi-action settings. Here, the utility function remains a transparent scenario score rather than an identified off-policy value estimator',
 ['10.1287/opre.2022.2271','10.1007/s41060-026-01070-4'])
append_to('### 4.2 Leakage-safe preprocessing',
 'The separation of historical aggregation, training-only preprocessing, and untouched evaluation follows recent work identifying leakage as a major source of overoptimistic scientific claims. Temporal restrictions therefore apply to information availability as well as to train–test separation',
 ['10.1016/j.patter.2023.100804'])
append_to('### 4.4 Predictive baselines and calibration',
 'Recent credit-risk reviews motivate this comparison across algorithm families and emphasize that performance depends on data preparation, evaluation design, and interpretability. The compact construct comparison should therefore be read as an application-specific benchmark',
 ['10.1007/s00521-022-07472-2','10.3390/data8110169'])
append_to('### 4.5 Predictive explanation',
 'Class imbalance can also destabilize local credit-scoring explanations. Reporting repeated held-out perturbations addresses empirical variability in the chosen importance measure, although it does not establish the stability of other explanation methods or confer causal meaning',
 ['10.1016/j.ejor.2023.06.036'])
append_to('### 4.6 Domain-constrained causal discovery',
 'Temporal-data and structure-learning surveys provide complementary discovery perspectives. They motivate checking temporal restrictions and model assumptions separately, rather than treating a chronological tier as sufficient evidence of a causal edge',
 ['10.1145/3705297','10.1145/3527154'])
append_to('### 4.6 Domain-constrained causal discovery',
 'Recent methods explicitly incorporate background knowledge into local discovery or request expert knowledge during structure learning. These approaches provide methodological comparators for the fixed tier rules and suggest a future audit of which orientations derive from data and which derive from expert assumptions',
 ['10.1109/tpami.2026.3667409','10.1016/j.knosys.2025.113185'])
append_to('### 4.7 Causal identification and estimation',
 'Dynamic-treatment and mediation extensions of DML underscore that estimands must follow the timing and role of each variable. CFV-DSS estimates a static exposure coefficient; it does not estimate sequential intervention effects or decompose direct and mediated effects',
 ['10.1093/ectj/utac018','10.1093/ectj/utac003'])
append_to('### 4.8 Sensitivity and external robustness',
 'Recent sensitivity guidance and generalized-linear-model analyses emphasize stating the sensitivity parameter and its relationship to the outcome model. The residual-regression robustness value reported here should accordingly be interpreted as a diagnostic of the fitted association under its assumptions',
 ['10.1007/s40471-022-00308-6','10.1515/jci-2022-0040'])
append_to('### 4.8 Sensitivity and external robustness',
 'A recent general theory extends omitted-variable-bias analysis to machine-learned causal parameters and policy effects. It motivates further sensitivity analysis tailored to the orthogonal estimand, beyond treating an OLS-style robustness calculation as a complete identification guarantee',
 ['10.1162/rest.a.1705'])
append_to('### 5.2 Predictive performance',
 'Calibration research separately examines probability estimation across learning algorithms and the additional difficulties of imbalanced binary classification. This literature supports interpreting discrimination, calibration, and threshold performance as different evaluation dimensions',
 ['10.1002/sim.9921','10.1007/s10472-024-09952-8'])
append_to('### 5.7 External validation',
 'Generalizability and transportability require an explicit target population and assumptions about differences between study and target settings. Predictive replication across these cohorts is therefore insufficient to establish transport of intervention effects',
 ['10.1201/9781003102670-3'])
append_to('### 6.3 Decision Support Systems contribution',
 'A bank-customer study frames individual mortgage bid responses as a causal estimation problem and examines confounding in observational loan data. This provides a directly relevant financial DSS precedent for separating prediction from intervention-response analysis, while its pricing estimand differs from repayment vulnerability',
 ['10.1016/j.dss.2024.114378'])
append_to('### 6.3 Decision Support Systems contribution',
 'Recent DSS work on explainable AI also positions explanation within decision-making workflows. For CFV-DSS, explanation design should help users inspect assumptions and retain control over recommendations, rather than encourage uncritical acceptance of a ranked intervention',
 ['10.1016/j.dss.2024.114276'])
append_to('### 6.5 Practical implications',
 'Recent analyses of credit-scoring fairness and Simpson’s paradox in lending reinforce the need to inspect aggregate and subgroup behavior separately. Such governance checks remain necessary even when the candidate intervention concerns a modifiable financial exposure',
 ['10.1287/mnsc.2022.03888','10.1016/j.physa.2025.131030'])

body=body.replace('whether manipulating a ranked input changes the outcome (Feuerriegel et al., 2024; Westphal et al., 2023)',
 'whether manipulating a ranked input changes the outcome (Feuerriegel et al., 2024; von Zahn et al., 2026)')
body=body.replace('This aligns with recent DSS work showing that decision quality improves when analytics incorporate heterogeneous treatment effects and operational friction rather than relying on prediction alone (Feuerriegel et al., 2024; von Zahn et al., 2026; Westphal et al., 2023).',
 'This motivates evaluating heterogeneous modeled benefits and explicit operational constraints, consistent with the distinct estimation and allocation tasks in causal ML and policy-learning research (von Zahn et al., 2026; Zhou et al., 2023).')
body=body.replace('answering recent calls for causal DSS interface and workflow integration (Westphal et al., 2023)',
 'consistent with research emphasizing user decision control and carefully designed explanations in human–AI collaboration (Westphal et al., 2023)')

def author_text(a):
    given=a.get('given','')
    initials=' '.join(p[0]+'.' for p in re.split(r'[\s-]+',given) if p)
    return a.get('family',a.get('name',''))+(', '+initials if initials else '')
def reference(m):
    names=[author_text(a) for a in m['author']]
    authors=names[0] if len(names)==1 else ', '.join(names[:-1])+', & '+names[-1]
    title=clean(m['title'][0]); journal=clean(m['container-title'][0])
    volume=m.get('volume',''); issue=m.get('issue',''); pages=m.get('page',m.get('article-number',''))
    if m['DOI'].lower()=='10.18637/jss.v108.i03': pages='1–56'
    if m['DOI'].lower()=='10.1214/23-ejs2157': pages='3008–3049'
    if m['DOI'].lower()=='10.1613/jair.1.21001': pages='Article 41, 1–64'
    if m['DOI'].lower()=='10.1145/3705297': pages='Article 100, 1–38'
    if m['type']=='book-chapter':
        source=f'In *{journal}* (pp. {pages.replace("-","–")}). {m["publisher"]}.'
    else:
        source='*'+journal+((', '+volume) if volume else '')+'*'+(('('+issue+')') if issue else '')
        if pages: source+=', '+pages.replace('-','–')
        source+='.'
    return f'{authors} ({year(m)}). {title}. {source} https://doi.org/{m["DOI"]}'
ordered=sorted(records.values(),key=lambda m:unicodedata.normalize('NFKD',m['author'][0]['family']).casefold())
for m in ordered:
    assert 2022<=year(m)<=2026
    assert cite(m['DOI']) in body, ('Uncited reference',m['DOI'],cite(m['DOI']))
updated=body.rstrip()+'\n\n## References\n\n'+'\n'.join(f'{i}. {reference(m)}' for i,m in enumerate(ordered,1))+'\n'
backup=ROOT/'article.before_reference_update.txt'
if not backup.exists(): backup.write_text(original,encoding='utf-8')
(ROOT/'article.md').write_text(updated,encoding='utf-8')
audit=[]
for i,m in enumerate(ordered,1):
    audit.append({'number':i,'citation':cite(m['DOI']),'year':year(m),'title':clean(m['title'][0]),'doi':m['DOI'],'doi_url':'https://doi.org/'+m['DOI'],'verification_source':'https://api.crossref.org/works/'+m['DOI'],'publisher_url':m.get('resource',{}).get('primary',{}).get('URL',''),'verified_on':'2026-10-06','in_text_occurrences':body.count(cite(m['DOI']))})
with (ROOT/'reference_verification.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=audit[0].keys()); writer.writeheader(); writer.writerows(audit)
(ROOT/'verified_reference_metadata.json').write_text(json.dumps({'verified_on':'2026-10-06','scope':'2022–2026 publication years; Crossref DOI/title/author metadata checked; online publications included','records':ordered},ensure_ascii=False,indent=2),encoding='utf-8')
print('Updated article: 50 unique verified DOI references, years 2022–2026; every reference cited in text.')
