import json,time
from verify_references import ROOT,fetch
dois = '''10.1016/j.engappai.2024.109082
10.1145/3705297
10.1186/s41937-024-00131-4
10.1145/3527154
10.1016/j.dss.2024.114378
10.1016/j.dss.2025.114507
10.18637/jss.v108.i03
10.1515/jci-2022-0040
10.1007/s40471-022-00308-6
10.1016/j.dss.2024.114170
10.1016/j.asoc.2024.111307
10.1002/cb.2372
10.1002/sim.9921
10.1007/s10902-023-00697-5
10.1016/j.ejor.2023.06.036
10.1353/obs.2024.a946583
10.1287/opre.2022.2271
10.1093/ectj/utac015
10.1093/ectj/utac018
10.1093/ectj/utac003
10.1201/9781003102670-3
10.1007/s41060-026-01070-4
10.1016/j.dss.2024.114276
10.1371/journal.pone.0308718'''.splitlines()
out={}
for doi in dois:
    try:
        m=fetch('https://api.crossref.org/works/'+doi)
        out[doi]=m
        print(doi,m.get('title'),m.get('published',{}).get('date-parts'),flush=True)
    except Exception as exc:
        print(doi,str(exc),flush=True)
    (ROOT/'reference_details.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    time.sleep(2)
