import json,time
from verify_references import ROOT,fetch
selection=json.loads((ROOT/'selected_references.json').read_text(encoding='utf-8'))
path=ROOT/'selected_candidate_details.json'
out=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
for doi in selection['candidates']:
    if doi in out: continue
    m=fetch('https://api.crossref.org/works/'+doi)
    out[doi]=m
    (ROOT/'selected_candidate_details.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(doi,m['title'][0],flush=True)
    time.sleep(2)
