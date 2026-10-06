import concurrent.futures, json, re, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
def fetch(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'CFV-DSS-bibliography-verification/1.0'})
            with urllib.request.urlopen(req, timeout=45) as response:
                return json.load(response)['message']
        except Exception:
            if attempt == 2: raise
            time.sleep(12 * (attempt + 1))

queries = [
 'credit scoring explainable machine learning',
 'consumer credit risk machine learning review',
 'credit scoring fairness explainability',
 'financial well-being vulnerability household debt',
 'causal discovery survey background knowledge',
 'double machine learning causal inference',
 'causal sensitivity omitted variable bias robustness',
 'policy learning treatment allocation budget constraints',
 'machine learning calibration imbalanced classification',
 'data leakage machine learning reproducibility',
 'causal inference transportability external validity',
 'human AI decision support explanations',
]
def search(q):
    params = urllib.parse.urlencode({'query.title':q, 'filter':'from-pub-date:2022-01-01,until-pub-date:2026-10-06,type:journal-article','rows':9})
    return q, fetch('https://api.crossref.org/works?'+params)['items']
if __name__ == '__main__':
    results = {}
    for q in queries:
        _, items = search(q)
        results[q] = items
        (ROOT/'reference_candidates.json').write_text(json.dumps({'queries':results},ensure_ascii=False,indent=2),encoding='utf-8')
        print('Retrieved:', q, flush=True)
        time.sleep(3)
    original = (ROOT/'article.md').read_text(encoding='utf-8')
    dois = re.findall(r'https://doi.org/([^\s]+)',original)
    def verify(doi):
        try: return doi, fetch('https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='/'))
        except Exception as exc: return doi, {'error':str(exc)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        existing = dict(pool.map(verify,dois))
    (ROOT/'reference_candidates.json').write_text(json.dumps({'queries':results,'existing':existing},ensure_ascii=False,indent=2),encoding='utf-8')
    for q, items in results.items():
        print('\nQUERY:',q)
        for m in items:
            print(m['DOI'],m.get('published',{}).get('date-parts'),m.get('title'),'; '.join(a.get('family','') for a in m.get('author',[])[:3]))
    print('\nEXISTING:')
    for doi,m in existing.items(): print(doi,m.get('published',{}).get('date-parts'),m.get('title'),m.get('error',''))
