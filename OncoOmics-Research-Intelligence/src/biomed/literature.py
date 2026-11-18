import re, requests
from collections import Counter

STOP={"the","and","for","with","that","this","from","were","was","are","have","has","into","using","used","between","among","after","before","their","these","those","which","while","than","also","not","but","can","may","our","study","studies","result","results","conclusion","background","methods","method","objective"}

def keywords(text,n=12):
    words=re.findall(r"[A-Za-z][A-Za-z-]{2,}",str(text).lower())
    counts=Counter(w for w in words if w not in STOP)
    return [{"term":k,"count":v} for k,v in counts.most_common(n)]

def summarize(text,n=3):
    sentences=[s.strip() for s in re.split(r"(?<=[.!?])\s+",str(text)) if len(s.strip())>35]
    if len(sentences)<=n: return " ".join(sentences)
    top={x["term"] for x in keywords(text,30)}
    scored=[]
    for i,s in enumerate(sentences):
        tokens=re.findall(r"[A-Za-z][A-Za-z-]{2,}",s.lower())
        scored.append((sum(t in top for t in tokens)/max(len(tokens),1),-i,s))
    return " ".join(x[2] for x in sorted(sorted(scored,reverse=True)[:n],key=lambda x:-x[1]))

def search_pubmed(query,limit=8):
    if not query.strip(): return []
    base="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    r=requests.get(base+"esearch.fcgi",params={"db":"pubmed","term":query,"retmax":min(limit,20),"retmode":"json","sort":"relevance"},timeout=20)
    r.raise_for_status()
    ids=r.json().get("esearchresult",{}).get("idlist",[])
    if not ids:return []
    f=requests.get(base+"efetch.fcgi",params={"db":"pubmed","id":",".join(ids),"retmode":"xml"},timeout=30)
    f.raise_for_status()
    from Bio import Entrez
    import io
    records=Entrez.read(io.StringIO(f.text))
    out=[]
    for item in records.get("PubmedArticle",[]):
        cit=item["MedlineCitation"]; art=cit["Article"]; pmid=str(cit["PMID"])
        abstract=" ".join(str(x) for x in art.get("Abstract",{}).get("AbstractText",[]))
        out.append({"pmid":pmid,"title":str(art.get("ArticleTitle","")),"abstract":abstract,"url":f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
    return out
