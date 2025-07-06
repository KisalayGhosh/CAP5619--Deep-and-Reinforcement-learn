import pandas as pd
import requests
import time

df = pd.read_csv("mutants.csv")

def query_variant(protein_change):
    url = f"https://myvariant.info/v1/query?q=BRCA1%20AND%20{protein_change}&fields=dbnsfp.polyphen2.hdiv.score,dbnsfp.sift.score"
    try:
        r = requests.get(url)
        if r.status_code == 200 and r.json().get("hits"):
            hit = r.json()["hits"][0]
            polyphen = hit.get("dbnsfp", {}).get("polyphen2", {}).get("hdiv", {}).get("score", None)
            sift = hit.get("dbnsfp", {}).get("sift", {}).get("score", None)
            return polyphen, sift
    except:
        pass
    return None, None

results = []
for i, row in df.iterrows():
    aa_change = f"p.{row['Original']}{int(row['Position'])}{row['Mutant']}"
    polyphen, sift = query_variant(aa_change)
    results.append({"PolyPhen2_score": polyphen, "SIFT_score": sift})
    time.sleep(0.3)

pd.DataFrame(results).to_csv("external_scores.csv", index=False)
print(" Fetched scores for available variants.")
