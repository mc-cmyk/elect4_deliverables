import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv(r"C:\Users\Mc Marteen\Desktop\Elective4_project\output\jollibee_complaints.csv")

vec = CountVectorizer(ngram_range=(1,2), min_df=3)
X = vec.fit_transform(df["text_clean"].fillna(""))
counts = X.sum(axis=0).A1
terms = pd.DataFrame({"term": vec.get_feature_names_out(), "count": counts})

# Filter to Tagalog-looking terms
tagalog = ["kulang","mali","walang","malamig","sobrang","tagal","hilaw",
           "matigas","bawas","wala","hindi","masama","maalat","tuyo",
           "sirang","panis","matagal","dumating","tamad","bastos"]

mask = terms["term"].apply(lambda t: any(w in t for w in tagalog))
print(terms[mask].sort_values("count", ascending=False).head(30).to_string(index=False))