"""ReviewLens - Part 2: complaint extraction from negative reviews.

Level 1 - chain-wide term frequency (unigrams to trigrams, min_df=3).
Level 2 - regex theme tagging, per-branch theme rates and lift vs the chain.

Outputs (written to output/):
  complaint_terms_chain.csv    top 50 complaint terms across the chain
  complaint_terms_branch.csv   top bigrams/trigrams per branch (dashboard drill-down)
  complaint_themes_branch.csv  branch x theme rate, chain rate and lift (long format)
  branch_priorities.csv        each branch's top 2 themes by lift

Run:  python extract_complaints.py
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from text_prep import NEGATIONS, normalize, tokenize

OUT = Path(__file__).parent / "output"
TOP_N_CHAIN = 50
TOP_N_BRANCH = 15
NOTABLE_LIFT = 1.3
MIN_THEME_HITS = 3  # a theme needs this many complaints at a branch to rank as a top issue

# Words too generic to say anything about *what* went wrong. A term is hidden
# only if every word in it is generic, so "late delivery" stays but "delivery" goes.
DISPLAY_STOPWORDS = {
    "order", "orders", "ordered", "ordering", "food", "foods", "jollibee",
    "jolibee", "jollibe", "jabee", "delivery", "deliver", "delivered",
    "foodpanda", "panda", "branch", "store", "restaurant", "meal", "meals",
    "item", "items", "time", "times", "got", "get", "just", "like", "really",
    "please", "pls", "plss", "sana", "ok", "okay", "ung", "nyo", "niyo", "po", "nag",
    "did", "does", "super", "sobrang", "ave",
} | NEGATIONS

# Items customers report as missing; shared by several MISSING_ITEM patterns.
ITEMS = r"""(gravy|gravi|rice|kanin|kan-?on|drinks?|fries|spaghetti|spag|sauce|ketchup
    |catsup|utensils?|cutler(y|ies)|spoons?|forks?|tissues?|straws?|ice|ice\s*cream
    |sundaes?|coke|float|toy|burgers?|yumburger|burger\s*steak|chicken|chiken|nuggets
    |dessert|cheese|patty|pie|soup|macaroni|condiments?|hotdog|pancake|iced\s+tea)"""

THEMES = {
    "MISSING_ITEM": r"""
        \bmissing\b | \bkulang\b | \bincomplete\b | \black(ing)?\b
        | \bforg[eo]t | \bnakalimutan | \bnaiwan
        | \b(kuwang|kuwangan|kulangan|kompletohin|kumpleto)
        | \bwala\b | \bwala\s+(po|man|nga)\b | \bwalay\b
        | \b(no|walang?|wla|walay|wa)\s+(po(ng)?\s+|yung\s+|ung\s+|man\s+lang\s+|pang?\s+|na\s+|syay\s+)?""" + ITEMS + r"""
        | \b(doesn'?t|don'?t|didn'?t)\s+have\s+(any\s+|my\s+)?""" + ITEMS + r"""
        | \bonly\s+(\d+|one|two|three)\b | \bonly\s+(got|received|delivered|arrived|came)
        | \b(\d+|isa|usa)\s*(pcs?\s+)?(lang|ra)\b
        | \b(weren'?t|wasn'?t|not)\s+(been\s+)?includ | \bno\s+cutlery
        | \b(hindi|di|d|wala)\s+(po\s+)?(kasama|kasali|sinama|naisama)
        | \b(wala|hindi|di|d)\b.{0,25}\b(recieve|receive|natanggap|nadawat)
        | \b(didn'?t|did\s+not|never)\s+(receive|include|give|get|come|arrive)
        | \bnot\s+(included|received|delivered|complete)\b
        | \b(hindi|di|wala)\s+(dumating|nakarating|naibigay|binigay|kasama|sinama|nilagay)
        | \bnever\s+(arrived|came)\b
    """,
    "WRONG_ORDER": r"""
        \bwrong\b | \bmali\b | \bmaling\b | \bnot\s+what\s+i\s+order
        | \binstead\s+of\b | \bdifferent\s+(order|item|flavor|size)
        | \bibang\s+(order|item|flavor) | \bnapalitan\b | \bpinalitan\b
        | \b(didn'?t|did\s+not|don'?t|do\s+not|never|not)\s+follow
        | \b(hindi|di|wala)\s+(sinunod|sinusunod|sinundan)
        | \bignored?\s+(my\s+|the\s+|our\s+)?(request|instruction|note)
        | \brequest(ed)?\b.{0,40}\b(but|pero|yet|instead)\b
        | \binstructions?\b.{0,20}\b(not|ignored|wasn'?t)\b | \bincorrect\b
        | \b(request|requested|requesting|noted?|notes|asked\s+for|specifically|sabi\s+ko|ingnan)\b
        | \b(thighs?|breast|legs?|drumsticks?|paa|pecho|petcho)\b
        | \biba\s+(ang|yung|ung|pinadala|binigay|na)\b
        | \b(i|we)\s+ordered\b.{0,40}\b(but|pero)\b.{0,30}
          \b(received|got|gave|served|sent|delivered|dumating|binigay|pinadala|niabot)
        | \b(you|they)\s+(served|sent|gave)\b
    """,
    "LATE_DELIVERY": r"""
        \blate\b | \bdelay(ed|s)?\b | \btook\s+(so\s+|too\s+|very\s+)?long
        | \b(so|too|very)\s+long\b | \blong\s+(wait|time)\b | \bwait(ed|ing)\b
        | \b(an?|one|two|three|\d+)\s*(hours?|hrs?|oras)\b | \bhours?\b
        | \bslow\b | \bsluggish\b
        | \b(ang\s*|an|napaka|sobrang\s*)?tagal\b | \bmatagal\b | \bnatagalan\b
        | \bmabagal\b | \bbagal\b | \bdugay\b | \bdugaya\b | \bhinay\b
        | \bpinaghintay | \bhintay | \b\d+\s*(mins?|minutes)\b | \bminutes\b
        | \bto+\s+lo+ng\b | \btoo\s+much\s+time\b | \btook\s+time\b
        | \blong(er)?\s+(than|prep|preparing)\b | \bestimated\s+time\b
    """,
    "COLD_FOOD": r"""
        \bcold\b | \bmalamig\b | \blamig\b | \bbugnaw\b | \btugnaw\b
        | \b(not|hindi|di|dili|wala)\s+(na\s+)?(hot|warm|mainit|init|inet)\b
        | \bluke\s*warm\b | \broom\s+temp
    """,
    "FOOD_QUALITY": r"""
        \bsoggy\b | \bstale\b | \bundercooked\b | \bovercooked\b | \braw\b | \bhilaw\b
        | \bburnt?\b | \bsunog\b | \bnangingitim | \bmatigas\b | \btigas\b | \bhard\b
        | \bdry\b | \bdried\b | \brubbery\b | \boily\b | \bmamantika\b | \bmantika
        | \btasteless\b | \bbland\b | \bwalang\s+lasa | \bmatabang\b | \btabang\b
        | \bsalty\b | \bmaalat\b | \balat\b | \bpanis | \bspoil | \bsira\b | \bsour\b
        | \bmaasim\b | \bsmell | \bmabaho\b | \bbaho\b | \bnot\s+fresh\b | \breheat
        | \btast(e|es|ed|y)\s+(bad|weird|off|old|funny|different|awful|terrible)
        | \b(pangit|panget)\s+(ng\s+|ang\s+)?lasa | \b(di|dili|walay)\s+lami
        | \b(not|hindi|di)\s+(tasty|delicious|masarap|crispy|crunchy|malutong)
        | \brancid\b | \bwatery\b | \bbulok\b | \bexpired\b | \bweeks?\s+old\b
        | \bdisgusting\b | \bgross\b | \byuck | \bkadiri\b | \bnot\s+cooked
    """,
    "SMALL_PORTION": r"""
        \bsmall(er|est)?\b | \btiny\b | \bliit\b | \bmaliit\b | \bnapakaliit\b
        | \bapakaliit\b | \bang\s*liit\b | \bsobrang\s+(ma)?liit\b | \bgamay\b | \bkagamay\b | \bkonti\b
        | \bkaunti\b | \bonti\b | \bdiet\s+size\b | \bpayat\b | \bportions?\b
        | \bserving\b | \bskinny\b | \bthin\b | \bquantity\b
        | \b(back|ribs?|wings?|pakpak|likod)\s+part | \bpuro\s+(back|pakpak|wings?|likod|ribs?|buto|bones?)
        | \ball\s+(back|ribs?|wings?|bones?)\b | \bpuro\s+buto
    """,
    "STAFF_SERVICE": r"""
        \bstaff\b | \bcrew\b | \bcashier\b | \bmanager\b | \brude\b | \battitude\b
        | \bbastos\b | \bsuplad[ao]\b | \bservice\b | \bserbisyo\b | \bincompetent\b
        | \bunprofessional\b | \bcareless | \bpabaya\b | \bwalang\s+(pakialam|pake)
        | \b(no|not)\s+(response|reply|answer)| \bnot\s+answering\b
        | \bcancel+(ed|ing|led)?\b | \bunavailable\b | \bout\s+of\s+stock\b
        | \bnot\s+available\b | \bnotif(y|ied) | \bcoordinat
    """,
    "PACKAGING": r"""
        \bpackag | \bspill(ed|s|ing)?\b | \bspilt\b | \btapon\b | \bnatapon\b
        | \btumapon\b | \bleak | \btagas\b | \btumagas\b | \bcontainers?\b
        | \bbox(es)?\b | \blids?\b | \bsealed?\b | \bcrushed\b | \bsquish
        | \bnayupi\b | \byupi\b | \bdurog\b | \bsabog\b | \bnagkalat\b | \bmessy\b
        | \bnapisa\b | \bbasa\b | \bwet\b | \bcups?\b | \bbroken\b | \bseep
        | \blalagyan\b | \bkarton\b
    """,
    "HYGIENE": r"""
        \bhairs?\b | \bbuhok\b | \binsects?\b | \bipis\b | \bcockroach | \broach
        | \blanggam\b | \bants?\b | \bflies\b | \bfly\b | \blangaw\b | \bdirty\b
        | \bmadumi\b | \bmarumi\b | \bdumi\b | \bunsanitary\b | \bunhygienic\b
        | \bforeign\s+(object|matter) | \buod\b | \bworms?\b | \bmold | \bamag\b
        | \binsekto | \bbugs?\b
    """,
}
THEME_LABELS = {
    "MISSING_ITEM": "Missing item", "WRONG_ORDER": "Wrong order",
    "LATE_DELIVERY": "Late delivery", "COLD_FOOD": "Cold food",
    "FOOD_QUALITY": "Food quality", "SMALL_PORTION": "Small portion",
    "STAFF_SERVICE": "Staff & service", "PACKAGING": "Packaging",
    "HYGIENE": "Hygiene",
}
THEME_RE = {k: re.compile(v, re.VERBOSE) for k, v in THEMES.items()}
_ELONGATION_RE = re.compile(r"(.)\1{2,}")


def theme_text(text):
    """Lowercase and squash elongations to one char ("lateeee" -> "late") so the
    regexes match the dictionary spelling; ordinary double letters survive."""
    return _ELONGATION_RE.sub(r"\1", str(text).lower())


def is_display_term(term, extra_stop=frozenset()):
    words = term.split()
    if not any(c.isalpha() for c in term):
        return False  # emoji/number-only terms say nothing actionable
    if len(words) > 1 and words[-1] in NEGATIONS:
        # "chicken is not hot" -> "chicken not" once stopwords go; the negation
        # belongs to the next word, so a phrase ending on it is an artifact.
        return False
    return not all(w in DISPLAY_STOPWORDS or w in extra_stop for w in words)


def term_counts(texts, min_df, ngram_range=(1, 3), extra_stop=frozenset()):
    vec = CountVectorizer(preprocessor=normalize, tokenizer=tokenize, token_pattern=None,
                          lowercase=False, ngram_range=ngram_range, min_df=min_df)
    try:
        X = vec.fit_transform(texts)
    except ValueError:  # nothing reaches min_df
        return pd.DataFrame(columns=["term", "n_words", "n_reviews", "count"])
    df = pd.DataFrame({
        "term": vec.get_feature_names_out(),
        "n_reviews": (X > 0).sum(axis=0).A1,
        "count": X.sum(axis=0).A1,
    })
    df["n_words"] = df["term"].str.count(" ") + 1
    df = df[df["term"].map(lambda t: is_display_term(t, extra_stop))]
    return df.sort_values(["n_reviews", "count"], ascending=False)[
        ["term", "n_words", "n_reviews", "count"]]


def main():
    complaints = pd.read_csv(OUT / "jollibee_complaints.csv", encoding="utf-8-sig")
    texts = complaints["text_clean"].astype(str)
    n_total = len(complaints)
    print(f"ReviewLens complaint extraction - {n_total:,} negative reviews, "
          f"{complaints['Branch'].nunique()} branches\n")

    # ---------------------------------------------------- Level 1: chain terms
    # Branch names show up in complaints ("acacia lane branch") but aren't issues.
    name_words = lambda s: frozenset(tokenize(normalize(s)))
    all_branch_words = frozenset().union(*map(name_words, complaints["Branch"].unique()))
    chain = term_counts(texts, min_df=3, extra_stop=all_branch_words) \
        .head(TOP_N_CHAIN).reset_index(drop=True)
    chain.insert(0, "rank", range(1, len(chain) + 1))
    chain["pct_reviews"] = chain["n_reviews"] / n_total
    chain.to_csv(OUT / "complaint_terms_chain.csv", index=False)

    branch_terms = []
    for branch, grp in complaints.groupby("Branch"):
        t = term_counts(grp["text_clean"].astype(str), min_df=2, ngram_range=(2, 3),
                        extra_stop=name_words(branch))
        t = t.head(TOP_N_BRANCH).assign(Branch=branch, pct_reviews=lambda d: d.n_reviews / len(grp))
        branch_terms.append(t)
    pd.concat(branch_terms)[["Branch", "term", "n_words", "n_reviews", "count", "pct_reviews"]] \
        .to_csv(OUT / "complaint_terms_branch.csv", index=False)

    # ---------------------------------------------------- Level 2: themes
    norm = texts.map(theme_text)
    for theme, rx in THEME_RE.items():
        complaints[theme] = norm.map(lambda s, rx=rx: bool(rx.search(s)))
    theme_cols = list(THEMES)
    complaints["n_themes"] = complaints[theme_cols].sum(axis=1)

    chain_rate = complaints[theme_cols].mean()
    by_branch = complaints.groupby("Branch")
    hits = by_branch[theme_cols].sum()
    sizes = by_branch.size()
    rates = hits.div(sizes, axis=0)
    lift = rates.div(chain_rate, axis=1)

    long = (rates.stack().rename("rate").to_frame()
            .join(hits.stack().rename("n_hits"))
            .join(lift.stack().rename("lift"))
            .reset_index().rename(columns={"level_1": "theme"}))
    long["theme_label"] = long["theme"].map(THEME_LABELS)
    long["n_complaints"] = long["Branch"].map(sizes)
    long["chain_rate"] = long["theme"].map(chain_rate)
    long["notable"] = (long["lift"] > NOTABLE_LIFT) & (long["n_hits"] >= MIN_THEME_HITS)
    long["City"] = long["Branch"].map(by_branch["City"].first())
    long = long[["Branch", "City", "n_complaints", "theme", "theme_label", "n_hits",
                 "rate", "chain_rate", "lift", "notable"]]
    long.to_csv(OUT / "complaint_themes_branch.csv", index=False)

    priorities = []
    for branch, grp in long.groupby("Branch"):
        eligible = grp[grp["n_hits"] >= MIN_THEME_HITS].sort_values("lift", ascending=False)
        row = {"Branch": branch, "City": grp["City"].iat[0],
               "n_complaints": int(grp["n_complaints"].iat[0]),
               "share_of_chain_complaints": grp["n_complaints"].iat[0] / n_total}
        for i in (1, 2):
            r = eligible.iloc[i - 1] if len(eligible) >= i else None
            row[f"top_issue_{i}"] = r["theme_label"] if r is not None else None
            row[f"top_issue_{i}_rate"] = r["rate"] if r is not None else None
            row[f"top_issue_{i}_lift"] = r["lift"] if r is not None else None
            row[f"top_issue_{i}_notable"] = bool(r["notable"]) if r is not None else False
        priorities.append(row)
    priorities = pd.DataFrame(priorities).sort_values(
        ["top_issue_1_lift", "n_complaints"], ascending=False)
    priorities.to_csv(OUT / "branch_priorities.csv", index=False)

    # ---------------------------------------------------- summary
    print("Top 15 complaint terms chain-wide:")
    print(chain.head(15)[["rank", "term", "n_reviews"]].to_string(index=False))
    print("\nTheme coverage (share of all complaints):")
    for theme in theme_cols:
        print(f"  {THEME_LABELS[theme]:<16} {chain_rate[theme]:6.1%}  ({int(complaints[theme].sum()):,})")
    untagged = (complaints["n_themes"] == 0).mean()
    print(f"  {'(no theme)':<16} {untagged:6.1%}  - mostly short or generic complaints")
    print(f"\nBranches with a notable issue (lift > {NOTABLE_LIFT}, >= {MIN_THEME_HITS} complaints): "
          f"{int(priorities['top_issue_1_notable'].sum())} of {len(priorities)}")
    print(priorities.head(10)[["Branch", "n_complaints", "top_issue_1", "top_issue_1_lift",
                               "top_issue_2", "top_issue_2_lift"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nSaved to {OUT}: complaint_terms_chain.csv, complaint_terms_branch.csv, "
          "complaint_themes_branch.csv, branch_priorities.csv")


if __name__ == "__main__":
    main()
