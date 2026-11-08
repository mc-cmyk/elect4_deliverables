"""Shared text preprocessing for ReviewLens.

Kept in its own module so the tokenizer referenced inside best_model.joblib can
be imported by any script that loads the model (a function defined inside
train_grid.py's __main__ would not unpickle elsewhere).

Design notes
- No stemming: Porter rules strip English suffixes and mangle Tagalog affixes
  (na-, ma-, pag-), merging unrelated words.
- Negation words are never removed, even if a stopword list contains them,
  because "not good" / "hindi masarap" / "dili lami" flip polarity.
- Emoji are kept as their own tokens; they carry a lot of sentiment in short
  reviews.
"""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

NEGATIONS = frozenset({
    # Required by the project spec
    "not", "no", "never", "nothing", "without",
    "hindi", "di", "wala", "walang", "huwag", "wag",
    "dili", "way", "wa", "walay",
    # English negators that sklearn's stopword list would otherwise drop
    "nor", "none", "nobody", "neither", "nowhere", "cannot", "cant",
    "dont", "didnt", "doesnt", "isnt", "wasnt", "werent", "wont", "arent",
    "hnd", "wla",  # common Taglish spellings of hindi / wala
})

TAGALOG_STOPWORDS = {
    "ang", "ng", "mga", "sa", "na", "ni", "si", "sina", "nina", "kay", "kina",
    "ay", "at", "o", "ko", "mo", "ka", "ako", "ikaw", "siya", "sya", "kami",
    "tayo", "kayo", "sila", "ito", "iyan", "iyon", "yan", "yun", "yon", "yung",
    "ung", "nung", "nang", "pa", "po", "opo", "lang", "lng", "din", "rin",
    "daw", "raw", "naman", "nman", "nga", "kasi", "kc", "pero", "para", "pag",
    "kung", "nila", "namin", "natin", "ninyo", "nyo", "niya", "nya", "akin",
    "sakin", "sayo", "amin", "atin", "dito", "doon", "dun", "diyan", "jan",
    "tapos", "tas", "ba", "eh", "e", "pala", "ano", "kami", "mag", "may",
    "ung", "un", "ito", "itong", "yang", "yong", "ngayon", "lahat",
}

BISAYA_STOPWORDS = {
    "ang", "sa", "og", "ug", "nga", "ni", "si", "ko", "ka", "siya", "sila",
    "kami", "mi", "ta", "kita", "ato", "amo", "ila", "iya", "nako", "nimo",
    "niya", "ako", "imo", "akong", "imong", "iyang", "kay", "kaayo", "kaau",
    "kaayu", "man", "pud", "sad", "sab", "lang", "ra", "na", "pa", "diri",
    "didto", "ana", "kani", "kana", "kadto", "mao", "bitaw", "gyud", "jud",
    "gud", "bya", "ba", "dayon", "gani", "pero", "kung", "para", "ning",
    "kini", "kanang", "ani", "nya", "unya",
}

STOPWORDS = frozenset(
    (set(ENGLISH_STOP_WORDS) | TAGALOG_STOPWORDS | BISAYA_STOPWORDS) - NEGATIONS
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
# "didn't" -> "didnt" so the contraction survives as one negation token
_APOS_RE = re.compile(r"['’`]")
# Skin-tone modifiers, variation selectors and zero-width joiners: dropped so
# "👎🏾" and "👎" become the same token.
_EMOJI_MODIFIERS_RE = re.compile("[\U0001F3FB-\U0001F3FF︎️‍]")
_EMOJI = (
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs ext-A
    "☀-➿"          # misc symbols, dingbats (❤ ✨ ✔)
    "⭐⭕"
)
_TOKEN_RE = re.compile(rf"[a-z0-9ñ]+|[{_EMOJI}]")


def normalize(text):
    """Lowercase, drop URLs, collapse 3+ repeated characters to 2."""
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _REPEAT_RE.sub(r"\1\1", text)  # "soooo" -> "soo"
    return text


def tokenize(text):
    """Split normalized text into word and emoji tokens, dropping stopwords."""
    text = _APOS_RE.sub("", text)
    text = _EMOJI_MODIFIERS_RE.sub("", text)
    return [t for t in _TOKEN_RE.findall(text) if t not in STOPWORDS]
