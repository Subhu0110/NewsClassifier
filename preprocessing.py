import re
import html
import string


def preprocess_text(text):

    text = html.unescape(text)

    text = re.sub(r"#39;", "'", text)

    text = re.sub(r"\\[a-zA-Z]", " ", text)

    text = re.sub(r"\(Reuters\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(AP\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bReuters\s*-\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAP\s*-\s*", "", text, flags=re.IGNORECASE)

    text = text.lower()

    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)

    text = text.replace("-", " ")

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text