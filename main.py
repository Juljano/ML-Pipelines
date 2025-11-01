import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from spacy.lang.de.stop_words import STOP_WORDS


def read_csv(path):
    try:
        df = pd.read_csv(path)
        x = df["review_body"]
        y = df["review_rating"] = df["review_rating"].replace({2: 1})
        y = df["review_rating"] = df["review_rating"].replace({4: 5})

        return x, y
    except FileNotFoundError:
        print("Die CSV konnte nicht gefunden werden")

def train_model(x,y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    pipelines = {
        "logreg": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 3))),
            ("clf", LogisticRegression(max_iter=500))
        ]),

        "svc": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 3))),
            ("clf", LinearSVC())
        ]),

        "rf": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000)),
            ("clf", RandomForestClassifier())
        ])
    }

    for name, pipe in pipelines.items():
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        print(name, metrics.f1_score(y_test, pred, average="macro"))


if __name__ == "__main__":
    x,y = read_csv("C:/Users/JML/KI - Projekte/Reviews_AI/cleaned_data/trustpilot_reviews_with_ratings.csv")
    train_model(x,y)