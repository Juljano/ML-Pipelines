import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
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

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words= list(STOP_WORDS))),

        ("classifier", RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            class_weight="balanced",
            random_state=42))
    ])

    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    print(metrics.classification_report(y_test, y_pred))




if __name__ == "__main__":
    x,y = read_csv("C:/Users/JML/KI - Projekte/Reviews_AI/cleaned_data/trustpilot_reviews_with_ratings.csv")
    train_model(x,y)