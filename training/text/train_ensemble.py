"""Train the NewsXpose text ensemble without committing the dataset."""

from argparse import ArgumentParser
from pathlib import Path
import re

import joblib
import nltk
import pandas as pd
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def stem_text(content: str) -> str:
    words = re.sub(r"[^a-zA-Z]", " ", content).lower().split()
    return " ".join(
        STEMMER.stem(word) for word in words if word not in STOPWORDS
    )


def load_dataset(dataset_path: Path, row_limit: int | None = None) -> pd.DataFrame:
    news = pd.read_csv(dataset_path, nrows=row_limit).fillna(" ")
    news["content"] = news["title"] + " " + news["text"]
    news["content"] = Parallel(n_jobs=-1)(
        delayed(stem_text)(text) for text in news["content"]
    )
    return news


def train(dataset_path: Path, output_path: Path, row_limit: int | None = None) -> None:
    news = load_dataset(dataset_path, row_limit)
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(news["content"].values)
    labels = news["label"].values

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=2,
    )

    model = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=2)),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=2,
                ),
            ),
            ("lr", LogisticRegression(C=1.0, random_state=2)),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=2,
                ),
            ),
            (
                "ada",
                AdaBoostClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=2,
                ),
            ),
        ],
        voting="soft",
    )
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test)) * 100
    print(f"Ensemble test accuracy: {accuracy:.2f}%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((model, vectorizer), output_path)
    print(f"Saved model to {output_path}")


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=repo_root / "data" / "WELFake_Dataset.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "ensemble_fake_news_detector.joblib",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Optional row limit for a quick training run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments.dataset, arguments.output, arguments.rows)
