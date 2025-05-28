# spam_classifier/main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Wczytanie danych
print("[INFO] Wczytywanie danych...")
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', header=None, names=['label', 'message'])
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# 2. Podział na zbiory
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

# 3. Wektoryzacja tekstu
print("[INFO] Wektoryzacja TF-IDF...")
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Modele do porównania
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVC": SVC(),
    "MLP Classifier": MLPClassifier(max_iter=300)
}

results = {}

# 5. Trening i ewaluacja
for name, model in models.items():
    print(f"[INFO] Trening modelu {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# 6. Wykres porównania modeli
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.ylim(0.9, 1.0)
plt.ylabel("Accuracy")
plt.title("Porównanie modeli - klasyfikacja spamu")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# 7. Zapis najlepszego modelu
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

joblib.dump(best_model, "best_spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print(f"[INFO] Gotowe! Najlepszy model: {best_model_name} z accuracy {results[best_model_name]:.4f}")
