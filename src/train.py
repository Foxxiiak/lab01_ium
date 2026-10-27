
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Wczytanie ustawien z pliku .env
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "iris")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Wczytanie gotowego zbioru danych Iris
if DATA_PATH == "iris":
    dataset = load_iris()
else:
    raise ValueError("Obecnie obslugiwany jest zbior iris (DATA_PATH=iris).")

# Zamiana danych na tabele pandas
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df["target"] = dataset.target

print("Pierwsze 5 wierszy:")
print(df.head())

print("\nRozmiar zbioru:")
print(df.shape)

print("\nStatystyki opisowe:")
print(df.describe())

# Oddzielenie cech od odpowiedzi, ktorej uczy sie model
X = df.drop(columns=["target"])
y = df["target"]

# Podzial na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

print("\nLiczba probek treningowych:", len(X_train))
print("Liczba probek testowych:", len(X_test))

# Trenowanie drzewa decyzyjnego
model = DecisionTreeClassifier(random_state=RANDOM_SEED)
model.fit(X_train, y_train)

# Sprawdzenie jak dobrze dziala model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    target_names=dataset.target_names
)

print("\nDokladnosc modelu (accuracy):", accuracy)
print("\nRaport klasyfikacji:")
print(report)

# Zapis wynikow do pliku
results_path = Path("wyniki.txt")
with results_path.open("w", encoding="utf-8") as file:
    file.write("Wyniki modelu - zbior Iris\n")
    file.write(f"Accuracy: {accuracy:.4f}\n\n")
    file.write("Raport klasyfikacji:\n")
    file.write(report)

print(f"\nWyniki zapisano do pliku: {results_path}")
