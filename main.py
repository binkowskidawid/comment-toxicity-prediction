# PRZEWIDYWANIE TOKSYCZNOŚCI KOMENTARZY
# Program wykorzystuje uczenie maszynowe do przewidywania poziomu toksyczności w komentarzach
# Z AUTOMATYCZNYM ZAPISYWANIEM I ŁADOWANIEM MODELU!

# Importowanie bibliotek potrzebnych do pracy z danymi i uczeniem maszynowym
from datasets import load_dataset  # Biblioteka do ładowania gotowych zbiorów danych
import pandas as pd  # Biblioteka do manipulacji danymi (DataFrame, analiza danych)
from sklearn.model_selection import train_test_split  # Funkcja do podziału danych na zbiór treningowy i testowy
from sklearn.feature_extraction.text import TfidfVectorizer  # Przekształca tekst na liczby (wektory TF-IDF)
from sklearn.linear_model import LinearRegression  # Model regresji liniowej do przewidywania
from sklearn.metrics import mean_squared_error, r2_score  # Metryki do oceny jakości modelu
import joblib  # Najlepsza biblioteka do zapisywania modelów scikit-learn (szybsza niż pickle)
import os.path  # Do sprawdzania czy pliki istnieją

# Definiujemy ścieżki do plików, w których będziemy zapisywać wytrenowany model i vectorizer
MODEL_FILE = "model.joblib"  # Plik z wytrenowanym modelem regresji liniowej
VECTORIZER_FILE = "vectorizer.joblib"  # Plik z wytrenowanym vectorizerem TF-IDF

# Definiujemy etykiety (labels) - różne rodzaje toksyczności, które chcemy przewidywać
labels = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]


# FUNKCJE DO ZARZĄDZANIA MODELEM

def models_exist():
    """Sprawdza czy zapisane pliki modelu i vectorizera istnieją na dysku"""
    # os.path.exists() zwraca True jeśli plik istnieje, False jeśli nie
    model_exists = os.path.exists(MODEL_FILE)
    vectorizer_exists = os.path.exists(VECTORIZER_FILE)
    # Oba pliki muszą istnieć, żeby móc załadować kompletny model
    return model_exists and vectorizer_exists


def save_model_and_vectorizer(model, vectorizer):
    """Zapisuje wytrenowany model i vectorizer do plików .joblib"""
    print("\n💾 Zapisywanie modelu i vectorizera na dysk...")
    # joblib.dump() zapisuje obiekty do pliku - szybsze i bardziej efektywne niż pickle dla numpy
    joblib.dump(model, MODEL_FILE)  # Zapisujemy model regresji liniowej
    joblib.dump(vectorizer, VECTORIZER_FILE)  # Zapisujemy vectorizer TF-IDF
    print(f"✅ Model zapisany w: {MODEL_FILE}")
    print(f"✅ Vectorizer zapisany w: {VECTORIZER_FILE}")
    print("\nKolejne uruchomienia będą znacznie szybsze! ⚡")


def load_model_and_vectorizer():
    """Wczytuje zapisany model i vectorizer z plików .joblib"""
    print("\n📦 Ładowanie zapisanego modelu z dysku...")
    # joblib.load() wczytuje obiekty z pliku
    model = joblib.load(MODEL_FILE)  # Wczytujemy zapisany model
    vectorizer = joblib.load(VECTORIZER_FILE)  # Wczytujemy zapisany vectorizer
    print("✅ Model i vectorizer załadowane pomyślnie!")
    print("⚡ Pominięto trening - używamy gotowego modelu!")
    return model, vectorizer


def train_new_model():
    """Trenuje nowy model od zera - wywoływane tylko gdy brak zapisanych plików"""
    print("\n🏃 Rozpoczynanie treningu nowego modelu...")
    print("To może potrwać kilka minut - następne uruchomienia będą szybsze!")
    
    # Ładowanie zbioru danych "civil_comments" od Google - zawiera komentarze z ocenami toksyczności
    print("\n1️⃣ Ładowanie danych z internetu...")
    dataset = load_dataset("google/civil_comments")
    # Konwertowanie zbioru treningowego na DataFrame pandas dla łatwiejszej manipulacji
    df = dataset["train"].to_pandas()
    
    # Wyświetlenie podstawowych informacji o zbiorze danych (ile wierszy, jakie kolumny)
    print(f"✅ Załadowano {len(df)} komentarzy do analizy")
    
    # X to dane wejściowe - teksty komentarzy (cechy/features)
    X = df["text"]
    # y to dane wyjściowe - oceny toksyczności dla każdej kategorii (target/etykiety)
    y = df[labels]
    
    # Podział danych na zbiór treningowy (80%) i testowy (20%)
    print("\n2️⃣ Podział danych na trening i test...")
    # X_train, y_train - dane do uczenia modelu
    # X_test, y_test - dane do testowania jakości modelu
    # random_state=42 zapewnia powtarzalne wyniki
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Dane treningowe: {len(X_train)} komentarzy")
    print(f"✅ Dane testowe: {len(X_test)} komentarzy")

    # Przykład zastosowania TF-IDF dla języka polskiego - EDUKACYJNE WYJAŚNIENIE
    # Załóżmy, że mamy dwa zdania w języku polskim:
    # "Kot biega po ogrodzie." vs "Kotka biega po ogrodzie."
    # TF-IDF najpierw tokenizuje tekst: ["Kot", "biega", "po", "ogrodzie"] vs ["Kotka", "biega", "po", "ogrodzie"]
    # Problem: "Kot" i "Kotka" to różne tokeny, ale to samo znaczenie
    # Rozwiązanie: lematyzacja sprowadza do: ["Kot", "biegać", "po", "ogród"]
    
    print("\n3️⃣ Tworzenie vectorizera TF-IDF...")
    # Tworzenie wektoryzera TF-IDF - przekształca tekst na liczby
    # max_features=5000 oznacza, że używamy tylko 5000 najważniejszych słów
    vectorizer = TfidfVectorizer(max_features=5000)
    # Uczenie wektoryzera na danych treningowych i przekształcanie tekstu na wektory liczb
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # Przekształcanie danych testowych używając już nauczonego wektoryzera
    X_test_tfidf = vectorizer.transform(X_test)
    print("✅ Vectorizer TF-IDF wytrenowany!")
    
    print("\n4️⃣ Trenowanie modelu regresji liniowej...")
    # Tworzenie modelu regresji liniowej - znajdzie liniową zależność między słowami a toksycznością
    model = LinearRegression()
    # Uczenie modelu na danych treningowych (wektory TF-IDF + etykiety toksyczności)
    model.fit(X_train_tfidf, y_train)
    print("✅ Model regresji liniowej wytrenowany!")
    
    print("\n5️⃣ Testowanie jakości modelu...")
    # Przewidywanie poziomów toksyczności dla danych testowych
    y_pred = model.predict(X_test_tfidf)
    # Mean Squared Error - średni błąd kwadratowy (im mniejszy, tym lepiej)
    mse = mean_squared_error(y_test, y_pred)
    # R2 score - współczynnik determinacji (im bliżej 1, tym lepiej model wyjaśnia dane)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean squared error: {mse:.4f}")
    print(f"R2 score: {r2:.4f}")
    
    # Zapisz wytrenowany model i vectorizer do plików
    save_model_and_vectorizer(model, vectorizer)
    
    return model, vectorizer


# GŁÓWNA LOGIKA PROGRAMU - SPRAWDZANIE CZY MODEL ISTNIEJE
print("="*60)
print("🤖 SYSTEM PRZEWIDYWANIA TOKSYCZNOŚCI KOMENTARZY")
print("="*60)

# Sprawdzamy czy zapisane pliki modelu już istnieją na dysku
if models_exist():
    print("\n🔍 Znaleziono zapisane pliki modelu!")
    # Jeśli pliki istnieją, po prostu je wczytujemy (szybko!)
    model, vectorizer = load_model_and_vectorizer()
else:
    print("\n⚠️ Nie znaleziono zapisanych plików modelu.")
    print("Trenowanie nowego modelu...")
    # Jeśli plików nie ma, trenujemy nowy model (długo, ale tylko raz!)
    model, vectorizer = train_new_model()

print("\n" + "="*60)
print("🎉 MODEL GOTOWY DO UŻYCIA!")
print("="*60)


# FUNKCJA DO OCENY TOKSYCZNOŚCI POJEDYNCZEGO KOMENTARZA
def get_comment_rating(comment):
    """Analizuje pojedynczy komentarz i zwraca przewidywane poziomy toksyczności"""
    # Przekształcenie nowego komentarza na wektor TF-IDF używając załadowanego/wytrenowanego vectorizera
    comment_tfidf = vectorizer.transform([comment])
    # Przewidywanie poziomów toksyczności za pomocą załadowanego/wytrenowanego modelu
    # Zwracamy pierwszy (i jedyny) wynik - tablicę z 7 wartościami dla 7 etykiet
    return model.predict(comment_tfidf)[0]


# TESTOWANIE MODELU NA PRZYKŁADOWYCH KOMENTARZACH
print("\n📊 ROZPOCZYNANIE TESTÓW MODELU")
print("-" * 40)

# Wyświetlenie listy etykiet, aby wiedzieć, co oznacza każdy wynik
print("\n🏷️ Etykiety analizowane przez model:")
for i, label in enumerate(labels):
    print(f"{i}: {label}")
print("\n(Im wyższa wartość, tym większa toksyczność w danej kategorii)")

# TEST 1: Komentarz potencjalnie toksyczny
print("\n🔴 TEST 1: Komentarz negatywny")
new_comment = "This is a terrible comment."
result1 = get_comment_rating(new_comment)
print(f"Komentarz: '{new_comment}'")
print(f"Wszystkie wyniki: {result1}")
print(f"Główna toksyczność: {result1[0]:.3f}")

# TEST 2: Komentarz pozytywny (powinien mieć niską toksyczność)
print("\n🟢 TEST 2: Komentarz pozytywny")
new_comment = "This is a very nice comment. Thank you!"
result2 = get_comment_rating(new_comment)
print(f"Komentarz: '{new_comment}'")
print(f"Wszystkie wyniki: {result2}")
print(f"Główna toksyczność: {result2[0]:.3f}")

# TEST 3: Komentarz wyraźnie toksyczny z groźbą (powinien mieć wysoką toksyczność)
print("\n🔴 TEST 3: Komentarz z groźbą")
new_comment = "I want to harm you!"
result3 = get_comment_rating(new_comment)
print(f"Komentarz: '{new_comment}'")
print(f"Wszystkie wyniki: {result3}")
print(f"Główna toksyczność: {result3[0]:.3f}")

print("\n" + "="*60)
print("🎆 TESTY ZAKOŃCZONE POMYŚLNIE!")
print("Następne uruchomienie będzie niemal natychmiastowe dzięki zapisanemu modelowi.")
print("="*60)
