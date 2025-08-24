# System Wykrywania Toksyczności - Kurs Machine Learning

## 📝 Opis projektu

Ten projekt pokazuje, jak stworzyć automatyczny system wykrywania toksyczności w komentarzach internetowych przy użyciu technik uczenia maszynowego. Program analizuje teksty i przewiduje poziomy toksyczności w różnych kategoriach.

**✨ FUNKCJE:** Modularna architektura z oddzielnymi komponentami trenowania, testowania i analizy dla profesjonalnego przepływu pracy rozwoju!

## 🎯 Czego się nauczysz

- Jak ładować i przetwarzać zbiory danych tekstowych
- Jak konwertować tekst na liczby (TF-IDF)
- Jak trenować modele uczenia maszynowego
- Jak oceniać jakość modelu
- Jak używać modeli do przewidywań
- **Jak zapisywać i ładować wytrenowane modele (optymalizacja)**
- **Profesjonalna struktura projektów Python**
- **Zasady programowania modularnego**

## 🏗️ Struktura projektu

```
PODSTAWY/
├── main.py                 # Interaktywny analizator komentarzy
├── train_model.py         # Samodzielne trenowanie modelu
├── test_model.py          # Testowanie i ocena
├── config.py              # Konfiguracja i stałe
├── model_utils.py         # Narzędzia zapisywania/ładowania modelu
├── text_processing.py     # Funkcje przetwarzania tekstu
├── README.md              # Dokumentacja angielska
├── README_PL.md           # Dokumentacja polska (ten plik)
├── requirements.txt       # Zależności projektu
├── pyproject.toml         # Nowoczesna konfiguracja projektu Python
├── uv.lock                # Plik blokady zależności
└── models/                # Zapisane pliki modelu
    ├── model.joblib
    └── vectorizer.joblib
```

## 📚 Wymagania

```bash
pip install -r requirements.txt
```

### Biblioteki używane w projekcie:

- **datasets** - ładowanie gotowych zbiorów danych
- **pandas** - manipulacja danymi
- **scikit-learn** - narzędzia uczenia maszynowego
- **joblib** - zapisywanie i ładowanie modeli (wbudowana w scikit-learn)
- **numpy** - obliczenia numeryczne
- **scipy** - obliczenia naukowe

## 🚀 Przewodnik szybkiego startu

### Krok 1: Zainstaluj zależności
```bash
pip install -r requirements.txt
```

### Krok 2: Wytrenuj model (tylko pierwszy raz)
```bash
python train_model.py
```
*To zajmuje 5-10 minut, ale trzeba to zrobić tylko raz*

### Krok 3: Rozpocznij interaktywną analizę
```bash
python main.py
```
*Natychmiastowe uruchomienie - analizuj komentarze w czasie rzeczywistym!*

### Krok 4: Uruchom testy (opcjonalnie)
```bash
python test_model.py
```

## 💾 Automatyczne zapisywanie modelu

### ⚡ Szybkość uruchomienia

**Pierwsze trenowanie (`python train_model.py`) ~5-10 minut:**
1. 🔄 Ładowanie danych z internetu
2. 🧠 Trenowanie modelu regresji liniowej
3. 📊 Testowanie jakości modelu
4. 💾 Automatyczne zapisanie modelu na dysk

**Kolejne analizy (`python main.py`) ~2-5 sekund:**
1. ✅ Odnajdywanie zapisanych plików
2. ⚡ Błyskawiczne ładowanie modelu
3. 🚀 Natychmiastowa gotowość do analizy

### 📁 Pliki modelu

System automatycznie tworzy dwa pliki w katalogu `models/`:

- **`model.joblib`** - wytrenowany model regresji liniowej
- **`vectorizer.joblib`** - wytrenowany wektorizer TF-IDF

**⚠️ Ważne:** Oba pliki są potrzebne do działania systemu. Nie usuwaj ich!

### 🔄 Ponowne trenowanie modelu

Aby wytrenować nowy model od początku:
1. Usuń pliki `model.joblib` i `vectorizer.joblib` z katalogu `models/`
2. Uruchom `python train_model.py` - automatycznie wytrenuje nowy model

## 🔬 Jak to działa - krok po kroku

System używa **inteligentnej architektury** - każdy komponent ma określony cel!

### 1. 🧠 Pipeline trenowania (`train_model.py`)

```python
# Kompletny przepływ pracy trenowania
def train_toxicity_model():
    X, y = load_training_data()                    # Ładowanie zbioru civil_comments
    X_train, X_test, y_train, y_test = split_data(X, y)  # 80% trening, 20% test
    vectorizer = create_vectorizer()               # Tworzenie procesora TF-IDF
    X_train_tfidf = vectorizer.fit_transform(X_train)    # Konwersja tekstu na liczby
    model = LinearRegression()                     # Tworzenie modelu
    model.fit(X_train_tfidf, y_train)            # Trenowanie modelu
    evaluate_model(model, X_test_tfidf, y_test)   # Test wydajności
    save_model_and_vectorizer(model, vectorizer)  # Zapis do późniejszego użycia
```

**Co się dzieje:**
- Pobiera zbiór danych Civil Comments od Google (prawdziwe komentarze internetowe z ocenami ekspertów)
- Konwertuje do pandas DataFrame dla łatwiejszej manipulacji
- Dzieli na zbiory treningowe (80%) i testowe (20%)
- Tworzy wektory TF-IDF z tekstu
- Trenuje model regresji liniowej
- Ocenia wydajność i zapisuje model

### 2. 📊 Pipeline analizy (`main.py`)

```python
# Interaktywny przepływ pracy analizy
def interactive_comment_analyzer():
    model, vectorizer = load_model_and_vectorizer()  # Ładowanie pre-trenowanego modelu
    while True:
        comment = input("Wprowadź komentarz: ")           # Pobieranie danych od użytkownika
        results = get_comment_rating(comment, model, vectorizer)  # Analiza
        display_results(results)                     # Wyświetlanie wyników toksyczności
```

**Co się dzieje:**
- Ładuje pre-trenowany model natychmiastowo (bez potrzeby trenowania)
- Przetwarza wprowadzenie użytkownika przez wektorizer TF-IDF
- Uzyskuje przewidywania toksyczności dla 7 kategorii
- Wyświetla wyniki z interpretacjami

### 3. 🧪 Pipeline testowania (`test_model.py`)

```python
# Kompleksowy przepływ pracy testowania
def test_predefined_comments():
    model, vectorizer = load_model_and_vectorizer()  # Ładowanie modelu
    for test_comment in TEST_COMMENTS:              # Test predefiniowanych przykładów
        results = get_comment_rating(test_comment, model, vectorizer)
        analyze_results(results)                     # Porównanie z oczekiwaniami
```

**Co się dzieje:**
- Testuje model na predefiniowanych komentarzach o znanym oczekiwanym zachowaniu
- Zapewnia interaktywny tryb testowania dla niestandardowych komentarzy
- Wyświetla szczegółowe podziały kategorii toksyczności

## 🏷️ Kategorie toksyczności

Model analizuje **7 typów toksyczności**:

| Kategoria | Opis |
|----------|------|
| `toxicity` | Ogólny poziom toksyczności |
| `severe_toxicity` | Poważny poziom toksyczności |
| `obscene` | Wulgarny język |
| `threat` | Groźby i zastraszanie |
| `insult` | Obelgi i ataki osobiste |
| `identity_attack` | Ataki na tożsamość |
| `sexual_explicit` | Treści o charakterze seksualnym |

## 🔧 Dokumentacja modułów

### `config.py` - Zarządzanie konfiguracją
Zawiera wszystkie stałe, ścieżki plików i parametry modelu:
```python
MODEL_FILE = "models/model.joblib"
LABELS = ["toxicity", "severe_toxicity", ...]
MAX_FEATURES = 5000
```

### `model_utils.py` - Zarządzanie modelem
Funkcje do zapisywania, ładowania i sprawdzania modeli:
```python
models_exist() -> bool                    # Sprawdź czy modele istnieją
save_model_and_vectorizer(model, vec)     # Zapisz wytrenowane modele
load_model_and_vectorizer() -> tuple      # Załaduj zapisane modele
```

### `text_processing.py` - Przetwarzanie tekstu
Wektoryzacja TF-IDF i analiza komentarzy:
```python
create_vectorizer() -> TfidfVectorizer    # Utwórz procesor TF-IDF
get_comment_rating(comment, model, vec)   # Analizuj pojedynczy komentarz
batch_analyze_comments(comments, ...)     # Analizuj wiele komentarzy
```

## 📊 Zrozumienie TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** konwertuje tekst na liczby:

- **TF (Term Frequency)** - jak często słowo pojawia się w dokumencie
- **IDF (Inverse Document Frequency)** - jak rzadkie jest słowo w całym zbiorze
- **Efekt:** Ważne słowa otrzymują wyższe wartości, częste słowa niższe

**Przykład:**
- Komentarz: "Ten film jest okropny"
- TF-IDF konwertuje na wektor: [0.0, 0.3, 0.0, 0.8, 0.5, ...]
- Każda pozycja odpowiada jednemu słowu w słowniku

## 📈 Metryki wydajności modelu

### Mean Squared Error (MSE)
- Mierzy średnią kwadratową różnicę między przewidywaniami a rzeczywistymi wartościami
- **Niższe jest lepsze** - pokazuje jak daleko są nasze przewidywania od rzeczywistości

### R² Score (Współczynnik determinacji)
- Wartości od 0 do 1 (mogą być ujemne dla bardzo złych modeli)
- **Bliższe 1 jest lepsze** - pokazuje jak dobrze model wyjaśnia dane
- > 0.7 = Doskonały, > 0.5 = Dobry, > 0.3 = Umiarkowany

## 🧪 Przykłady testowe

System automatycznie testuje te scenariusze:

### Test 1: "This is a terrible comment."
- **Oczekiwany rezultat:** Średnia toksyczność
- **Powód:** Zawiera negatywny język

### Test 2: "This is a very nice comment. Thank you!"
- **Oczekiwany rezultat:** Niska toksyczność
- **Powód:** Zawiera pozytywne słowa

### Test 3: "I want to harm you!"
- **Oczekiwany rezultat:** Wysoka toksyczność
- **Powód:** Jawna groźba

## 📊 Interpretacja wyników

Każdy komentarz otrzymuje 7 ocen (po jednej dla każdej etykiety):

```python
# Przykładowy wynik:
[0.1, 0.05, 0.02, 0.8, 0.1, 0.03, 0.01]
#  |     |     |    |    |     |     |
#  |     |     |    |    |     |     └─ sexual_explicit: 0.01 (bardzo niska)
#  |     |     |    |    |     └─ identity_attack: 0.03 (niska) 
#  |     |     |    |    └─ insult: 0.1 (niska)
#  |     |     |    └─ threat: 0.8 (wysoka!) 
#  |     |     └─ obscene: 0.02 (bardzo niska)
#  |     └─ severe_toxicity: 0.05 (niska)
#  └─ toxicity: 0.1 (niska)
```

**Interpretacja:**
- Wartości bliskie 0: niska toksyczność
- Wartości bliskie 1: wysoka toksyczność  
- W przykładzie: komentarz ma wysoką ocenę za "threat" (groźbę)

## 💡 Przykłady użycia

### Interaktywna analiza
```bash
$ python main.py
💬 Wprowadź komentarz do analizy: Hello everyone!

📊 WYNIKI ANALIZY TOKSYCZNOŚCI
Komentarz: 'Hello everyone!'
Ogólna toksyczność: 0.023

Szczegółowy podział:
        toxicity: 0.023 (BARDZO NISKA)
  severe_toxicity: 0.012 (BARDZO NISKA)
          obscene: 0.008 (BARDZO NISKA)
           threat: 0.015 (BARDZO NISKA)
           insult: 0.019 (BARDZO NISKA)
   identity_attack: 0.011 (BARDZO NISKA)
   sexual_explicit: 0.007 (BARDZO NISKA)

🎯 INTERPRETACJA: ✅ Bardzo niska toksyczność - komentarz wydaje się bezpieczny
```

### Testowanie wsadowe
```bash
$ python test_model.py
# Wybierz opcję 1 dla predefiniowanych testów
# Wybierz opcję 2 dla testowania niestandardowych komentarzy
# Wybierz opcję 3 dla trybu interaktywnego
```

## 🎓 Koncepty ML - wyjaśnienie

### Co to jest uczenie nadzorowane?
- Mamy dane wejściowe (komentarze) i oczekiwane wyniki (oceny toksyczności)
- Model uczy się na przykładach z prawidłowymi odpowiedziami
- Potem może przewidywać dla nowych, niewidzianych danych

### Dlaczego dzielimy dane na train/test?
- **Overfitting** - model może "zapamiętać" dane treningowe
- Test na oddzielnych danych pokazuje rzeczywistą jakość
- Jak egzamin - nie można się uczyć z pytań egzaminacyjnych

### Regresja vs Klasyfikacja?
- **Klasyfikacja:** przewiduje kategorie (spam/nie spam)
- **Regresja:** przewiduje liczby (poziom toksyczności od 0 do 1)
- Używamy regresji, bo toksyczność to wartość ciągła

## 📖 Funkcje zaawansowane

### Niestandardowa analiza komentarzy
```python
from text_processing import get_comment_rating
from model_utils import load_model_and_vectorizer

model, vectorizer = load_model_and_vectorizer()
result = get_comment_rating("Twój komentarz tutaj", model, vectorizer)
print(f"Wynik toksyczności: {result[0]}")
```

### Przetwarzanie wsadowe
```python
from text_processing import batch_analyze_comments

comments = ["Komentarz 1", "Komentarz 2", "Komentarz 3"]
results = batch_analyze_comments(comments, model, vectorizer)
```

## 🔄 Ulepszenia modelu

### 1. Lepsze przetwarzanie tekstu
```python
# Ulepszona konfiguracja TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,      # więcej słów
    ngram_range=(1, 2),      # używaj par słów
    min_df=2,               # ignoruj bardzo rzadkie słowa
    stop_words='english'    # usuń stop words
)
```

### 2. Alternatywne modele
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Random Forest - zwykle lepszy niż regresja liniowa
model = RandomForestRegressor(n_estimators=100)

# Support Vector Machine
model = SVR(kernel='rbf')
```

### 3. Walidacja krzyżowa
```python
from sklearn.model_selection import cross_val_score

# Sprawdź model na różnych podziałach danych
scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
print(f"Średni wynik: {scores.mean():.3f}")
```

## ⚠️ Ograniczenia

1. **Język:** Model trenowany na języku angielskim
2. **Kontekst:** Może nie rozumieć sarkazmu czy ironii
3. **Stronniczość:** Może mieć uprzedzenia ze zbioru treningowego
4. **Prosty model:** Regresja liniowa ma ograniczenia

## 🛠️ Rozwiązywanie problemów

**Problem:** "Nie znaleziono wytrenowanego modelu"
- **Rozwiązanie:** Uruchom `python train_model.py` najpierw

**Problem:** Program się zawiesza lub pokazuje błędy
- **Rozwiązanie:** Usuń pliki `.joblib` i wytrenuj model ponownie

**Problem:** Dziwne wyniki po aktualizacji kodu
- **Rozwiązanie:** Usuń stare pliki modelu, aby wytrenować świeży model

**Problem:** Niewystarczająca ilość miejsca na dysku
- **Rozwiązanie:** Pliki modelu zajmują ~50MB - sprawdź miejsce na dysku

## 🔄 Następne kroki

1. **Spróbuj innych modeli:** Random Forest, Neural Networks
2. **Lepsze preprocessing:** stemming, lemmatyzacja
3. **Więcej danych:** użyj większego zbioru
4. **Głębokie uczenie:** BERT, transformers
5. **Lepsza ewaluacja:** więcej metryk, confusion matrix

## 📖 Dodatkowe materiały

- [Dokumentacja Scikit-learn](https://scikit-learn.org/)
- [Wyjaśnienie TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Zbiór danych Civil Comments](https://www.tensorflow.org/datasets/catalog/civil_comments)
- [Dokumentacja angielska](README.md)

---

## 📄 Open Source

Ten projekt jest open source i dostępny do celów edukacyjnych i badawczych. Zapraszam do:

- 🔍 Studiowania kodu i technik uczenia maszynowego
- 🛠️ Modyfikowania i eksperymentowania z różnymi modelami
- 📚 Wykorzystania jako zasobu edukacyjnego do projektów ML
- 🤝 Współtworzenia ulepszeń i poprawek błędów
- 📖 Dzielenia się wiedzą i pomagania innym w nauce

**Współtworzenie:**
- Forkuj repozytorium
- Twórz gałęzie funkcji dla swoich zmian
- Przesyłaj pull requesty z jasnymi opisami
- Przestrzegaj istniejącego stylu kodu i standardów dokumentacji

**Użycie edukacyjne:**
Idealne do nauki klasyfikacji tekstu, wektoryzacji TF-IDF, zachowywania modeli i profesjonalnej struktury projektów Python.

## 📞 Wsparcie

- **Dokumentacja angielska:** [README.md](README.md)
- **Dokumentacja polska:** Ten plik
- **Problemy:** Najpierw sprawdź pliki modelu i zależności