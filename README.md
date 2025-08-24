# Przewidywanie Toksyczności Komentarzy - Kurs Machine Learning

## 📝 Opis projektu

Ten projekt pokazuje, jak stworzyć system automatycznego wykrywania toksyczności w komentarzach internetowych używając technik uczenia maszynowego. Program analizuje teksty i przewiduje poziom toksyczności w różnych kategoriach.

## 🎯 Co się nauczysz

- Jak ładować i przetwarzać zbiory danych tekstowych
- Jak przekształcać tekst na liczby (TF-IDF)
- Jak trenować model uczenia maszynowego
- Jak oceniać jakość modelu
- Jak używać modelu do przewidywań

## 📚 Wymagania

```bash
pip install datasets pandas scikit-learn
```

### Biblioteki używane w projekcie:

- **datasets** - ładowanie gotowych zbiorów danych
- **pandas** - manipulacja danymi
- **scikit-learn** - narzędzia do uczenia maszynowego

## 🔬 Krok po kroku - jak działa kod

### 1. Ładowanie danych

```python
dataset = load_dataset("google/civil_comments")
df = dataset["train"].to_pandas()
```

**Co się dzieje:**
- Pobieramy zbiór danych "Civil Comments" od Google
- Zawiera prawdziwe komentarze z internetu z ocenami ekspertów
- Konwertujemy na format pandas DataFrame dla łatwiejszej pracy

### 2. Przygotowanie etykiet

```python
labels = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
X = df["text"]  # Teksty komentarzy (dane wejściowe)
y = df[labels]  # Oceny toksyczności (dane wyjściowe)
```

**Rodzaje toksyczności:**
- `toxicity` - ogólna toksyczność
- `severe_toxicity` - poważna toksyczność  
- `obscene` - wulgarność
- `threat` - groźby
- `insult` - obelgi
- `identity_attack` - ataki na tożsamość
- `sexual_explicit` - treści seksualne

### 3. Podział danych

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Dlaczego dzielimy dane:**
- **80% na trening** - do uczenia modelu
- **20% na test** - do sprawdzenia, czy model działa na nowych danych
- `random_state=42` - zapewnia powtarzalne wyniki

### 4. Przekształcanie tekstu na liczby (TF-IDF)

```python
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

**Co to jest TF-IDF:**
- **TF (Term Frequency)** - jak często słowo pojawia się w dokumencie
- **IDF (Inverse Document Frequency)** - jak rzadkie jest słowo w całym zbiorze
- **Efekt:** Ważne słowa mają wyższą wartość, popularne słowa (jak "i", "a") mają niższą

**Przykład:**
- Komentarz: "Ten film jest okropny"
- TF-IDF zamienia to na wektor liczb: [0.0, 0.3, 0.0, 0.8, 0.5, ...]
- Każda pozycja odpowiada jednemu słowu ze słownika

### 5. Trenowanie modelu

```python
model = LinearRegression()
model.fit(X_train_tfidf, y_train)
```

**Regresja liniowa:**
- Znajduje liniową zależność między słowami a poziomem toksyczności
- Dla każdego słowa przypisuje wagę (dodatnią lub ujemną)
- Słowa jak "stupid", "hate" dostaną wysokie wagi dodatnie
- Słowa jak "love", "thank you" dostaną wagi ujemne

### 6. Ocena modelu

```python
y_pred = model.predict(X_test_tfidf)
print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
print(f"R2 score: {r2_score(y_test, y_pred)}")
```

**Metryki:**
- **MSE (Mean Squared Error)** - średni błąd kwadratowy
  - Im mniejszy, tym lepiej
  - Pokazuje, jak bardzo nasze przewidywania odbiegają od rzeczywistości
- **R² Score** - współczynnik determinacji  
  - Wartości od 0 do 1 (może być ujemny dla bardzo złych modeli)
  - Im bliżej 1, tym lepiej model wyjaśnia dane

## 🚀 Jak uruchomić program

```bash
python main.py
```

**Co zobaczyć:**
1. Informacje o zbiorze danych
2. Metryki jakości modelu
3. Listę etykiet
4. Testy na przykładowych komentarzach

## 💡 Przykłady użycia

Program automatycznie testuje 3 komentarze:

### Test 1: "This is a terrible comment."
- **Oczekiwany wynik:** średnia toksyczność
- **Dlaczego:** słowo "terrible" ma negatywną konotację

### Test 2: "This is a very nice comment. Thank you!"
- **Oczekiwany wynik:** niska toksyczność  
- **Dlaczego:** pozytywne słowa jak "nice", "thank you"

### Test 3: "I want to harm you!"
- **Oczekiwany wynik:** wysoka toksyczność
- **Dlaczego:** jawna groźba

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

## 🔧 Jak testować własne komentarze

Dodaj na końcu pliku `main.py`:

```python
# Testuj własne komentarze
moj_komentarz = "Wpisz tutaj swój komentarz"
wynik = get_comment_rating(moj_komentarz)
print(f"Wyniki dla '{moj_komentarz}':")
for i, etykieta in enumerate(labels):
    print(f"{etykieta}: {wynik[i]:.3f}")
```

## 📈 Jak ulepszyć model

### 1. Lepsze przetwarzanie tekstu
```python
# Dodaj więcej funkcji TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,  # więcej słów
    ngram_range=(1, 2),  # używaj par słów
    min_df=2,           # ignoruj bardzo rzadkie słowa
    stop_words='english' # usuń stop words
)
```

### 2. Inne modele
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

## 🎓 Koncepty ML wyjaśnione

### Co to jest supervised learning?
- Mamy dane wejściowe (komentarze) i oczekiwane wyniki (oceny toksyczności)
- Model uczy się na przykładach z prawidłowymi odpowiedziami
- Potem może przewidywać dla nowych danych

### Dlaczego dzielimy dane na train/test?
- **Overfitting** - model może "zapamiętać" dane treningowe
- Test na oddzielnych danych pokazuje rzeczywistą jakość
- Jak egzamin - nie można się uczyć z pytań egzaminacyjnych

### Czym różni się regresja od klasyfikacji?
- **Klasyfikacja:** przewiduje kategorie (spam/nie spam)
- **Regresja:** przewiduje liczby (poziom toksyczności od 0 do 1)
- Używamy regresji, bo toksyczność to wartość ciągła

## ⚠️ Ograniczenia

1. **Język:** Model trenowany na języku angielskim
2. **Kontekst:** Może nie rozumieć sarkazmu czy ironii  
3. **Stronniczość:** Może mieć uprzedzenia ze zbioru treningowego
4. **Prosty model:** Regresja liniowa ma ograniczenia

## 🔄 Następne kroki

1. **Spróbuj innych modeli:** Random Forest, Neural Networks
2. **Lepsze preprocessing:** stemming, lemmatizacja
3. **Więcej danych:** użyj większego zbioru
4. **Głębokie uczenie:** BERT, transformers
5. **Ewaluacja:** więcej metryk, confusion matrix

## 📖 Dodatkowe materiały

- [Scikit-learn documentation](https://scikit-learn.org/)
- [TF-IDF wyjaśnienie](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Civil Comments Dataset](https://www.tensorflow.org/datasets/catalog/civil_comments)

---

**Gratulacje!** 🎉 Właśnie stworzyłeś swój pierwszy model do analizy sentymentu. To podstawy, które możesz rozwijać w bardziej zaawansowanych projektach ML.