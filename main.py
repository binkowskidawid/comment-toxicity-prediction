# PRZEWIDYWANIE TOKSYCZNOŚCI KOMENTARZY
# Program wykorzystuje uczenie maszynowe do przewidywania poziomu toksyczności w komentarzach

# Importowanie bibliotek potrzebnych do pracy z danymi i uczeniem maszynowym
from datasets import load_dataset  # Biblioteka do ładowania gotowych zbiorów danych
import pandas as pd  # Biblioteka do manipulacji danymi (DataFrame, analiza danych)
from sklearn.model_selection import train_test_split  # Funkcja do podziału danych na zbiór treningowy i testowy
from sklearn.feature_extraction.text import TfidfVectorizer  # Przekształca tekst na liczby (wektory TF-IDF)
from sklearn.linear_model import LinearRegression  # Model regresji liniowej do przewidywania
from sklearn.metrics import mean_squared_error, r2_score  # Metryki do oceny jakości modelu

# Ładowanie zbioru danych "civil_comments" od Google - zawiera komentarze z ocenami toksyczności
dataset = load_dataset("google/civil_comments")
# Konwertowanie zbioru treningowego na DataFrame pandas dla łatwiejszej manipulacji
df = dataset["train"].to_pandas()

# Wyświetlenie podstawowych informacji o zbiorze danych (ile wierszy, jakie kolumny)
print(df)

# Definiujemy etykiety (labels) - różne rodzaje toksyczności, które chcemy przewidywać
labels = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
# X to dane wejściowe - teksty komentarzy (cechy/features)
X = df["text"]
# y to dane wyjściowe - oceny toksyczności dla każdej kategorii (target/etykiety)
y = df[labels]

# Podział danych na zbiór treningowy (80%) i testowy (20%)
# X_train, y_train - dane do uczenia modelu
# X_test, y_test - dane do testowania jakości modelu
# random_state=42 zapewnia powtarzalne wyniki
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Przykład zastosowania TF-IDF dla języka polskiego
# Załóżmy, że mamy dwa zdania w języku polskim:

# "Kot biega po ogrodzie."
# "Kotka biega po ogrodzie."

# Krok 1: Tokenizacja
# TF-IDF najpierw tokenizuje tekst, co oznacza, że każde słowo jest traktowane jako odrębna jednostka (token).
# Dla powyższych zdań, tokenizacja wyglądałaby następująco:
# Zdanie 1: ["Kot", "biega", "po", "ogrodzie"]
# Zdanie 2: ["Kotka", "biega", "po", "ogrodzie"]

# Krok 2: Obliczenie TF
# Obliczamy Term Frequency (TF) dla każdego słowa w zdaniu:
# Zdanie 1: "Kot"=1/4, "biega"=1/4, "po"=1/4, "ogrodzie"=1/4
# Zdanie 2: "Kotka"=1/4, "biega"=1/4, "po"=1/4, "ogrodzie"=1/4

# Krok 3: Obliczenie IDF
# Inverse Document Frequency (IDF) oblicza się na podstawie liczby dokumentów, w których dane słowo występuje.
# Zakładając że mamy tylko te dwa zdania
# "Kot" pojawia się w 1 z 2 dokumentów: IDF("Kot") = log(2/1) = 0.301
# "Kotka" pojawia się w 1 z 2 dokumentów: IDF("Kotka") = log(2/1) = 0.301
# "biega", "po", "ogrodzie" pojawiają się w obu dokumentach: IDF = log(2/2) = 0

# Krok 4: Obliczenie TF-IDF
# Mnożymy TF przez IDF dla każdego słowa:
# Zdanie 1: "Kot"=1/4 * 0.301, "biega"=1/4 * 0, "po"=1/4 * 0, "ogrodzie"=1/4 * 0
# Zdanie 2: "Kotka"=1/4 * 0.301, "biega"=1/4 * 0, "po"=1/4 * 0, "ogrodzie"=1/4 * 0

# Wynik
# Dla zdania 1 najwyższą wagę TF-IDF będzie miało słowo "Kot", a dla zdania 2 – "Kotka".
# Problem polega na tym, że "Kot" i "Kotka" to różne formy tego samego słowa, ale TF-IDF traktuje je jako różne terminy,
# co może prowadzić do gorszej analizy, jeśli nasz model nie rozpoznaje, że są to po prostu odmiany tego samego słowa.

# Rozwiązanie
# Aby poprawić działanie TF-IDF w języku polskim, możemy zastosować lematyzację (sprowadzenie słów do ich podstawowej formy).

# Zdanie 1 po lematyzacji: ["Kot", "biegać", "po", "ogród"]
# Zdanie 2 po lematyzacji: ["Kot", "biegać", "po", "ogród"]
# Po lematyzacji TF-IDF będzie działać skuteczniej, ponieważ "Kot" i "Kotka" zostaną sprowadzone do tej samej formy
# "Kot", co pozwoli modelowi lepiej rozpoznać, że oba zdania mówią o tym samym podmiocie.

# Tworzenie wektoryzera TF-IDF - przekształca tekst na liczby
# max_features=5000 oznacza, że używamy tylko 5000 najważniejszych słów
vectorizer = TfidfVectorizer(max_features=5000)
# Uczenie wektoryzera na danych treningowych i przekształcanie tekstu na wektory liczb
X_train_tfidf = vectorizer.fit_transform(X_train)
# Przekształcanie danych testowych używając już nauczonego wektoryzera
X_test_tfidf = vectorizer.transform(X_test)

# Tworzenie modelu regresji liniowej - znajdzie liniową zależność między słowami a toksycznością
model = LinearRegression()
# Uczenie modelu na danych treningowych (wektory TF-IDF + etykiety toksyczności)
model.fit(X_train_tfidf, y_train)

# Przewidywanie poziomów toksyczności dla danych testowych
y_pred = model.predict(X_test_tfidf)
# Mean Squared Error - średni błąd kwadratowy (im mniejszy, tym lepiej)
print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
# R2 score - współczynnik determinacji (im bliżej 1, tym lepiej model wyjaśnia dane)
print(f"R2 score: {r2_score(y_test, y_pred)}")


# Funkcja do oceny toksyczności pojedynczego komentarza
def get_comment_rating(comment):
    # Przekształcenie nowego komentarza na wektor TF-IDF używając nauczonego wektoryzera
    comment_tfidf = vectorizer.transform([comment])
    # Przewidywanie poziomów toksyczności i zwrócenie pierwszego (i jedynego) wyniku
    return model.predict(comment_tfidf)[0]


# Wyświetlenie listy etykiet, aby wiedzieć, co oznacza każdy wynik
print(labels)

# Test 1: Komentarz potencjalnie toksyczny
new_comment = "This is a terrible comment."
# Wyświetlenie wszystkich przewidywanych wartości toksyczności dla tego komentarza
print(get_comment_rating(new_comment))
# Wyświetlenie konkretnego wyniku dla pierwszej etykiety (toxicity)
print(f"Toxicity score for {new_comment}: {get_comment_rating(new_comment)[0]}")

# Test 2: Komentarz pozytywny (powinien mieć niską toksyczność)
new_comment = "This is a very nice comment. Thank you!"
# Wyświetlenie wszystkich przewidywanych wartości toksyczności
print(get_comment_rating(new_comment))
# Wyświetlenie wyniku dla głównej etykiety toksyczności
print(f"Toxicity score for {new_comment}: {get_comment_rating(new_comment)[0]}")

# Test 3: Komentarz wyraźnie toksyczny z groźbą (powinien mieć wysoką toksyczność)
new_comment = "I want to harm you!"
# Wyświetlenie wszystkich przewidywanych wartości toksyczności
print(get_comment_rating(new_comment))
# Wyświetlenie wyniku dla głównej etykiety toksyczności
print(f"Toxicity score for {new_comment}: {get_comment_rating(new_comment)[0]}")
