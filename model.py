import pandas as pd
import joblib
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb

# Fungsi untuk preprocessing teks
def clean_and_stem(text):
    import re
    from nltk.tokenize import RegexpTokenizer
    text = text.lower()
    text = re.sub(r'https\S+|@\S+|#\S+|\'\w+|[^\w\s]', '', text)
    tokenizer = RegexpTokenizer('\w+')
    tokens = tokenizer.tokenize(text)
    stopwords_list = stopwords.words("indonesian")
    tokens = [word for word in tokens if word not in stopwords_list]
    stemmer = StemmerFactory().create_stemmer()
    return ' '.join(stemmer.stem(token) for token in tokens)

# Memuat dan memproses data
df = pd.read_csv("gabungan-semua.csv", encoding="latin-1")
df['Rating'] = df['Rating'].apply(lambda x: x.replace(',', '').replace('/5', '').strip()).astype(float).astype(int)
df.loc[df['Rating'] == 41, 'Rating'] = 4
df['Sentiment'] = df['Rating'].apply(lambda rating: 1 if rating > 3 else -1)
df['Processed_Review'] = df['Review'].apply(clean_and_stem)

# Menyimpan dataset yang telah diproses
df.to_csv("processed_reviews.csv", index=False)

# Melatih model
X_train, X_test, y_train, y_test = train_test_split(df['Processed_Review'], df['Sentiment'], test_size=0.1, random_state=3)
model_pipeline = make_pipeline_imb(
    TfidfVectorizer(max_df=0.5, min_df=2),
    SMOTE(),
    SVC(kernel='linear')
)
model_pipeline.fit(X_train, y_train)

# Menyimpan model yang telah dilatih
joblib.dump(model_pipeline, 'model_pipeline.pkl')
