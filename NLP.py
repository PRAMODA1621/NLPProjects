
paragraph="""
Royal Challengers Bengaluru, formerly known as Royal Challengers Bangalore, commonly known as RCB, is a professional T20 franchise cricket team based in Bengaluru, Karnataka, that competes in the Indian Premier League. Founded in 2008 by United Spirits, the team's home ground is M. Chinnaswamy Stadium. RCB won their first title in 2025.[4] The team has also finished as the runners-up on three occasions: in 2009, 2011, and 2016. They have also qualified for the playoffs in ten of the eighteen seasons.

As of 2025, the team is captained by Rajat Patidar and coached by Andy Flower. The franchise has competed in the Champions League, finishing as runners-up in the 2011 season. As of 2024, RCB was valued at $117 million, making it one of the most valuable franchises.[5] It is also the most popular and followed cricket franchise on Instagram with more than 21 million followers.[6]
"""
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
sentences=nltk.sent_tokenize(paragraph)
print(sentences)
stemmer=PorterStemmer()
sample=stemmer.stem('thinking')
print(sample)
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
s1=lemmatizer.lemmatize('goes')
print(s1)
len(sentences)
import re
corpus=[]
for i in range(len(sentences)):
   review= re.sub('[^a-zA-Z]',' ',sentences[i])
   review=review.lower()
   corpus.append(review)
print(corpus)
for i in corpus:
    words=nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            print(stemmer.stem(word))
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(binary=True,ngram_range=(3,3))
cv.fit_transform(corpus)
print(cv.vocabulary_)
from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer()
X=cv.fit_transform(corpus)
print(X[0].toarray())
print(corpus[0])