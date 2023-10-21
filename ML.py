from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle


# Step 1: Prepare the dataset
# Assuming you have a dataset with Ta
#
# mil words and their corresponding labels (nouns or verbs)
isTrainedFlag = False

def learningPhase():
    if isTrainedFlag == False:
        #isTrainedFlag = True
        df = pd.read_csv(r"finaltrain.csv")
        df = df.dropna()
        words = df["word"].tolist()
        labels = df["pos"].tolist()
        df.head()

        # Step 2: Feature extraction
        Vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 2))
        X = Vectorizer.fit_transform(words)

        # Step 3: Split the dataset into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )

        # Step 4: Train the machine learning model
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(X_train, y_train)
        pickle.dump(classifier, open('df.pk1', 'wb'))
        pickle.dump(Vectorizer, open('vect.pk1','wb'))
        # Step 5: Predict the part of speech for new Tamil words
        # X_new = Vectorizer.transform(new_words)

def generatesingleoutput(input):
    new_words = [input]
    # Convert any NaN values to empty strings
    for i in range(len(new_words)):
        if pd.isna(new_words[i]):
            new_words[i] = ""

    classifier = pickle.load(open('df.pk1', 'rb'))
    Vectorizer = pickle.load(open('vect.pk1', 'rb'))
    predictions = classifier.predict(Vectorizer.transform(new_words))
    output=[]
    # Step 6: Print the predicted part of speech for new words
    for word, prediction in zip(new_words, predictions):
        print(f"{word},{prediction}")
        output.append(f"{word},{prediction}")
    return output

def generateOutput():
    with open("The last - Sheet2.csv", "r", encoding="utf-8") as f:
        new_words = pd.read_csv(f)["word"].tolist()
        # Convert any NaN values to empty strings
        for i in range(len(new_words)):
            if pd.isna(new_words[i]):
                new_words[i] = ""
    classifier = pickle.load(open('df.pk1', 'rb'))
    Vectorizer = pickle.load(open('vect.pk1', 'rb'))
    predictions = classifier.predict(Vectorizer.transform(new_words))

    # Step 6: Print the predicted part of speech for new words
    for word, prediction in zip(new_words, predictions):
        print(f"{word},{prediction}")

if __name__ == "__main__":
    learningPhase()
  #  generateOutput()
    print(generatesingleoutput("ஷண்"))

