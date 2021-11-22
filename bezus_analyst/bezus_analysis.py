from typing import Iterable, NoReturn, Optional
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
from typing import List
nltk.download('vader_lexicon')

def print_scores(
        y_true: Iterable, y_predict: Iterable, prefix: str = "", threshold: Optional[float] = None
) -> NoReturn:
    log_loss_metric = log_loss(y_true, y_predict)
    if threshold is None:
        best_acc = 0
        best_f1 = 0
        best_threshold = None
        for tmp_threshold in np.arange(0, 1, 0.001):
            f1 = f1_score((y_true > tmp_threshold), (y_predict > tmp_threshold))
            acc = accuracy_score((y_true > tmp_threshold), (y_predict > tmp_threshold))
            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc
                best_threshold = tmp_threshold
                continue
        print(
            f"{prefix} Threshold: {best_threshold} Best F1-score: {best_f1:.4f}, Best Accuracy: {best_acc:.4f}, LogLoss: {log_loss_metric:.4f}"
        )
    else:
        f1 = f1_score((y_true > threshold), (y_predict > threshold))
        acc = accuracy_score((y_true > threshold), (y_predict > threshold))
        print(f"{prefix} F1-score: {f1:.4f}, Accuracy: {acc:.4f} LogLoss: {log_loss_metric:.4f}")


def create_event_happened_day(variable: pd.Series, percent: float = 0) -> pd.Series:
    target: List[int] = []
    for index in range(len(variable) - 1):
        target.append(int((variable[index] - variable[index + 1]) > 0))
    target.append(np.nan)
    return pd.Series(target, index=variable.index)


def make_data(sales_path: str, tweets_path: str) -> pd.DataFrame:
    df = pd.read_csv(tweets_path)
    df = df.dropna(axis=1, how='all')
    musk = df[['date', 'tweet']]
    musk['date'] = pd.to_datetime(musk['date'], format='%Y.%m.%d')
    musk['tweet'] = musk['tweet'].astype(str)
    musk_grp = musk.groupby(['date'], as_index=False).agg({'tweet': ' '.join}, Inplace=True)
    tesla = pd.read_csv(sales_path)
    tesla['Date'] = pd.to_datetime(tesla['Date'], format='%m/%d/%Y')
    merged = pd.merge(left=musk_grp, left_on='date', right=tesla, right_on='Date')
    final = merged.copy()
    final['length'] = final.tweet.str.len()
    final = final[final['length'] > 200]

    return final


def sentimental_analysis(df: pd.DataFrame) -> pd.DataFrame:
    dataframe = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
    dataframe["Comp"] = ''
    dataframe["Negative"] = ''
    dataframe["Neutral"] = ''
    dataframe["Positive"] = ''
    sentiment_i_a = SentimentIntensityAnalyzer()
    for indexx, row in df.T.iteritems():
        try:
            sentence_i = unicodedata.normalize('NFKD', df.loc[indexx, 'tweet'])
            sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
            dataframe.at[indexx, 'Comp'] = sentence_sentiment['compound']
            dataframe.at[indexx, 'Negative'] = sentence_sentiment['neg']
            dataframe.at[indexx, 'Neutral'] = sentence_sentiment['neu']
            dataframe.at[indexx, 'Positive'] = sentence_sentiment['pos']
        except TypeError:
            print(df.loc[indexx, 'tweet'])
            print(indexx)
    dataframe = dataframe.set_index(df['date'])
    dataframe['ind'] = create_event_happened_day(dataframe['Close'])
    dataframe = dataframe.dropna()
    dataframe['Comp'] = dataframe['Comp'].astype(float)
    dataframe['Negative'] = dataframe['Negative'].astype(float)
    dataframe['Neutral'] = dataframe['Neutral'].astype(float)
    dataframe['Positive'] = dataframe['Positive'].astype(float)
    dataframe['ind'] = dataframe['ind'].astype(int)

    return dataframe


def main() -> NoReturn:

    data = make_data('AMZN.csv', 'bezus.csv')
    data_sa = sentimental_analysis(data)
    x = data_sa.copy()
    x_train, x_test = x.loc[:"2019-01-01"], x.loc["2019-01-01":"2019-11-15"]
    x_train, y_train = x_train.iloc[:, :-1], x_train.iloc[:, -1]
    x_test, y_test = x_test.iloc[:, :-1], x_test.iloc[:, -1]

    plt.figure(figsize=(30, 15))
    models = [
        LogisticRegression(fit_intercept=False, n_jobs=-1, C=100, max_iter=5000),
        SVC(C=100, probability=True),

        XGBClassifier(n_estimators=100, reg_lambda=0.1, max_depth=100, n_jobs=-1),
        RandomForestClassifier(n_estimators=100),
        KNeighborsClassifier(2),
    ]

    for model in models:
        model.fit(x_train, y_train)
        y_predict = model.predict_proba(x_test)[:, 1]

        class_name = model.__class__.__name__
        print_scores(y_train, model.predict(x_train), f"{class_name} train day event prediction")
        print_scores(y_test, y_predict, f"{class_name} test day event prediction")
        print("")

        plt.plot(y_test.index, y_predict, label=class_name)
    # with open("tweet_prob.txt", "w") as file:
    #     np.savetxt(file, models[-1].predict_proba(x_test)[:, 1])

    plt.plot(y_test, label="True")
    plt.legend()
    plt.title(f"Probability that price will fall after Elon Musk tweet")
    # plt.show()


if __name__ == '__main__':
    main()