from transformers import pipeline
import requests, os
from datetime import datetime, UTC, timedelta
import csv

##### FILTER and SENTIMENT ANALYSIS

def fear_greed_index(limit=1):
    """
    :param limit: is for how many items (data value) to return
    :return: the score, classification, timestamp of the index
    """
    url = f"https://api.alternative.me/fng/?limit={limit}&date_format=us"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            # data comes in json with data as the key
            raw_data = response.json()["data"]
            results = []

            # iterate the each data value
            for data in raw_data:
                score = int(data["value"])
                classification = data["value_classification"]
                timestamp = data["timestamp"]
                date_obj = datetime.strptime(timestamp, "%m-%d-%Y")
                formatted_date = date_obj.strftime("%m%d%Y")

                results.append((score, classification, formatted_date))

            return results
        else:
            print("Failed to fetch: Error")
            return None, None, None

    except Exception as e:
        print(f"Failed to fetch: Error {e}")
        return None, None, None

def is_relevant_article(article, keyword, min_mentions=2, check_conclusion=False):
    # extract content title description
    content = article.get('content') or ''
    title = article.get('title') or ''
    description = article.get('description') or ''

    keyword_lower = keyword.lower()
    keyword_count = content.lower().count(keyword_lower)

    keyword_in_conclusion = False
    # check if keyword in conclusion
    if check_conclusion:
        words = content.lower().split()
        conclusion = ' '.join(words[-200:])

        if keyword_lower in conclusion:
            keyword_in_conclusion = True
        else:
            keyword_in_conclusion = False

    # high impact words that might affect bitcoin price
    impact_terms = [
        "regulation", "ban", "etf", "adoption", "hack", "inflation",
        "fed", "lawsuit", "approval", "sec", "institution", "elon"
    ]

    # check if impact word exists
    impact_present = any(term in content.lower() for term in impact_terms)

    # check that bitcoin must appear in both title and description
    if keyword_lower not in title.lower() or keyword_lower not in description.lower():
        return False

    # return value based on check_conclusion
    if check_conclusion:
        return keyword_in_conclusion and (impact_present or keyword_count >= min_mentions)

    return impact_present or keyword_count >= min_mentions

def fetch_articles(keyword, date, api_key):
    # nesapi query
    url = (
        'https://newsapi.org/v2/everything?'
        f'q={keyword}&'
        f'from={date}&'
        'sortBy=popularity&'
        f'apiKey={API_KEY}'
    )

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f'Request failed, Status Code: {response.status_code}')


    articles = response.json().get("articles", [])

    # filter articles based on function above
    return [article for article in articles
            ### CHANGE THIS PARAMETER FOR FILTER
            # min_mentions = 2 is good because its more balanced
            if is_relevant_article(article, keyword, min_mentions=1, check_conclusion=True)]

def analyze_sentiment(articles, pipe):

    results = []

    # get the non-empty article content
    contents = [a['content'] for a in articles if a.get('content')]

    # pipeline
    sentiments = pipe(contents, batch_size=8)
    for art, sent in zip(articles, sentiments):
        results.append((art,sent))
    return results

# to count sentiment scores
def count_sentiments(results):
    score = 0
    count = 0
    for _, sentiment in results:
        if sentiment['label'] == 'positive':
            score += sentiment['score']
        elif sentiment['label'] == 'negative':
            score -= sentiment['score']
        count += 1

    # no valid articles
    if count == 0:
        return "Neutral", 0.0, 0, 0

    avg = score / count

    # change limit for how many index to get (count backwards)
    fg_index = fear_greed_index(limit=1)

    for score, classification, formatted_date in fg_index:
        fg_score = score / 100
        fg_classification = classification
        fg_date = formatted_date

    # print(fg_score, fg_classification, fg_date)

    # thresholds
    label = 'Positive' if avg > 0.15 else 'Negative' if avg < -0.15 else 'Neutral'

    avg_fg = avg

    if label == 'Positive':
        if fg_classification == 'Extreme Greed':
            # 75-100
            avg_fg = (fg_score * 0.5) + (avg * 0.5)
        elif fg_classification == 'Greed':
            # 50 - 75
            avg_fg = (fg_score * 0.35) + (avg * 0.65)
        elif fg_classification == 'Fear':
            # 25-50
            avg_fg = (avg * 0.6) - (fg_score * 0.4)
        elif fg_classification == 'Extreme Fear':
            # 0-25
            avg_fg = (avg * 0.3) - (fg_score * 0.9)
        else:
            avg = avg

    if label == 'Negative':
        if fg_classification == 'Fear':
            avg_fg = (fg_score * 0.2) + (avg * 0.8)

        elif fg_classification == 'Extreme Fear':
            avg_fg = (fg_score * 0.3) + (avg * 0.7)
        elif fg_classification == 'Extreme Greed':
            # 75-100
            avg_fg = (fg_score * 0.75) + (avg * 0.4)
        elif fg_classification == 'Greed':
            # 50 - 75
            avg_fg = (fg_score * 0.5) + (avg * 0.5)

        else:
            avg = avg

    return label, avg, avg_fg, count

##### GENERATE SIGNAL

def generate_signal(avg_fg_score, threshold_positive=0.3, threshold_negative=0.3):
    if avg_fg_score >= threshold_positive:
        return "BUY"
    elif avg_fg_score >= threshold_negative:
        return "SELL"
    else:
        return "HOLD"

# MAIN CODE
keyword = 'bitcoin' # Part of filtering process for the article
# date = '2025-05-27' # 24 hour delay because its free

API_KEY = os.getenv('API_KEY') or open('API_KEY').read().strip()

pipe = pipeline("text-classification", model = "ProsusAI/finbert")

### FOR ONE USE ONLY
# articles = fetch_articles(keyword, date, API_KEY)
# results = analyze_sentiment(articles, pipe)
#
# for article, sentiment in results:
#     print(f'Article: {article["title"]}, Label: {sentiment["label"]}, Score: {sentiment["score"]} \n {'-'*40}')
#
# labels, avg_score, avg_fg_score, art_count = count_sentiments(results)
#
# print(f'Articles Count: {art_count}\n'
#       f'Overall Sentiment: {labels}\n'
#       f'Average Score: {avg_score:.4f}\n'
#       f'Average Score with Fear and Greed: {avg_fg_score:.4f}'
#       )

### GENERATING CSV WITH DATE, AVG_FG_SCORE, SIGNAL FOR BACKTEST
start_date = datetime.strptime('2025-05-18', '%Y-%m-%d')
end_date = datetime.strptime("2025-06-15", "%Y-%m-%d")

output = []

while start_date <= end_date:
    date_str = start_date.strftime('%Y-%m-%d')

    print(date_str)
    articles = fetch_articles(keyword, date_str, API_KEY)
    results = analyze_sentiment(articles, pipe)

    label, avg_score, avg_fg_score, art_count = count_sentiments(results)
    signal = generate_signal(avg_fg_score, 0.28, 0.28)

    output.append([date_str, avg_fg_score, signal])

    start_date += timedelta(days=1)

with open("sentiment_signals.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["date", "avg_fg_score", "signal"])
    writer.writerows(output)
