from transformers import pipeline
import requests, os

def fetch_articles(keyword, date, api_key):
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

    return [article for article in articles
            if keyword.lower() in article['title'].lower() or keyword.lower() in article['description'].lower()]

def analyze_sentiment(articles, pipe):
    results = []
    contents = [a['content'] for a in articles if a.get('content')]
    sentiments = pipe(contents, batch_size=8)
    for art, sent in zip(articles, sentiments):
        results.append((art,sent))
    return results

def count_sentiments(results):
    score = 0
    count = 0
    for _, sentiment in results:
        if sentiment['label'] == 'positive':
            score += sentiment['score']
        elif sentiment['label'] == 'negative':
            score -= sentiment['score']
        count += 1
    if count == 0:
        return "Neutral"
    avg = score / count
    label = 'Positive' if avg > 0.15 else 'Negative' if avg < -0.15 else 'Neutral'
    return label, avg, count

# MAIN CODE
keyword = 'bitcoin' # Part of filtering process for the article
date = '2025-06-9'

API_KEY = os.getenv('API_KEY') or open('API_KEY').read().strip()

pipe = pipeline("text-classification", model = "ProsusAI/finbert")


articles = fetch_articles(keyword, date, API_KEY)
results = analyze_sentiment(articles, pipe)

for article, sentiment in results:
    print(f'Article: {article["title"]}, Label: {sentiment["label"]}, Score: {sentiment["score"]} \n {'-'*40}')

labels, avg_score, art_count = count_sentiments(results)
print(f'Articles Count: {art_count}'
      f'Overall Sentiment: {labels}\n'
      f'Average Score: {avg_score:.4f}')
