from transformers import pipeline
import requests, os

def is_relevant_article(article, keyword, min_mentions=5, check_conclusion=False):
    content = article.get('content') or ''
    title = article.get('title') or ''
    description = article.get('description') or ''

    keyword_lower = keyword.lower()
    keyword_count = content.lower().count(keyword_lower)

    if check_conclusion:
        words = content.lower().split()
        conclusion = ' '.join(words[-200:])
        keyword_in_conclusion = keyword_lower in conclusion
    else:
        keyword_in_conclusion = True

    # high impact words that might affect bitcoin price
    impact_terms = [
        "regulation", "ban", "etf", "adoption", "hack", "inflation",
        "fed", "lawsuit", "approval", "sec", "institution", "elon"
    ]

    impact_present = any(term in content.lower() for term in impact_terms)

    # check that bitcoin must appear in both title and description
    if keyword_lower not in title.lower() or keyword_lower not in description.lower():
        return False

    return keyword_in_conclusion and (impact_present or keyword_count >= min_mentions)

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
            ### CHANGE THIS PARAMETER FOR FILTER
            # min_mentions = 2 is good because its more balanced
            if is_relevant_article(article, keyword, min_mentions=2, check_conclusion=True)]

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
        return "Neutral", 0.0, 0
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

print(f'Articles Count: {art_count}\n'
      f'Overall Sentiment: {labels}\n'
      f'Average Score: {avg_score:.4f}')
