# Save this as app.py and run with: streamlit run app.py

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import re
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import streamlit as st

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ----------------- Data Simulation -----------------
@st.cache_data
def simulate_data():
    num_entries = 600
    data = []

    positive_comments = ["Love this!", "Amazing content!", "So beautiful", "Great product", "Absolutely stunning!"]
    neutral_comments = ["Okay post", "Not bad", "Decent content", "Meh", "It's fine"]
    negative_comments = ["I don't like this", "Terrible", "So bad", "Uninspiring", "Worst post ever"]
    content_types = ["Image", "Reel", "Story"]
    emoji_counts = [0, 1, 2, 3, 5]
    hashtag_counts = [0, 1, 3, 5, 7, 10]

    for i in range(num_entries):
        version = 'A' if i < 300 else 'B'
        impressions = random.randint(800, 1500)
        ctr = np.random.uniform(0.05, 0.2) if version == 'A' else np.random.uniform(0.06, 0.22)
        clicks = int(impressions * ctr)
        conv_rate = np.random.uniform(0.1, 0.3)
        conversions = int(clicks * conv_rate)
        likes = random.randint(10, 300)
        followers = random.randint(500, 5000)

        sentiment_type = random.choices(['positive', 'neutral', 'negative'], weights=[0.4, 0.4, 0.2])[0]
        if sentiment_type == 'positive':
            comment = random.choice(positive_comments)
        elif sentiment_type == 'neutral':
            comment = random.choice(neutral_comments)
        else:
            comment = random.choice(negative_comments)

        timestamp = datetime.datetime.now() - datetime.timedelta(minutes=random.randint(0, 10000))

        data.append({
            'version': version,
            'timestamp': timestamp,
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'followers': followers,
            'likes': likes,
            'caption_length': random.randint(20, 300),
            'hashtag_count': random.choice(hashtag_counts),
            'emoji_count': random.choice(emoji_counts),
            'content_type': random.choice(content_types),
            'comment': comment,
            'true_sentiment': sentiment_type
        })

    return pd.DataFrame(data)

# ----------------- Data Preprocessing -----------------
df = simulate_data()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

df['cleaned_comment'] = df['comment'].apply(clean_text)

analyzer = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['comment'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

def label_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['predicted_sentiment'] = df['sentiment_score'].apply(label_sentiment)

df['CTR'] = df['clicks'] / df['impressions']
df['Conversion Rate'] = df['conversions'] / df['clicks']
df['Engagement Rate'] = df['likes'] / df['followers']
df['hour_posted'] = df['timestamp'].dt.hour

def detect_outliers(metric):
    q1 = df[metric].quantile(0.25)
    q3 = df[metric].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[metric] < lower) | (df[metric] > upper)]

df['optimization_suggestions'] = df.apply(lambda row: ['Add more hashtags' if row['hashtag_count'] < 3 else '',
                                                        'Use more emojis' if row['emoji_count'] < 1 else '',
                                                        'Use longer captions' if row['caption_length'] < 50 else ''], axis=1)

# ----------------- Streamlit Dashboard -----------------
st.title("📸 Instagram Marketing Analytics Dashboard")

menu = st.sidebar.radio(
    "Choose Analysis",
    ("Overview", "Sentiment Analysis", "Engagement Trends", "Content Analysis", "Outlier Detection", "Optimization Suggestions")
)

# ----- Overview Metrics -----
if menu == "Overview":
    st.header("🔹 Overall Metrics Summary")
    st.dataframe(df[['CTR', 'Conversion Rate', 'Engagement Rate']].groupby(df['version']).mean().round(3))

    st.header("🔹 Classification Report")
    st.text(classification_report(df['true_sentiment'].str.capitalize(), df['predicted_sentiment'], digits=3))

# ----- Sentiment Analysis -----
elif menu == "Sentiment Analysis":
    st.header("🔹 Sentiment Distribution")
    fig, ax = plt.subplots()
    df['predicted_sentiment'].value_counts().plot(kind='bar', color=['green', 'gray', 'red'], ax=ax)
    plt.title('Sentiment Breakdown')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    st.pyplot(fig)

    st.header("🔹 Sentiment by Version")
    fig, ax = plt.subplots()
    sentiment_counts = df.groupby(['version', 'predicted_sentiment']).size().unstack()
    sentiment_counts.plot(kind='bar', stacked=True, colormap='coolwarm', ax=ax)
    plt.title('Sentiment Distribution by A/B Version')
    plt.ylabel('Count')
    st.pyplot(fig)

# ----- Engagement Trends -----
elif menu == "Engagement Trends":
    st.header("🔹 Engagement Over Time")
    df_time = df.copy()
    df_time['hour'] = df_time['timestamp'].dt.floor('1H')
    grouped = df_time.groupby(['hour', 'version'])['Engagement Rate'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(data=grouped, x='hour', y='Engagement Rate', hue='version', marker="o", ax=ax)
    plt.title("Engagement Rate Over Time by Version")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ----- Content Analysis -----
elif menu == "Content Analysis":
    st.header("🔹 CTR by Content Type")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x='content_type', y='CTR', estimator=np.mean, palette='viridis', ax=ax)
    plt.title('Average CTR by Content Type')
    plt.ylabel('CTR')
    st.pyplot(fig)

    st.header("🔹 Hashtag Count vs Engagement")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='hashtag_count', y='Engagement Rate', hue='version', alpha=0.7, ax=ax)
    plt.title('Hashtag Count vs Engagement Rate')
    st.pyplot(fig)

# ----- Outlier Detection -----
elif menu == "Outlier Detection":
    st.header("🔹 CTR Outliers")
    st.dataframe(detect_outliers('CTR')[['CTR', 'content_type', 'version']])

    st.header("🔹 Engagement Rate Outliers")
    st.dataframe(detect_outliers('Engagement Rate')[['Engagement Rate', 'content_type', 'version']])

# ----- Optimization Suggestions -----
elif menu == "Optimization Suggestions":
    st.header("🔹 Sample Post and Suggestions")
    st.write(df[['comment', 'optimization_suggestions']].sample(1))

    st.header("🔹 Best Posting Hour Analysis")
    best_hour = df.groupby('hour_posted')['Engagement Rate'].mean().idxmax()
    st.success(f"✅ Best hour to post: {best_hour}:00")

