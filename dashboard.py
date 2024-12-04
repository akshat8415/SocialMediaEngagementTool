import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['positive_percentage'] = data['positive_percentage'].str.rstrip('%').astype(float)
    data['negative_percentage'] = data['negative_percentage'].str.rstrip('%').astype(float)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['id_short'] = data['id'].astype(str).str[-4:]
    return data

def apply_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f09433 0%, #e6683c 25%, #dc2743 50%, #cc2366 75%, #bc1888 100%);
            background-attachment: fixed;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .block-container {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
        }
        h1, h2, h3, p {
            color: #333333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    apply_custom_css()
    file_path = "user_posts_with_bert_sentiment.csv"
    data = load_data(file_path)

    st.title("🌈 Social Media Dashboard")
    st.sidebar.header("Filter Options")
    start_date = st.sidebar.date_input("Start Date", data['timestamp'].min().date())
    end_date = st.sidebar.date_input("End Date", data['timestamp'].max().date())

    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    end_date = pd.to_datetime(end_date).tz_localize('UTC')
    filtered_data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]

    st.subheader("📊 Key Metrics")
    total_likes = filtered_data['like_count'].sum()
    total_comments = filtered_data['comment_count'].sum()
    average_likes = filtered_data['like_count'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Likes", total_likes)
    col2.metric("Total Comments", total_comments)
    col3.metric("Average Likes", f"{average_likes:.2f}")

    st.subheader("Engagement per Post")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='id_short', y='like_count', data=filtered_data, color='skyblue', ax=ax)
    sns.barplot(x='id_short', y='comment_count', data=filtered_data, color='orange', ax=ax)
    ax.set_title("Likes and Comments per Post")
    ax.set_xlabel("Post ID (Last 4 Digits)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Sentiment Analysis")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='id_short', y='positive_percentage', data=filtered_data, color='green', ax=ax)
    sns.barplot(x='id_short', y='negative_percentage', data=filtered_data, color='red', ax=ax)
    ax.set_title("Positive vs Negative Sentiment")
    ax.set_xlabel("Post ID (Last 4 Digits)")
    ax.set_ylabel("Percentage")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
