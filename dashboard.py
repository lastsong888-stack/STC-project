import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

# ─── إعداد الصفحة ───
st.set_page_config(page_title="STC Bank Sentiment Dashboard", layout="wide")

# ─── تحميل البيانات ───
@st.cache_data
def load_data():
    # يقرأ الملف من نفس مجلد المشروع على GitHub
    df = pd.read_csv('dashboard_data_950.csv', encoding='utf-8-sig')
    return df

df = load_data()

# ─── العنوان ───
st.title("📊 STC Bank — Customer Sentiment Dashboard")
st.markdown("**Decision-Support Tool for the Director of Customer Experience**")
st.markdown("---")

st.info(f"📌 This dashboard displays **{len(df)} reviews** classified by **Gemini (LLM)**, which achieved 93.67% accuracy. The remaining 1,055 reviews could not be classified due to API quota limits (documented in M5).")

# ─── الشريط الجانبي ───
st.sidebar.header("🔍 Filters")
topics = ["All"] + list(df['topic'].unique())
selected_topic = st.sidebar.selectbox("Select Topic", topics)
months = ["All"] + sorted(df['month'].unique())
selected_month = st.sidebar.selectbox("Select Month", months)
sentiments = ["All", "Positive", "Negative", "Neutral"]
selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiments)

# ─── تطبيق الفلاتر ───
df_filtered = df.copy()
if selected_topic != "All": df_filtered = df_filtered[df_filtered['topic'] == selected_topic]
if selected_month != "All": df_filtered = df_filtered[df_filtered['month'] == selected_month]
if selected_sentiment != "All": df_filtered = df_filtered[df_filtered['llm_pred'] == selected_sentiment]

st.sidebar.markdown(f"**📄 Showing {len(df_filtered)} reviews**")

# ─── المؤشرات ───
st.header("📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
pos_count = len(df_filtered[df_filtered['llm_pred'] == 'Positive'])
neg_count = len(df_filtered[df_filtered['llm_pred'] == 'Negative'])
neu_count = len(df_filtered[df_filtered['llm_pred'] == 'Neutral'])
total = len(df_filtered)
col1.metric("Total Reviews", total)
col2.metric("Positive", f"{pos_count} ({pos_count/total*100:.1f}%)" if total > 0 else "0")
col3.metric("Negative", f"{neg_count} ({neg_count/total*100:.1f}%)" if total > 0 else "0")
col4.metric("Neutral", f"{neu_count} ({neu_count/total*100:.1f}%)" if total > 0 else "0")
st.markdown("---")

# ─── الرسوم البيانية ───
st.header("1. Sentiment Distribution")
fig1 = px.pie(df_filtered, names='llm_pred', title="Sentiment Distribution", color='llm_pred', color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'})
st.plotly_chart(fig1, use_container_width=True)

st.header("2. Sentiment by Topic")
if len(df_filtered) > 0:
    topic_sentiment = df_filtered.groupby(['topic', 'llm_pred']).size().reset_index(name='count')
    fig2 = px.bar(topic_sentiment, x='topic', y='count', color='llm_pred', title="Sentiment Distribution by Topic", color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}, barmode='group')
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

st.header("3. Sentiment Over Time")
if len(df_filtered) > 0:
    month_sentiment = df_filtered.groupby(['month', 'llm_pred']).size().reset_index(name='count')
    fig3 = px.line(month_sentiment, x='month', y='count', color='llm_pred', title="Sentiment Trends Over Time", color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}, markers=True)
    st.plotly_chart(fig3, use_container_width=True)

st.header("4. Top Words in Negative Reviews")
neg_texts = df_filtered[df_filtered['llm_pred'] == 'Negative']['content_clean'].dropna()
if len(neg_texts) > 0:
    vectorizer = CountVectorizer(max_features=20)
    X = vectorizer.fit_transform(neg_texts)
    word_counts = pd.DataFrame({'word': vectorizer.get_feature_names_out(), 'count': X.toarray().sum(axis=0)}).sort_values('count', ascending=False).head(20)
    fig4 = px.bar(word_counts, x='count', y='word', orientation='h', title="Top 20 Words in Negative Reviews", color_discrete_sequence=['#e74c3c'])
    st.plotly_chart(fig4, use_container_width=True)
