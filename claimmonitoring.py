import streamlit as st
from bertopic.representation import KeyBERTInspired
from bertopic.representation import PartOfSpeech
from bertopic.representation import MaximalMarginalRelevance
from bertopic import BERTopic
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px


import pandas as pd

st.set_page_config(layout="wide", page_title="Trend Monitoring System")
st.header('FACTCHECK CLAIM MONITORING SYSTEM :sunglasses:',divider='rainbow', )
cont1=st.container()


#add date column using random dates between 2023 and 2024
import random
import datetime

pd.set_option('display.max_colwidth', None)

@st.cache_data
def load_data():
    df=pd.read_csv("test.csv", parse_dates=['date'])
    
    #st.session_state.df=df
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    df.dropna(subset=['date'], inplace=True)
    
    return df

@st.cache_data
def claim_extractor(df):

    # The main representation of a topic
    main_representation = KeyBERTInspired() #

    # Additional ways of representing a topic
    #aspect_model1 = PartOfSpeech("en_core_web_sm")
    aspect_model2 = [KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance(diversity=.5)]

    # Add all models together to be run in a single `fit`nam
    representation_model = {
    "Main": main_representation,
    #"Aspect1":  aspect_model1,
    "Aspect2":  aspect_model2 
    }
    
    docs=df['c'].tolist()
    time_s=df['date'].tolist()
    
    topic_model = BERTopic(representation_model=representation_model).fit(docs)
    
    #Modeling over time
    topics_overtime=topic_model.topics_over_time(docs, time_s)
    
    topics_overtime['representative_text']=topics_overtime['Topic'].map(topic_model.get_representative_docs())
    
    st.session_state.topics_overtime=topics_overtime
    st.session_state.topic_model=topic_model
    
    st.session_state.fig=topic_model.visualize_topics_over_time(topics_over_time=topics_overtime,title="Topic Trends")#, top_n_topics=5, topics=topic_model.get_topics(), n_words=5)
    #cont1.plotly_chart(st.session_state.fig,use_container_width=True)
    
    return 


def filter_by_topic(df, topic):
    return df[df['Topic'] == topic]

def generate_word_cloud(selected_topic, ww):
    text = ' '.join(ww.query('Topic == @selected_topic')['Words'].values)
    wordcloud = WordCloud(width=1200, height=600, background_color='white').generate(text)
    wordcloud.to_file("wordcloud.png")
    img = plt.imread("wordcloud.png")
    fig = px.imshow(img, width=1200, height=600)
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    #st.session_state.fig2=fig
    return fig
    

def main():
    
    
    start_date = st.sidebar.date_input("Start Date", value=st.session_state.df['date'].min())
    end_date = st.sidebar.date_input("End Date", value=start_date + datetime.timedelta(days=30))
    
    st.session_state.df = st.session_state.df[(st.session_state.df['date'] >= start_date) & (st.session_state.df['date'] <= end_date)]
    
    st.sidebar.button("Run Analysis", on_click=claim_extractor, args=(st.session_state.df,))
    
    st.markdown("---")
    
    if 'topics_overtime' in st.session_state:
        ww = st.session_state.topics_overtime.query('Topic != -1')[['Topic', 'representative_text', 'Words']]
        
        col1, col2,col3 = st.columns([1, 2, 4],vertical_alignment='top',)
        col1.subheader("Topics")
        
        www = ww.drop(columns=['Words'], axis=1)
        www = www.rename(columns={'representative_text': 'Representative Text'})
        x = www.groupby('Topic').first()
        
        topics = x.index.tolist()
        selected_topic = col1.radio("Select a Topic", topics, index=None)
        
        with col2:
            if selected_topic is not None:
                st.session_state.selected_topic = selected_topic
        
            if 'selected_topic' in st.session_state and st.session_state.selected_topic is not None:
                filtered_df = filter_by_topic(x.reset_index(), st.session_state.selected_topic)
                col2.subheader("Representative Text")
                temp = filtered_df['Representative Text'].values[0]
                temp = "\n\n".join(temp)
                col2.write(temp)
                fig2 = generate_word_cloud(st.session_state.selected_topic, ww)
                #col3 = st.container()
                #col3.subheader(f"Word Cloud Topic {st.session_state.selected_topic}")
            
                col3.plotly_chart(fig2, use_container_width=False)
        
        if 'fig' not in st.session_state: 
            topic_model = st.session_state.topic_model
            fig = topic_model.visualize_topics_over_time(topics_over_time=st.session_state.topics_overtime, title="Topic Trends")
            st.session_state.fig = fig
        
        cont1.plotly_chart(st.session_state.fig, use_container_width=True)

    
    
        
if __name__=="__main__":
    st.session_state.df = load_data()
    main()
    
    
    
    