import streamlit as st
from bertopic.representation import KeyBERTInspired
from bertopic.representation import PartOfSpeech
from bertopic.representation import MaximalMarginalRelevance
from bertopic import BERTopic
import pandas as pd

st.set_page_config(layout="wide", page_title="Trend Monitoring System")
st.header('FACTCHECK CLAIM MONITORING SYSTEM :sunglasses:',divider='rainbow', )

st.session_state.topics_overtime=pd.DataFrame()

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


def claim_extractor(df):

    # The main representation of a topic
    main_representation = KeyBERTInspired() #

    # Additional ways of representing a topic
    aspect_model1 = PartOfSpeech("en_core_web_sm")
    aspect_model2 = [KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance(diversity=.5)]

    # Add all models together to be run in a single `fit`nam
    representation_model = {
    "Main": main_representation,
    "Aspect1":  aspect_model1,
    "Aspect2":  aspect_model2 
    }
    
    docs=df['c'].tolist()
    time_s=df['date'].tolist()
    
    topic_model = BERTopic(representation_model=representation_model).fit(docs)
    
    #Modeling over time
    topics_overtime=topic_model.topics_over_time(docs, time_s)
    topics_overtime['representative_text']=topics_overtime['Topic'].map(topic_model.get_representative_docs())
    
    st.session_state.topics_overtime=topics_overtime

    fig=topic_model.visualize_topics_over_time(topics_over_time=topics_overtime,title="Topic Trends")#, top_n_topics=5, topics=topic_model.get_topics(), n_words=5)
    st.plotly_chart(fig,use_container_width=True)
    return 


def main():
    st.session_state.df=load_data()
    #date=st.date_input("Start Date",value=[st.session_state.df['date'].min(),st.session_state.df['date'].max()])
    
    
    start_date=st.sidebar.date_input("Start Date",value=st.session_state.df['date'].min())
    end_date=st.sidebar.date_input("End Date",value=start_date+datetime.timedelta(days=30))
    
    #if start_date and end_date:
    st.session_state.df=st.session_state.df[(st.session_state.df['date']>=start_date) & (st.session_state.df['date']<=end_date)]
    #st.session_state.df=st.session_state.df.query('date >=@start_date and date <=@end_date')

    
    claim_extractor(st.session_state.df)
    
       
    ww=st.session_state.topics_overtime.query('Topic!=-1')[['Timestamp','Topic','representative_text','Words']]
    #batch_size=20
    
    pagination= st.container()
    pagination.dataframe(data=ww, use_container_width=True)
    
if __name__=="__main__":
    main()
    