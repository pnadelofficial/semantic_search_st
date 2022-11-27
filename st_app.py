import streamlit as st
from semantic_search import SemanticSearch, sentence_tokenize
import pandas as pd
import spacy

st.markdown('<h1>Semantic Search</h1>', unsafe_allow_html=True)
st.markdown('<small>Assembled by Peter Nadel, Tufts University</small>', unsafe_allow_html=True)

uploaded_file = st.file_uploader('Choose a CSV.')
col_name = st.text_input('Text column name (if not text)', 'text')

if uploaded_file is not None:
    @st.cache
    def get_data():
        base_df = pd.read_csv(uploaded_file).dropna()
        if col_name != 'text':
            df = sentence_tokenize(base_df, col_name=col_name)
        else:
            df = sentence_tokenize(base_df)
        return df
    df = get_data()

    @st.cache(allow_output_mutation=True)
    def load_model():
        return spacy.load('en_core_web_md')
    nlp = load_model()

    semantic_search = SemanticSearch(df, nlp)

    @st.cache(suppress_st_warning=True)
    def spacyify():
        semantic_search.spacyify(col_name='sents', streamlit=True)
    spacyify()

    entries = st.number_input('Choose number of excerpts.', min_value=1, value=5)
    context_size = st.number_input('Choose context size (number of sentences before and after).', min_value=1, value=2)
    cols_to_display = st.text_input('Enter names of columns to be displayed', '')
    
    search_text = st.text_input('Search term', '')
    if search_text != '':
        search = semantic_search.search(
            'serialized_data/spacy_model_output', 
            search_text, 
            entries=entries, 
            context_size=context_size,
            streamlit=True,
            kwargs=cols_to_display
            )

        st.markdown(
            f'<h2>{search[1]}</h2>'
            ,unsafe_allow_html=True
        )

        for i in range(len(search[0])):
            for col in search[0].columns[2:-1]:
                st.markdown(
                    f'<small>{col.title()}: {search[0][col].to_list()[i]}</small>'
                    ,unsafe_allow_html=True
                )
            st.markdown(
                f'<small>Similarity Score: {round(search[0].sent_docs.to_list()[i], 3)}</small>'
                ,unsafe_allow_html=True
            )
            st.markdown(
                f'<p>{search[0].context.to_list()[i]}</p>'
                ,unsafe_allow_html=True
            )
            st.markdown('<hr>', unsafe_allow_html=True)
