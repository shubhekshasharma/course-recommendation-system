import plotly.express as px
import pandas as pd
import streamlit as st
from recommendations.llm_utils import (
    get_llm_client, 
    get_course_workload_and_additional_keywords, 
    get_recommendations_reasoning
)
from recommendations.recommend import (
    find_courses_by_same_cluster, 
    find_courses_by_text_similarity, 
    find_courses_by_preferred_credit_level
)
import json
import plotly.express as px
import pandas as pd


def get_recommendations(user_input):
    client = get_llm_client()

    response = get_course_workload_and_additional_keywords(client, user_input)
    response_dict = json.loads(response)

    credit_category = response_dict['category']
    interest_key_words = response_dict["interest_key_words"]
    
    user_interest_key_words_text = " ".join(interest_key_words)

    clustered_courses = find_courses_by_same_cluster(user_interest_key_words_text)

    similar_interest_courses = find_courses_by_text_similarity(
        user_text=user_interest_key_words_text, 
        same_cluster_courses=clustered_courses, 
        top_k=25
    )

    courses_by_preferred_level = find_courses_by_preferred_credit_level(
        similar_courses=similar_interest_courses, 
        preferred_credit_level=credit_category
    )

    recommendation_results = get_recommendations_reasoning(
        client=client, 
        results=courses_by_preferred_level, 
        user_input=user_interest_key_words_text
    )
    recommendation_json_results = json.loads(recommendation_results)
    # Similar interest are used for finding similar courses for vizualization. 

    return similar_interest_courses, recommendation_json_results


def plot_workload_vs_interest_highlighted(recommended_df, highlight_list, top_n=10):
    """
    Bubble chart for recommended courses with highlighted courses.
    Y-axis shows categorical workload levels: Low, Standard, High.
    """
    # Extract highlighted keys
    highlight_keys = set([c['key'].strip() for c in highlight_list])

    # Ensure 'key' in df is string and strip spaces
    recommended_df['key'] = recommended_df['key'].astype(str).str.strip()

    # Take top N courses by similarity
    recommended_df = recommended_df.sort_values(by='similarity', ascending=False).head(top_n).copy()

    # Fill missing values
    recommended_df['minimum_credits'] = recommended_df['minimum credits'].fillna(1)
    recommended_df['predicted_credit_level'] = recommended_df['predicted_credit_level'].fillna('Standard')  # default

    # Convert predicted_credit_level to categorical with fixed order
    level_order = ['Low', 'Standard', 'High']
    recommended_df['predicted_credit_level'] = pd.Categorical(
        recommended_df['predicted_credit_level'],
        categories=level_order,
        ordered=True
    )

    # Map categories to numeric for plotting
    recommended_df['predicted_credit_numeric'] = recommended_df['predicted_credit_level'].cat.codes

    # Create highlight column
    recommended_df['highlight'] = recommended_df['key'].apply(
        lambda x: 'Highlighted' if x in highlight_keys else 'Other'
    )

    # Define colors
    color_map = {'Highlighted': 'red', 'Other': 'lightblue'}

    # Create bubble chart
    fig = px.scatter(
        recommended_df,
        x='similarity',
        y='predicted_credit_numeric',
        size='minimum_credits',
        color='highlight',
        color_discrete_map=color_map,
        hover_name='key',
        hover_data=['title', 'similarity', 'minimum_credits', 'predicted_credit_level'],
        size_max=35,
        opacity=0.8,
        labels={
            'similarity': 'Interest Match (Similarity)',
            'predicted_credit_numeric': 'Predicted Workload',
            'minimum_credits': 'Credits',
            'highlight': ''
        },
        title='Workload vs Interest Match — Recommended Courses'
    )

    # Update Y-axis to show categorical labels
    fig.update_yaxes(
        tickmode='array',
        tickvals=[0, 1, 2],
        ticktext=level_order,
        title='Predicted Workload'
    )

    # X-axis 0-1 similarity
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(height=650, showlegend=True)

    return fig





st.set_page_config(page_title="Course Recommender", layout="wide")

max_width_style = """
<style>
    .block-container {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }
</style>
"""
st.markdown(max_width_style, unsafe_allow_html=True)

st.title("Course Recommendation System")

user_input = st.text_area(
    "Describe your interests or what type of courses you want:",
    height=150,
    placeholder="Example: I like biology but want lighter workload courses...",
)

if st.button("Recommend"):

    if not user_input.strip():
        st.warning("Please enter something!")
    else:
        with st.spinner("Generating recommendations..."):
            all_recommended_data, results_llm = get_recommendations(user_input)

        st.success("Here are your recommended courses:")

        col1, col2 = st.columns(2, gap="large")


        for i, rec in enumerate(results_llm):

            col = col1 if i % 2 == 0 else col2

            with col:
                with st.container(border=True):

                    st.subheader(f"{rec['key']} — {rec['title']}")

                    st.markdown(f"**Minimum Credits:** {rec['minimum_credits']}")
                    st.markdown(f"**Similarity:** {round(rec['similarity'], 3)}")

                    st.markdown("**Description:**")
                    st.write(rec["description"])

                    st.markdown("**Reasoning:**")
                    st.write(rec["reasoning"])

        # Plot the new bubble chart
        fig = plot_workload_vs_interest_highlighted(all_recommended_data, highlight_list=results_llm)
        st.plotly_chart(fig, use_container_width=True)