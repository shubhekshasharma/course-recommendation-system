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


def get_recommendations(user_input):
    client = get_llm_client()

    response = get_course_workload_and_additional_keywords(client, user_input)
    response_dict = json.loads(response)

    credit_category = response_dict['category']
    interest_key_words = response_dict["interest_key_words"]
    
    user_interest_key_words_text = " ".join(interest_key_words)

    clustered_courses = find_courses_by_same_cluster(user_interest_key_words_text)

    similar_courses = find_courses_by_text_similarity(
        user_text=user_interest_key_words_text, 
        same_cluster_courses=clustered_courses, 
        top_k=25
    )

    courses_by_preferred_level = find_courses_by_preferred_credit_level(
        similar_courses=similar_courses, 
        preferred_credit_level=credit_category
    )

    recommendation_results = get_recommendations_reasoning(
        client=client, 
        results=courses_by_preferred_level, 
        user_input=user_interest_key_words_text
    )
    recommendation_json_results = json.loads(recommendation_results)
    return recommendation_json_results



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
            results = get_recommendations(user_input)

        st.success("Here are your recommended courses:")

        col1, col2 = st.columns(2, gap="large")

        # Loop through results 2 per row
        for i, rec in enumerate(results):

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