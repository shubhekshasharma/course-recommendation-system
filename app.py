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
import numpy as np
import altair as alt


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

    # return 4 recommendations max for readability
    return similar_interest_courses, recommendation_json_results[:4], credit_category


def plot_workload_vs_interest_highlighted(
    recommended_df, 
    highlight_list, 
    preferred_credit_level, 
    top_n=10
):
    """
    Bubble chart for recommended courses with highlighted courses.
    Includes:
    - Highlight band
    - Recommended vs Similar colors
    - Categorical workload axis
    - Bubble size by minimum credits
    """
    highlight_keys = set([c['key'].strip() for c in highlight_list])

    recommended_df = recommended_df.copy()
    recommended_df["key"] = recommended_df["key"].astype(str).str.strip()

    # Separate highlighted vs non-highlighted
    # Our intention is to display all highlighted ones and some non-highlighted similar ones
    highlight_df = recommended_df[recommended_df["key"].isin(highlight_keys)]
    non_highlight_df = recommended_df[~recommended_df["key"].isin(highlight_keys)]
    # Take top N from non-highlighted
    top_non_highlight_df = non_highlight_df.sort_values("similarity", ascending=False).head(top_n)
    # Combine with all highlighted courses
    recommended_df = pd.concat([highlight_df, top_non_highlight_df]).copy()

    # Fix columns
    recommended_df["minimum_credits"] = recommended_df["minimum credits"].fillna(1)

    # Workload categories
    level_order = ["High", "Standard", "Low"]

    recommended_df["predicted_credit_level"] = (
        recommended_df["predicted_credit_level"]
        .fillna("Standard")
        .astype(pd.CategoricalDtype(categories=level_order, ordered=True))
    )

    # Human friendly label for Y-axis
    recommended_df["workload"] = recommended_df["predicted_credit_level"]
    # Convert similarity to %
    recommended_df["interest_match"] = (recommended_df["similarity"] * 100).round(1)

    # Recommendation status
    recommended_df["is_recommended"] = recommended_df["key"].apply(
        lambda x: x in highlight_keys
    )
    recommended_df["Recommendation Status"] = recommended_df["is_recommended"].apply(
        lambda x: "Recommended Courses" if x else "Other Similar Courses"
    )


    WORKLOAD_ORDER = level_order
    # Base chart
    base = alt.Chart(recommended_df).encode(
        x=alt.X(
            "interest_match",
            title="Interest Match (%)",
            scale=alt.Scale(domain=[0, 100])
        ),
        y=alt.Y(
            "workload",
            title="Predicted Workload",
            sort=WORKLOAD_ORDER
        ),
        tooltip=[
            alt.Tooltip("key", title="Course ID"),
            alt.Tooltip("title", title="Title"),
            alt.Tooltip("interest_match", title="Interest Match (%)"),
            alt.Tooltip("workload", title="Workload Level"),
            alt.Tooltip("minimum_credits", title="Credits"),
        ]
    )

    # Highlight band across workload categories
    highlight_band = alt.Chart(
        pd.DataFrame({'workload': [preferred_credit_level]})
    ).mark_rect(
        opacity=0.2,
        color='red',
        height=40 # Visual height of the band
    ).encode(
        y=alt.Y('workload', sort=WORKLOAD_ORDER),
        tooltip=[]
    )

    # Points layer (bubbles)
    points = base.mark_circle().encode(
        color=alt.Color(
            "Recommendation Status",
            scale=alt.Scale(
                domain=["Recommended Courses", "Other Similar Courses"],
                range=["#0A7E36", "#7BAFD4"]
            ),
            legend=alt.Legend(title="")
        ),
        size=alt.Size(
            "minimum_credits",
            legend=None,
            scale=alt.Scale(range=[200, 1000])  # bubble sizes
        ),
        opacity=alt.value(0.85)
    )

    chart = (
        highlight_band + points
    ).properties(
        width="container", height=500, title="Workload vs Interest Match — Recommended Courses"
    ).configure_axis(
        labelFontSize=13,
        titleFontSize=14
    ).configure_title(
        fontSize=18
    ).interactive()

    return chart



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
            all_recommended_data, results_llm, preferred_credit_level = get_recommendations(user_input)

        st.success("Here are your recommended courses:")

        for rec in results_llm:
            expander_title = f"{rec['key']} — {rec['title']}"
            
            with st.expander(expander_title, expanded=True):
                st.markdown(f"** Minimum Credits:** {rec['minimum_credits']}")
                st.markdown(f"** Similarity:** {round(rec['similarity']*100, 1)}%")
                st.markdown(f"** Description:**")
                st.write(rec["description"])
                st.markdown(f"** Reasoning:**")
                st.write(rec["reasoning"])
                st.write("")

        # Plot the new bubble chart
        fig = plot_workload_vs_interest_highlighted(
            all_recommended_data, 
            highlight_list=results_llm, 
            preferred_credit_level=preferred_credit_level
        )

        st.altair_chart(fig, use_container_width=True)