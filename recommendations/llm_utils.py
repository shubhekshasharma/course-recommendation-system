import os
import streamlit as st

LLM_LITE_TOKEN = st.secrets["LLM_LITE_TOKEN"]
LLM_LITE_URL = st.secrets["LLM_LITE_URL"]


def get_llm_client():
    from openai import OpenAI

    client = OpenAI(
        api_key=LLM_LITE_TOKEN,
        base_url=LLM_LITE_URL
    )
    return client



def get_course_workload_and_additional_keywords(client, user_input: str):

    prompt = f"""
    You are a course recommendation system for a university. You have access to a database of courses with their descriptions and credit hours.
    For this task, you will only determine the workload user prefers based on their input.

    Note when determining the workload, we have 3 categories:
    - Low
    - Standard
    - High

    Do not assume other workload categories exist.
    The user's input is as follows:
    {user_input}

    Your tasks:
    1. Determine the workload preference (must be one of: Low, Standard, High)
    2. Extract and infer academic interest keywords for course matching

    WORKLOAD CATEGORIES:
    - Low: Light workload, fewer courses, more flexibility
    - Standard: Moderate workload, typical course load
    - High: Heavy workload, many courses, intensive study

    IMPORTANT: If the user does not mention anything about workload, effort level, time commitment, or course load, default to "Standard".

    INTEREST KEYWORDS EXTRACTION:
    Extract comprehensive academic keywords that represent the user's interests. These keywords will be used to match against course descriptions and titles.

    Guidelines for keyword extraction:
    - Include ALL relevant subject areas, topics, and fields mentioned or implied
    - Add related academic disciplines and subdisciplines
    - Include synonyms and related terminology commonly used in course catalogs
    - Focus ONLY on academic/subject matter keywords (exclude workload preferences, time commitments, or vague terms)
    - Use specific academic terms that would appear in course titles and descriptions
    - Include technical terminology, techniques, and methodologies specific to the subject
    - Add variations and related forms of words (e.g., "photograph", "photography", "photographic", "photographer")
    - Think about what words would appear in a dedicated course focused entirely on this subject
    - Prioritize depth over breadth - include 15-20 highly relevant keywords rather than generic terms

    Examples:
    Input: "I'm interested in the Oscars and film studies"
    Keywords: ["film", "cinema", "movies", "acting", "theatre", "performing arts", "drama", "entertainment", "media studies", "visual arts", "filmmaking", "cinematography", "directing", "screenwriting", "film production", "film theory", "film history"]

    Input: "Interstellar travel and space exploration"
    Keywords: ["space", "astronomy", "astrophysics", "aerospace", "rockets", "spacecraft", "cosmology", "planetary science", "physics", "engineering", "exploration", "interstellar", "orbital", "satellites", "propulsion"]

    Input: "Machine learning and AI for healthcare applications"
    Keywords: ["machine learning", "artificial intelligence", "AI", "healthcare", "medical", "health", "data science", "computer science", "biomedical", "algorithms", "neural networks", "deep learning", "clinical", "medicine", "predictive modeling", "diagnostics"]

    Input: "I like photography"
    Keywords: ["photography", "photographic", "camera", "digital photography", "portrait", "documentary photography", "photojournalism", "image making", "visual storytelling", "photo editing", "composition", "lighting", "darkroom", "photographer", "photographs", "images", "picture taking", "photo critique"]

    Return the result in JSON format:
    {{
        "category": "Low" | "Standard" | "High",
        "reasoning": "Explanation of why this workload category was chosen based on the user's input.", 
        "interest_key_words": ["film", "acting", "theatre", "performing arts"]
    }}
    """

    response = client.chat.completions.create(
        model="GPT 4.1 Mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that recommends course for a university."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=300
    )

    if hasattr(response, "choices") and len(response.choices) > 0:
        print(response.choices[0].message.content)
    else:
        print("No valid choices found in response:")
        print(response)
        print("Error: Model returned no output.")

    return response.choices[0].message.content



def get_recommendations_reasoning(client, results, user_input: str):

    records = results.to_dict(orient='records')

    prompt = f"""
    You are a course recommendation system for a university.
    Our ML models recommended the following courses based on the user's input:
    {records}

    Now, provide a brief reasoning for why each course was recommended based on the user's input below:
    {user_input}

    Return the result in JSON format as a list of objects with the following fields:
    {{
        "key": "Course ID",
        "title": "Course Title",
        "description": "Course Description", 
        "minimum_credits": "Min credits",
        "similarity: Similarity score in float with 2 decimal places,
        "reasoning": "Explanation of why this course was recommended based on the user's input."
    }}

    Make a limit of 4 recommendations only in your result.
    If you notice a long course description, summarize it. 

    Ensure it is a valid JSON string. 

    Note: if there are no input records then do not hallucianate. In that case you should return no results. 
    """

    response = client.chat.completions.create(
        model="GPT 4.1 Mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that recommends course for a university."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=10000
    )

    if hasattr(response, "choices") and len(response.choices) > 0:
        print(response.choices[0].message.content)
    else:
        print("No valid choices found in response:")
        print(response)
        print("Error: Model returned no output.")

    return response.choices[0].message.content
