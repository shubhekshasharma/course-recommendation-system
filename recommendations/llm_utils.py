import os

LLM_LITE_TOKEN = os.getenv("LLM_LITE_TOKEN")
LLM_LITE_URL = os.getenv("LLM_LITE_URL")


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

    Aditionally, suggest extract relevant keywords and suggest additional keywords that can reflect user's interests. 
    For example, for "Oscars and film studies":
    extracted relevant keywords are: film
    suggested keywords are: acting, theatre, performing arts, etc.

    For example, for "Interstellar travel":
    extracted suggested keywords are: rockets, travelling, space, astronomy, etc. 

    Return the result in JSON format with the following fields (use the following as a template):
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

    prompt = f"""
    You are a course recommendation system for a university. You have access to a database of courses with their descriptions and credit hours.
    You have already recommended the following courses based on the user's input:
    {results.to_dict(orient='records')}

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
