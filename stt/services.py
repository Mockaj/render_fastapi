import json
from openai import OpenAI
from typing import List

from stt.config import settings
from stt.prompt import user_prompt
# Load the OpenAI API key from environment variables or .env.rag file

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def transcribe_audio_file(filepath: str) -> str:
    with open(filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            response_format="text",
            language="cs",
            prompt=user_prompt
        )
    return transcription


def format_prompt(text: str) -> List[dict]:
    return [
        {
            "role": "system",
            "content": """
__ASK__
You are a professional copywriter tasked with correcting misspellings and grammatical errors in a text containing an ophthalmologist's diagnosis. The text may include technical terms and recommendations. Your goal is to produce a corrected version of the text and extract any recommendations mentioned.

__CONTEXT__
The text is in Czech language and is produced by speech to text conversion. The conversion is not entirely precise and thus there are some mistakes. It is your task to correct these mistakes. It is important to consider the fact that it is an ophthalmologist diagnosis that is included in the provided text so correct these mistakes in such a way that the diagnosis makes sense.

__CONSTRAINTS__
Follow these steps to complete the task:

1. Carefully read through the entire text.

2. Correct any misspellings and grammatical errors you find in the text. Pay close attention to sentence structure, punctuation, and word usage.

3. When encountering technical terms related to ophthalmology, do not alter them unless you are absolutely certain they are misspelled. If you are unsure about a technical term, leave it unchanged.

4. As you review the text, look for any mentions of recommendations. These might be introduced with phrases like "I recommend," "it is advised," or "the patient should."

5. If you find any recommendations, extract them and prepare to list them separately.

6. If there are no recommendations mentioned in the text, include an empty array for the "recommendations" key, like this:

{
  "text": "Your corrected version of the text goes here",
  "recommendations": []
}

7. If you encounter any words that you are unsure about, especially if they might be technical terms, leave them unchanged in the corrected text.

8. Ensure that the corrected text in the "text" field does not include the recommendations you've extracted. The recommendations should only appear in the "recommendations" array.

9. Answer purely in Czech language!

__OUTPUT_FORMAT__
After correcting the text and identifying any recommendations, prepare your output in the following JSON format:

{
  "text": "Your corrected version of the text goes here, excluding any recommendations",
  "recommendations": ["Recommendation 1", "Recommendation 2", "etc."]
}
"""
        },
        {
            "role": "user",
            "content": f"Prosím oprav podle instrukcí tento text: {text}"
        }
    ]


def invoke_llm(messages: List[dict]) -> dict:
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4.1-mini",
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content


def parse_output(response: str) -> str:
    response_json = json.loads(response)

    if response_json.get("recommendations") == []:
        response_text = response_json.get("text")
    else:
        recommendations = "\nDoporučení:\n" + "\n".join([f"• {rec}" for rec in response_json.get("recommendations")])
        response_text = response_json.get("text") + recommendations

    return response_text


def run_pipeline(text: str) -> str:
    messages = format_prompt(text)
    response = invoke_llm(messages)
    return parse_output(response)
