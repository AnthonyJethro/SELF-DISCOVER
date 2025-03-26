import os
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
import requests  # For interacting with the LM Studio server

load_dotenv()

generation_config = {
    "temperature": 0,
    "top_k": 1,
    "max_output_tokens": 4000,
}


class LLM:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.model = self.create_model(model_name)

    def create_model(self, model_name):
        match model_name:
            case "gemini-pro-vision":
                genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
                return genai.GenerativeModel(model_name)
            case "gemini-pro":
                genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
                return genai.GenerativeModel(
                    model_name, generation_config=generation_config
                )
            case "OpenAI":
                return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            case "Gemma3":
                # No initialization needed for LM Studio; requests will be sent directly
                return None
            case _:
                print("Not Implemented")

    def __call__(self, prompt, image=None):
        if self.model_name == "gemini-pro-vision":
            response = self.model.generate_content([image, prompt])
        elif self.model_name == "gemini-pro":
            response = self.model.generate_content(prompt)
            return response.text
        elif self.model_name == "OpenAI":
            res = self.model.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "user", "content": f"{prompt}"},
                ],
                temperature=0,
            )
            return res.choices[0].message.content
        elif self.model_name == "Gemma3":
            # Send a request to the LM Studio server
            url = "http://localhost:1234/v1/completions"  # LM Studio's OpenAI-compatible endpoint
            headers = {"Content-Type": "application/json"}
            data = {
                "model": "gemma-3-12b-it",  # Use the model name from LM Studio
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 4000,
            }
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                return response.json().get("choices", [{}])[0].get("text", "").strip()
            else:
                raise Exception(f"Error from LM Studio: {response.text}")