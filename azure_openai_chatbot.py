
import os
from openai import AzureOpenAI

endpoint = "Your Azure OpenAI endpoint, e.g., https://your-resource-name.openai.azure.com/"
apikey = "Your Azure OpenAI API key"


model_name = "Your model name, e.g., gpt-35-turbo"
deployment = "Your model name, e.g., gpt-35-turbo"

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint,
    api_key=apikey,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=deployment
)

print(response.choices[0].message.content)

