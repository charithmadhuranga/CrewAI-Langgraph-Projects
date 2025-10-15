from  dotenv import load_env
import os

from google.auth import message
from openai import models

load_env(overide=True)

openaapi_key = os.getenv('OPENAPI_API_KEY')

if openaapi_key:
    print(f"OPEN API KEY is SetUP and key is start with {openaapi_key[:8]}")
else:
    print("OPENAPI KEY is not Setup")


openai = OpenAi(api_key=openaapi_key)

messages = [{"role":"user", "content":"what is 2+2"}]
response = openai.chat.completion.create(model="gpt-4o-mini",messages=messages)
print(response.choices[0].message.content)

question_ask = "please propose a challenging question to assees someone's IQ"
messages = [{"role":"user", "content":question_ask}]

response = openai.chat.completion.create(model="gpt-4o-mini",messages=messages)

question = response.choices[0].message.content
print(question)

messages_q = [{"role":"user", "content":question}]

answere = openai.chat.completion.create(model="gpt-4o-mini",messages=messages_q)
print(answere.choices[0].message.content)

