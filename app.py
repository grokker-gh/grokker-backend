from fastapi import FastAPI
import json
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from pygrok import Grok
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def gen_new_prompt(log_line: str) -> str:
    return f"grokker, generate me a grok pattern for this: {log_line}"


open_ai_key = os.getenv('OPENAI_API_KEY')


class GrokPattern(BaseModel):
    pattern: str
    surity: float


class LogLine(BaseModel):
    text: str


@app.get("/")
def read_root():
    return 'Grokker backend is up and running!'


@app.post("/grok", description="Get the grok pattern for the given log line")
async def get_grok_pattern_for_plain_text(log_line: LogLine):
    grok_json = await get_gpt_response(log_line.text)
    response_from_gpt = json.loads(grok_json)
    return response_from_gpt


@app.post("/parse", description="Parse the log line with the given grok pattern")
async def parse_grok_pattern(pattern: str, log_line: str):
    try:
        grok = Grok(pattern)
        grok_match = grok.match(log_line)
        return grok_match
    except Exception as e:
        return {"status": "failed", "message": "failed parsing the log line with the grok pattern"}


client = AsyncOpenAI(
    api_key=open_ai_key
)


async def get_gpt_response(log_line: str) -> str:
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "grokker is a tool expert in writing grok pattern and here it will write grok patterns for your log lines."
            },
            {
                "role": "user",
                "content": "grokker, generate me a grok pattern for this: [Thu Jun 09 06:07:04 2005] [notice] LDAP: Built with OpenLDAP LDAP SDK "
            },
            {
                "role": "assistant",
                "content": "{\"pattern\": \"\\\\[%{DAY} %{MONTH} %{MONTHDAY} %{TIME} %{YEAR}\\\\] \\\\[%{WORD:loglevel}\\\\] %{WORD:source}: %{GREEDYDATA:message}\"}"
            },
            {
                "role": "user",
                "content": gen_new_prompt(log_line=log_line),
            }
        ],
        model='gpt-3.5-turbo',
    )

    for message in chat_completion.choices:
        print(message.message.content)

    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content
