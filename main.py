import os
import json
import asyncio
import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="DialectIQ Backend")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

SYSTEM_PROMPT = (
    "You are the dialect-aware translation engine inside an app called DialectIQ. "
    "You receive speech-to-text transcribed in a source language which may contain "
    "regional dialect, slang, or colloquial phrasing. Detect the dialect, normalize "
    "to standard form preserving meaning, then translate naturally. "
    "Respond ONLY as JSON in this exact shape: "
    '{"detected_dialect": string, "standard_form": string, "translation": string}.'
)


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


@app.post("/api/translate")
async def translate(payload: dict):

    from_lang = payload.get("from_lang")
    to_lang = payload.get("to_lang")
    text = payload.get("text")

    if not all([from_lang, to_lang, text]):
        return JSONResponse(
            {"error": "missing_fields"},
            status_code=400
        )

    if not GEMINI_API_KEY:
        return JSONResponse(
            {"error": "server_missing_api_key"},
            status_code=500
        )

    user_content = (
        f"Source language: {from_lang}\n"
        f"Target language: {to_lang}\n"
        f'Transcribed speech: "{text}"'
    )

    timeout = httpx.Timeout(
        connect=8,
        read=20,
        write=8,
        pool=5
    )

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            async with httpx.AsyncClient(
                timeout=timeout
            ) as client:

                response = await client.post(
                    f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                    headers={
                        "Content-Type":
                        "application/json"
                    },
                    json={
                        "systemInstruction": {
                            "parts": [
                                {
                                    "text":
                                    SYSTEM_PROMPT
                                }
                            ]
                        },
                        "contents": [
                            {
                                "parts": [
                                    {
                                        "text":
                                        user_content
                                    }
                                ]
                            }
                        ],
                        "generationConfig": {
                            "responseMimeType":
                            "application/json",

                            "maxOutputTokens":
                            500
                        }
                    }
                )

            print(
                f"Gemini status: {response.status_code}",
                flush=True
            )

            data = response.json()

            # QUOTA HIT
            if response.status_code == 429:

                if attempt < max_attempts - 1:

                    print(
                        "Retrying after quota...",
                        flush=True
                    )

                    await asyncio.sleep(5)

                    continue

                return {
                    "detected_dialect":
                    "Unavailable",

                    "standard_form":
                    text,

                    "translation":
                    "Translation temporarily unavailable. Please try again."
                }

            # OVERLOADED
            if response.status_code == 503:

                if attempt < max_attempts - 1:

                    await asyncio.sleep(2)

                    continue

                return {
                    "detected_dialect":
                    "Unavailable",

                    "standard_form":
                    text,

                    "translation":
                    "Translation service busy."
                }

            # OTHER ERRORS
            if response.status_code != 200:

                return {
                    "detected_dialect":
                    "Unavailable",

                    "standard_form":
                    text,

                    "translation":
                    "Unable to translate."
                }

            raw = (
                data["candidates"][0]
                ["content"]["parts"][0]
                ["text"]
                .strip()
            )

            if raw.startswith("```"):

                raw = (
                    raw
                    .replace(
                        "```json",
                        ""
                    )
                    .replace(
                        "```",
                        ""
                    )
                    .strip()
                )

            parsed = json.loads(raw)

            return {
                "detected_dialect":
                parsed.get(
                    "detected_dialect",
                    "Standard"
                ),

                "standard_form":
                parsed.get(
                    "standard_form",
                    text
                ),

                "translation":
                parsed.get(
                    "translation",
                    text
                )
            }

        except Exception as e:

            print(
                str(e),
                flush=True
            )

            return {
                "detected_dialect":
                "Unavailable",

                "standard_form":
                text,

                "translation":
                "Server error."
            }


@app.get("/api/health")
def health():

    return {
        "ok": True,
        "key_configured":
        bool(
            GEMINI_API_KEY
        )
    }
