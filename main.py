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
    "You receive speech-to-text transcribed in a source language, which may contain "
    "regional dialect, slang, or colloquial phrasing. Detect that dialect/slang, "
    "normalize it to standard form in the source language preserving meaning and "
    "cultural nuance, then translate the standardized form into the target language "
    "with natural, fluent, non-literal phrasing. Respond ONLY as raw JSON in this shape: "
    '{"detected_dialect": string, "standard_form": string, "translation": string}.'
)


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/api/translate")
async def translate(payload: dict):

    from_lang = (payload or {}).get("from_lang")
    to_lang = (payload or {}).get("to_lang")
    text = (payload or {}).get("text")

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
        connect=8.0,
        read=20.0,
        write=8.0,
        pool=5.0
    )

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            async with httpx.AsyncClient(timeout=timeout) as client:

                response = await client.post(
                    f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                    headers={
                        "Content-Type": "application/json"
                    },
                    json={
                        "systemInstruction": {
                            "parts": [
                                {"text": SYSTEM_PROMPT}
                            ]
                        },
                        "contents": [
                            {
                                "parts": [
                                    {
                                        "text": user_content
                                    }
                                ]
                            }
                        ],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "maxOutputTokens": 500
                        }
                    }
                )

            print(
                f"[translate] Gemini status={response.status_code}",
                flush=True
            )

            data = response.json()

            # QUOTA EXCEEDED
            if response.status_code == 429:

                return {
                    "detected_dialect": "Unavailable",
                    "standard_form": text,
                    "translation":
                    "Daily translation limit reached. Please wait and try again."
                }

            # TEMPORARY OVERLOAD
            if response.status_code == 503:

                if attempt < max_attempts - 1:
                    await asyncio.sleep(2)
                    continue

                return {
                    "detected_dialect": "Unavailable",
                    "standard_form": text,
                    "translation":
                    "Translation service is busy. Try again shortly."
                }

            # OTHER API ERRORS
            if response.status_code != 200:

                return {
                    "detected_dialect": "Unavailable",
                    "standard_form": text,
                    "translation":
                    "Unable to translate right now."
                }

            raw = (
                data["candidates"][0]
                ["content"]["parts"][0]
                ["text"]
                .strip()
            )

            if raw.startswith("```"):
                raw = raw.replace("```json", "")
                raw = raw.replace("```", "").strip()

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

        except json.JSONDecodeError:

            return {
                "detected_dialect": "Unavailable",
                "standard_form": text,
                "translation":
                "Translation completed but output format was invalid."
            }

        except httpx.TimeoutException:

            return {
                "detected_dialect": "Unavailable",
                "standard_form": text,
                "translation":
                "Translation timed out. Try again."
            }

        except Exception as e:

            print(e, flush=True)

            return {
                "detected_dialect": "Unavailable",
                "standard_form": text,
                "translation":
                "Unexpected server error."
            }


@app.get("/api/health")
def health():

    return {
        "ok": True,
        "key_configured": bool(GEMINI_API_KEY)
    }
