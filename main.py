import os
import json
import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
 
app = FastAPI(title="DialectIQ Backend")
 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
 
SYSTEM_PROMPT = (
    "You are the dialect-aware translation engine inside an app called DialectIQ. "
    "You receive speech-to-text transcribed in a source language, which may contain "
    "regional dialect, slang, or colloquial phrasing. Detect that dialect/slang, "
    "normalize it to standard form in the source language preserving meaning and "
    "cultural nuance, then translate the standardized form into the target language "
    "with natural, fluent, non-literal phrasing. Respond with ONLY a raw JSON object, "
    "no markdown fences, no commentary, in exactly this shape: "
    '{"detected_dialect": string, "standard_form": string, "translation": string}. '
    "For detected_dialect: name the specific regional dialect or slang style if you can "
    "identify one (for example 'Madurai Tamil colloquial'), otherwise use \"Standard\"."
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
        return JSONResponse({"error": "missing_fields"}, status_code=400)
 
    if not GEMINI_API_KEY:
        return JSONResponse({"error": "server_missing_api_key"}, status_code=500)
 
    user_content = (
        f"Source language: {from_lang}\n"
        f"Target language: {to_lang}\n"
        f'Transcribed speech: "{text}"'
    )
 
    timeout = httpx.Timeout(connect=8.0, read=20.0, write=8.0, pool=5.0)
    print(f"[translate] calling Gemini: from={from_lang} to={to_lang} text={text[:60]!r}", flush=True)
 
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                headers={"content-type": "application/json"},
                json={
                    "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "contents": [{"parts": [{"text": user_content}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "maxOutputTokens": 1000,
                    },
                },
            )
        print(f"[translate] Gemini responded with status {resp.status_code}", flush=True)
 
        data = resp.json()
 
        if resp.status_code != 200:
            print(f"[translate] Gemini error body: {data}", flush=True)
            return JSONResponse(
                {"error": "gemini_api_error", "status": resp.status_code, "detail": data},
                status_code=502,
            )
 
        raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
 
        parsed = json.loads(raw)
        print(f"[translate] success: {parsed}", flush=True)
        return parsed
 
    except json.JSONDecodeError as exc:
        print(f"[translate] JSON parse failed. Raw model output: {raw!r}. Error: {exc}", flush=True)
        return JSONResponse({"error": "could_not_parse_model_output"}, status_code=502)
    except httpx.TimeoutException as exc:
        print(f"[translate] TIMEOUT talking to Gemini: {exc!r}", flush=True)
        return JSONResponse({"error": "gemini_timeout", "detail": str(exc)}, status_code=504)
    except Exception as exc:
        print(f"[translate] UNEXPECTED ERROR: {type(exc).__name__}: {exc}", flush=True)
        return JSONResponse({"error": "server_error", "detail": str(exc)}, status_code=500)
 
 
@app.get("/api/health")
def health():
    return {"ok": True, "key_configured": bool(GEMINI_API_KEY)}
