import os
import json
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InvoiceRequest(BaseModel):
    invoice_text: str

class InvoiceExtraction(BaseModel):
    invoice_no: Optional[str] = None
    date: Optional[str] = None
    vendor: Optional[str] = None
    amount: Optional[float] = None
    tax: Optional[float] = None
    currency: Optional[str] = None

@app.post("/extract", response_model=InvoiceExtraction)
def extract_invoice_data(payload: InvoiceRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing in Render")

    # Fixed schema required by the problem statement
    invoice_schema = {
        "type": "OBJECT",
        "properties": {
            "invoice_no": {"type": "STRING"},
            "date": {"type": "STRING"},
            "vendor": {"type": "STRING"},
            "amount": {"type": "NUMBER"},
            "tax": {"type": "NUMBER"},
            "currency": {"type": "STRING"}
        }
    }

    system_instruction = (
        "You are an expert invoice data extractor. Extract the details from the provided raw text.\n"
        "CRITICAL RULES:\n"
        "1. 'date' MUST be converted to ISO format (YYYY-MM-DD). Example: '15 March 2026' -> '2026-03-15'.\n"
        "2. 'amount' is the subtotal BEFORE tax. Do not include tax in this field.\n"
        "3. 'tax' is ONLY the tax amount.\n"
        "4. 'currency' should be the standard 3-letter code (e.g., 'INR', 'USD').\n"
        "5. If a field cannot be found in the text, you must omit it so it returns null."
    )

    url = "https://aipipe.org/geminiv1beta/models/gemini-2.5-flash:generateContent"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload_data = {
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": [
            {
                "parts": [{"text": payload.invoice_text}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": invoice_schema
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload_data)
        response.raise_for_status() 
        
        data = response.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        # Strip markdown in case Gemini hallucinates backticks
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("\n", 1)[0]
            
        return json.loads(raw_text.strip())
        
    except Exception as e:
        error_msg = str(e)
        if isinstance(e, requests.exceptions.HTTPError):
            error_msg += f" - Response Text: {response.text}"
        print(f"CRITICAL EXTRACTION ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
