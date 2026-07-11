import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from google import genai
from google.genai import types

app = FastAPI()

# Rule 4: CORS must be enabled
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is missing.")
client = genai.Client(api_key=api_key)

# API Spec Input: {"invoice_text": "..."}
class InvoiceRequest(BaseModel):
    invoice_text: str

# API Spec Output: Exact 6 keys. Optional ensures 'null' is used if missing (Rule 1).
class InvoiceExtraction(BaseModel):
    invoice_no: Optional[str] = None
    date: Optional[str] = None
    vendor: Optional[str] = None
    amount: Optional[float] = None
    tax: Optional[float] = None
    currency: Optional[str] = None

@app.post("/extract", response_model=InvoiceExtraction)
async def extract_invoice_data(payload: InvoiceRequest):
    # Enforcing Rules 2 and 3 through strict system prompting
    system_instruction = (
        "You are an expert invoice data extractor. Extract the details from the provided raw text.\n"
        "CRITICAL RULES:\n"
        "1. 'date' MUST be converted to ISO format (YYYY-MM-DD). Example: '15 March 2026' -> '2026-03-15'.\n"
        "2. 'amount' is the subtotal BEFORE tax. Do not include tax in this field.\n"
        "3. 'tax' is ONLY the tax amount.\n"
        "4. 'currency' should be the standard 3-letter code (e.g., 'INR', 'USD').\n"
        "5. If a field cannot be found in the text, you must omit it so it returns null."
    )

    try:
        # Using Structured Outputs to guarantee the exact JSON format
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=payload.invoice_text,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InvoiceExtraction,
                system_instruction=system_instruction,
                temperature=0.0 # Keep it deterministic 
            )
        )
        
        # Parse the guaranteed JSON string from Gemini and return it
        extracted_data = json.loads(response.text)
        return extracted_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))