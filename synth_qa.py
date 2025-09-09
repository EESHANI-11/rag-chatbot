import os, json, time, random
from openai import OpenAI
from config import PDF_LOCAL_PATH, PDF_NAME
import fitz
from dotenv import load_dotenv
load_dotenv()  # loads GROQ_API_KEY / GROQ_MODEL from .env

OUT = "data/synth_eval.jsonl"
N_MAX = 10

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

def load_pages(path):
    doc = fitz.open(path)
    for i in range(len(doc)):
        yield i+1, doc.load_page(i).get_text("text")

def main():
    pages = list(load_pages(PDF_LOCAL_PATH))
    print(f"Wrote {N_MAX} items to {OUT}") if not pages else None
    random.seed(7)
    sampled = random.sample(pages, k=min(N_MAX, len(pages)))

    with open(OUT, "w", encoding="utf-8") as f:
        for pg, text in sampled:
            prompt = (
                f"Document: {PDF_NAME}\n"
                f"Page {pg} text (excerpt):\n{text[:3000]}\n\n"
                "Create ONE factual question and its short answer grounded only in the text. "
                "Return as JSON with fields: question, answer."
            )
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.0, max_tokens=300
                )
                msg = resp.choices[0].message.content.strip()
                # try to parse a json object from response
                try:
                    obj = json.loads(msg)
                except Exception:
                    # simple fallback extraction
                    if "question" in msg.lower():
                        q = msg.split("question",1)[-1]
                    obj = {"question": msg.split("?")[0].strip()+"?", "answer": "See page context."}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except Exception as e:
                print("gen error:", e)
                time.sleep(1)
    print(f"Wrote {min(N_MAX, len(pages))} items to {OUT}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    main()
