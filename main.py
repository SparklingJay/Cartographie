import os
import json
import pandas as pd
from tabulate import tabulate
from groq import Groq
from dotenv import load_dotenv

# Charger la clé API depuis .env
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def query_model(model_name: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Choisir le modèle que tu veux tester
    model = "openai/gpt-oss-120b"

    # Charger le prompt depuis un fichier externe
    with open("prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

    print(f"--- Querying {model} ---")
    output = query_model(model, prompt)

    # Sauvegarde brute
    with open("data/output_openai.txt", "w", encoding="utf-8") as f:
        f.write(output)

    # Tentative de parsing JSON (si le modèle respecte bien le format)
    try:
        data = json.loads(output)
        df = pd.DataFrame(data)
        df.to_excel("data/table_metiers.xlsx", index=False)
        print(tabulate(df, headers="keys", tablefmt="grid"))
    except json.JSONDecodeError:
        print("⚠️ Impossible de parser la sortie en JSON. Vérifie la réponse brute.")
