import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from io import StringIO

# Charger la clé API depuis .env
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def query_model(model_name: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=8000
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Choisir le modèle que tu veux tester
    model = "gemma2-9b-it"

    # Charger le prompt depuis un fichier externe
    with open("prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

    print(f"--- Querying {model} ---")
    output = query_model(model, prompt)

    # Sauvegarde brute
    with open("data_gemma/output_raw.csv", "w", encoding="utf-8") as f:
        f.write(output)

    # Nettoyage : suppression lignes vides ou parasites
    clean_output = "\n".join([line for line in output.splitlines() if line.strip()])

    # Conversion en DataFrame
    try:
        df = pd.read_csv(StringIO(output))
        df.to_excel("data_gemma/table_metiers.xlsx", index=False)
        print("✅ Tableau exporté vers data/table_metiers.xlsx")
        print(df.head())
    except Exception as e:
        print("⚠️ Erreur lors de la lecture du CSV :", e)
