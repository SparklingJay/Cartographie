import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import time

# Charger la clé API
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Fonction pour interroger le modèle
def query_model(task: str) -> str:
    prompt = f"""
Tu joues le rôle d’un expert en usages de l’IA à l’université.
Voici une tâche : "{task}".

Consigne :
- Décris en UNE phrase simple et actionnelle ce qu’un chatbot IA de type ChatGPT peut concrètement faire pour cette tâche. 
- Utilise un verbe d’action concret (ex. rédiger, corriger, résumer, traduire, structurer, etc).
- Si un chatbot n'est pas utile pour cette tâche, affiche "/". 
- Interdiction d’utiliser des formulations vagues et générales comme "aider à la gestion", "faciliter le suivi".
- La réponse doit être directement réutilisable dans un tableau.
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=11000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Erreur API pour la tâche '{task}': {e}")
        return "Erreur API"

if __name__ == "__main__":
    # Charger ton CSV existant
    input_file = "data_add/Carto.csv"
    output_file = "data_add/Carto_enrichi.csv"

    df = pd.read_csv(input_file, sep=";")

    # Vérification des colonnes attendues
    expected_cols = ["Famille de métiers", "Métier", "Tâche", "Impact potentiel"]
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"Le CSV doit contenir les colonnes : {expected_cols}")

    # Nouvelle colonne
    new_col = []
    for i, row in df.iterrows():
        task = row["Tâche"]
        print(f"➡️ Traitement de la tâche {i+1}/{len(df)} : {task}")
        answer = query_model(task)
        new_col.append(answer)
        time.sleep(2)  # éviter de spammer l’API

    df["Exemple"] = new_col

    # Sauvegarde
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ Fichier enrichi sauvegardé dans {output_file}")
