from exam.llm_provider import get_llm

def main():
    print("Test della funzione gemini_api_key...")

    # 1. Ottenere l'LLM con il modello di default
        # Potrebbe chiedere la chiave API la prima volta
    default_llm = get_llm()
    print(f"\nOttenuto LLM (default):")
    print(f"  Modello: {default_llm}")
    print(f"  Classe: {type(default_llm)}")



if __name__ == "__main__":
    main()