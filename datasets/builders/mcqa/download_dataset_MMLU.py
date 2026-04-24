from datasets import load_dataset, DatasetDict
from huggingface_hub import login, delete_repo

# Login
# Le tue categorie
categorie_scelte = {
    #  "elementary_mathematics", #0.43
     "high_school_biology", #0.64
    "college_medicine", #0.53
    "professional_medicine", #0.49
    # "high_school_statistics", #0.5
    # "high_school_mathematics", #0.35 --
    # "conceptual_physics", #0.49
    "clinical_knowledge", #0.57
    # "electrical_engineering", #0.55
    # "machine_learning", #0.34 --> 0.39
    # "college_physics", #0.35
    # "abstract_algebra", #0.31 --
    # "college_chemistry", #0.47
    # "college_computer_science", #0.4 --> 0.45
    # "college_mathematics", #0.32
    # "computer_security" #0.59 -->0.67
}

# Nome del repository DA MANTENERE
REPO_NAME = "VinceEPFL/mmlu_med_only_subset"

print("  ATTENZIONE: Questo sovrascriverà il dataset esistente!")
response = input("Vuoi continuare? (y/n): ")
if response.lower() != 'y':
    print("Operazione annullata.")
    exit()

# 1. Carica il dataset originale - SOLO split test
print("\n1. Carico il dataset originale ema1234/mmlu...")
original_dataset = load_dataset("ema1234/mmlu", split="test")
print(f"    Caricato: {len(original_dataset)} esempi")
print(f"    Features: {original_dataset.features}")

# 2. Filtra mantenendo la struttura ESATTA
print("\n2. Filtro per le categorie scelte...")
filtered_dataset = original_dataset.filter(
    lambda x: x["subject"] in categorie_scelte
)
print(f"    Filtrati: {len(filtered_dataset)} esempi")

# 3. Verifica che la struttura sia identica
assert original_dataset.features == filtered_dataset.features, "ERRORE: La struttura è cambiata!"
print("    Struttura verificata - identica all'originale")

# 4. Crea DatasetDict con SOLO lo split test
print("\n3. Preparo il dataset per l'upload...")
final_dataset = DatasetDict({
    "test": filtered_dataset  # SOLO test, come l'originale
})

# 5. Elimina il vecchio dataset se esiste (opzionale)
try:
    print(f"\n4. Elimino il vecchio dataset {REPO_NAME}...")
    delete_repo(repo_id=REPO_NAME, repo_type="dataset")
    print("   ✓ Vecchio dataset eliminato")
except:
    print("     Vecchio dataset non trovato o non eliminabile")

# 6. Upload del nuovo dataset
print(f"\n5. Upload del nuovo dataset su {REPO_NAME}...")
final_dataset.push_to_hub(
    REPO_NAME,
    private=False,
    commit_message="Filtered MMLU dataset - exact copy with selected subjects only"
)

print(f"\n FATTO! Dataset aggiornato: https://huggingface.co/datasets/{REPO_NAME}")

# 7. Verifica finale
print("\n6. Verifica finale...")
try:
    # Test di caricamento come fa lighteval
    test_dataset = load_dataset(REPO_NAME, split="test")
    print(f"    Dataset caricabile con split='test': {len(test_dataset)} esempi")
    
    # Test con tutto il dataset
    test_full = load_dataset(REPO_NAME)
    print(f"    Dataset caricabile completo: splits = {list(test_full.keys())}")
    
    # Verifica features
    print(f"    Features: {test_dataset.features}")
    
    # Esempio
    print(f"\n    Esempio:")
    example = test_dataset[0]
    for key, value in example.items():
        print(f"      {key}: {value}")
        
except Exception as e:
    print(f"   ❌ ERRORE nel test: {e}")
