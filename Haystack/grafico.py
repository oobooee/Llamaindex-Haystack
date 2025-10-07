import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# === Configurazione ===
cartella_csv = "result/test"
metriche = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
metagente_color = "#d62728"  # rosso usato nel grafico del professore

# === 1. Carica tutti i CSV ===
csv_files = glob.glob(os.path.join(cartella_csv, "*.csv"))

# === 2. Loop su ogni CSV ===
for file_path in csv_files:
    nome_file = os.path.basename(file_path)
    nome_base = os.path.splitext(nome_file)[0]
    
    # Carica il CSV
    df = pd.read_csv(file_path)
    df["System"] = "Metagente"
    
    # Crea figura con 3 subplot (1 per metrica)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metrica in enumerate(metriche):
        ax = axes[i]
        sns.violinplot(
            data=df,
            x="System",
            y=metrica,
            ax=ax,
            inner="box",
            cut=0,
            palette={"Metagente": metagente_color}
        )
        
        # Titolo con media
        media = df[metrica].mean()
        ax.set_title(f"{metrica}\nMedia: {media:.3f}", fontsize=12)

        # Tick asse Y come nel grafico ufficiale
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(0, 1.0)
    
    plt.suptitle(f"Distribuzione ROUGE – {nome_base}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # spazio per il titolo
    output_file = os.path.join(cartella_csv, f"{nome_base}_violinplot_____.png")
    plt.savefig(output_file)
    plt.close()
    print(f"✅ Salvato: {output_file}")
