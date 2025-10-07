from haystack import component
import pandas as pd
@component
class CSVRowExtractor:
    """
    Estrae 'readme_text' e 'description' da una riga del CSV (come dizionario).
    """
    @component.output_types(readme_text=str, description=str)
    def run(self, row: dict):
        df = pd.read_csv("data/train_data1.csv")
        return {
            "readme_text": row.get("readme", ""),
            "description": row.get("description", "")
        }
