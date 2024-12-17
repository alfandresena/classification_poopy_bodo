
import pandas as pd
from lstm import LSTMTextClassifier
import torch

def main():
    # Configuration
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    
    try:
        df = pd.read_csv("bodo_poopy.csv")
        # Extraire les 5 premières lignes et les 5 dernières lignes
        top_5 = df.head(5)
        bottom_5 = df.tail(5)

        # Combiner les deux extraits dans un seul tableau (DataFrame)
        combined_df = pd.concat([top_5, bottom_5])

        # Sauvegarder dans un nouveau fichier CSV
        output_file_path = 'extrait_premieres_et_dernires_lignes.csv'
        combined_df.to_csv(output_file_path, index=False)

        print("Les 5 premières et 5 dernières lignes ont été extraites et sauvegardées.")

        # Supprimer les 5 premières et 5 dernières lignes
        df_cleaned = df.iloc[5:-5]

        # Sauvegarder le résultat dans un nouveau fichier CSV
        output_file_path = 'fichier_sans_5_premieres_et_dernieres_lignes.csv'
        df_cleaned.to_csv(output_file_path, index=False)
        print("Les 5 premières et 5 dernières lignes ont été supprimées.")

        # Chargement des données
        data = pd.read_csv('fichier_sans_5_premieres_et_dernieres_lignes.csv')

        # Initialisation du classificateur LSTM
        clf = LSTMTextClassifier(
            embedding_dim=100,   # Taille de l'embedding
            hidden_dim=128,      # Dimension cachée de LSTM
            num_layers=2,        # Nombre de couches LSTM
            dropout=0.3          # Taux de dropout
        )
        
        # Préparation des données - MODIFICATION ICI
        X, y = clf.prepare_data(data['text'], data['Label'])  # Utilisez 'Label' au lieu de 'category'
        
        # Entraînement
        clf.train(
            X, 
            y, 
            epochs=50,           # Nombre d'époques
            lr=0.001,            # Learning rate
            batch_size=32        # Taille du batch
        )
        
        # Exemples de prédiction
        test_data = pd.read_csv('extrait_premieres_et_dernires_lignes.csv')
        test_texts = test_data['text'].tolist()

        
        # Faire des prédictions
        predictions = clf.predict(test_texts)
        
        # Afficher les prédictions
        print("\nPredictions:")
        for text, prediction in zip(test_texts, predictions):
            print(f"Catégorie: {prediction}\n")
        
        # Sauvegarde du modèle
        if hasattr(clf, 'model') and clf.model is not None:
            torch.save(clf.model.state_dict(), 'bodo_poopy_model.pth')
            print("Modèle sauvegardé avec succès.")
    
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()