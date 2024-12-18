# Classification des Chansons de Bodo et Popy avec LSTM

Ce projet consiste à classifier des chansons des artistes malgaches Bodo et Popy en utilisant un modèle basé sur un réseau de neurones LSTM (Long Short-Term Memory). Le modèle apprend à différencier les chansons en fonction des paroles grâce à un traitement de texte et des techniques d'apprentissage profond.

## Fonctionnalités

- Vectorisation des textes via `CountVectorizer`.
- Classification multi-classes basée sur un réseau LSTM avec des paramètres personnalisables (dimension d'embedding, nombre de couches, dropout, etc.).
- Entraînement et validation du modèle avec gestion des pertes et de la précision.
- Prédiction des classes (artiste probable) pour des textes non vus.

## Structure de la Classe

### `LSTMTextClassifier`
La classe `LSTMTextClassifier` encapsule toutes les étapes nécessaires pour préparer les données, construire le modèle, entraîner et effectuer des prédictions. Voici une description succincte des principales méthodes :

- **`__init__`** : Initialise les paramètres du modèle (dimension d'embedding, dimension des états cachés, nombre de couches, taux de dropout).
- **`prepare_data`** : Prépare et vectorise les textes et étiquettes pour les convertir en tenseurs PyTorch.
- **`build_model`** : Définit l'architecture du modèle LSTM comprenant une couche d'embedding linéaire, une couche LSTM et des couches entièrement connectées pour la sortie.
- **`train`** : Entraîne le modèle sur les données d'entrée et valide sa performance.
- **`predict`** : Prédit les classes des nouveaux textes en utilisant le modèle entraîné.

### Exemple d'Architecture

L'architecture du modèle inclut :
- Une couche d'embedding pour transformer les données d'entrée.
- Une couche LSTM pour capturer les relations temporelles dans les données textuelles.
- Des couches entièrement connectées pour classifier les données en classes.

---

## Configuration et Installation

1. Clonez ce projet :
   ```bash
   git clone https://github.com/votre-repository/deep_learning_classification.git
   cd deep_learning_classification
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Assurez-vous que PyTorch est installé avec le support CUDA si votre système dispose d'un GPU :
   ```bash
   pip install torch torchvision
   ```

---

## Utilisation

### Préparation des Données

Les données doivent être fournies sous forme de deux listes :
- `texts` : Une liste contenant les paroles des chansons.
- `labels` : Une liste contenant les noms des artistes correspondants (par exemple, "Bodo" ou "Popy").

```python
from lstm_classifier import LSTMTextClassifier

# Exemple de données
texts = ["Paroles chanson 1", "Paroles chanson 2", "Paroles chanson 3"]
labels = ["Bodo", "Popy", "Bodo"]

# Initialisation et préparation
classifier = LSTMTextClassifier()
X, y = classifier.prepare_data(texts, labels)
```

### Entraînement du Modèle

```python
# Entraînez le modèle
classifier.train(X, y, epochs=50, lr=0.001, batch_size=32)
```

### Prédiction

```python
# Prédiction pour de nouvelles chansons
new_texts = ["Paroles inconnues"]
predictions = classifier.predict(new_texts)
print(predictions)  # Résultat attendu : ["Bodo"] ou ["Popy"]
```

---

## Résultats Attendus

- **Précision du modèle** : La précision sur les données de validation dépend de la qualité des données textuelles et des hyperparamètres choisis.
- **Prédictions** : Le modèle retournera le nom de l'artiste (Bodo ou Popy) basé sur les paroles fournies.

---

## Technologies Utilisées

- **Langage** : Python
- **Bibliothèques** :
  - PyTorch (pour le modèle LSTM)
  - Scikit-learn (pour la vectorisation des textes et l'encodage des étiquettes)
  - Pandas et NumPy (pour la manipulation des données)

---

## Auteurs

Ce projet a été développé dans le cadre de l'exploration des applications de l'apprentissage profond à la musique malgache.

ANDRIAMANJATO Lahatriniavo Fandresena
