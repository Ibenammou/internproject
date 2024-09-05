# Système de Reconnaissance Faciale pour Patients

## But du Projet

Ce projet vise à développer un système de reconnaissance faciale capable d'identifier les patients dans un environnement hospitalier. Lorsqu'un patient est reconnu grâce à la caméra, le système envoie automatiquement un message via WhatsApp à un médecin pour l'informer de la présence du patient. Le médecin peut alors décider si le patient peut entrer immédiatement ou doit attendre. Ce système vise à améliorer l'efficacité des opérations hospitalières et à faciliter la gestion des patients.

## Fonctionnalités

- **Reconnaissance Faciale** : Utilisation d'un modèle Keras pour identifier les patients en temps réel.
- **Notification au Médecin** : Envoi automatique d'un message WhatsApp au médecin concerné pour confirmer l'entrée du patient ou le temps d'attente.
- **Intégration de la Caméra** : Capture des images en direct via la caméra pour la reconnaissance faciale.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les éléments suivants :

- **Python 3.x**
- **OpenCV** : Pour la capture et le traitement des images.
- **Keras** : Pour le modèle de reconnaissance faciale.
- **NumPy** : Pour la manipulation des données.
- **pywhatkit** : Pour l'envoi des messages WhatsApp.

Installez les bibliothèques nécessaires avec :

```bash
pip install opencv-python keras numpy pywhatkit
