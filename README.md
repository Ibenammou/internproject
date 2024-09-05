# Système de Reconnaissance Faciale pour Patients

## But du Projet

Ce projet vise à développer un système de reconnaissance faciale capable d'identifier les patients dans un environnement hospitalier. Lorsqu'un patient est reconnu grâce à la caméra, le système envoie automatiquement un message via WhatsApp à un médecin pour l'informer de la présence du patient. Le médecin peut alors décider si le patient peut entrer immédiatement ou doit attendre. Ce système vise à améliorer l'efficacité des opérations hospitalières et à faciliter la gestion des patients.

## Fonctionnalités

- **Reconnaissance Faciale** : Utilisation d'un modèle Keras pour identifier les patients en temps réel.
![Reconnaissance Faciale](https://github.com/Ibenammou/internproject/blob/master/image1.jpg)
- **Notification au Médecin** : Envoi automatique d'un message WhatsApp au médecin concerné pour confirmer l'entrée du patient ou le temps d'attente.
![Notification au Médecin](https://github.com/Ibenammou/internproject/blob/master/image%202.jpg)
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
pip install opencv-python keras numpy pywhatkit.


Les composants principaux de ce projet sont : un Raspberry Pi 4, une carte électronique permettant d'intégrer le code avec les autres composants, une caméra Microsoft pour détecter les visages des patients, des résistances pour protéger la carte, une breadboard servant de support de travail, des fils de connexion reliant les différents composants, et un écran LCD pour afficher les messages.
