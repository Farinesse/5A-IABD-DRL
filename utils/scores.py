import csv

def save_game_history(history, file_name="results/game_history.csv"):
    """Sauvegarde l'historique des scores."""
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(history)