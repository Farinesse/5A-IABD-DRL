import time
from utils.styles import print_colored

def rolling_dice_animation(dice):
    """Affiche une animation de lancer de dés."""
    print_colored("🎲 Lancer les dés...", "yellow")
    for i in range(3):
        print_colored("..." + "." * (i + 1), "cyan")
        time.sleep(0.5)
    print_colored(f"🎲 Résultat des dés : {dice}", "green")
