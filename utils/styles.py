from colorama import Fore, Style

def print_colored(text, color):
    """Affiche un texte coloré."""
    colors = {
        "red": Fore.RED,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "cyan": Fore.CYAN,
        "blue": Fore.BLUE,
    }
    print(colors.get(color, Fore.WHITE) + text + Style.RESET_ALL)
