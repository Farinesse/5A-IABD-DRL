import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment.line_word import LineWorld
from environment.grid_word import GridWorld

from environment.tictactoe import TicTacToe
import random

def main():
    line_world = LineWorld(length=5)
    line_world.display()

    # Boucle de jeu principale
    while True:
        print("\nChoisissez une action :")
        print("0: Rester sur place")
        print("1: Aller à gauche")
        print("2: Aller à droite")
        action = int(input("Votre choix (0, 1, 2) : "))

        if action not in line_world.available_actions():
            print("Action invalide ! Veuillez choisir parmi les actions disponibles.")
            continue

        next_state, reward, done = line_world.step(action)
        print(f"État suivant: {next_state}, Récompense: {reward}, Terminé: {done}")
        line_world.display()

        if done:
            print("Jeu terminé ! L'agent est dans un état terminal.")
            break

            # Initialiser l'environnement GridWorld avec une grille de 4x4


    grid_world = GridWorld(width=5, height=5)

    # Afficher l'état initial de l'agent
    grid_world.display()

    # Boucle de jeu principale
    while True:
        print("\nChoisissez une action :")
        print("0: Aller en haut")
        print("1: Aller en bas")
        print("2: Aller à gauche")
        print("3: Aller à droite")
        action = int(input("Votre choix (0, 1, 2, 3) : "))

        if action not in grid_world.available_actions():
            print("Action invalide ! Veuillez choisir parmi les actions disponibles.")
            continue

        next_state, reward, done = grid_world.step(action)
        print(f"État suivant: {next_state}, Récompense: {reward}, Terminé: {done}")
        grid_world.display()

        if done:
            print("Jeu terminé ! L'agent est dans un état terminal.")
            break

def play(game: TicTacToe, player_x, player_o, print_game=True):
    if print_game:
        game.print_board_nums()

    letter = 'X'  # Premier joueur est X
    while game.empty_squares():
        if letter == 'O':
            square = random.choice(game.available_moves())  # IA aléatoire
        else:
            square = player_x(game)  # Joueur humain choisit une case

        if game.make_move(square, letter):
            if print_game:
                print(f'{letter} a joué sur la case {square}')
                game.print_board()
                print('')

            if game.current_winner:
                if print_game:
                    print(f'{letter} a gagné!')
                return letter  # Retourner le gagnant
            letter = 'O' if letter == 'X' else 'X'  # Switch joueur

    if print_game:
        print('C\'est un match nul!')


def human_move(game: TicTacToe):
    valid_square = False
    val = None
    while not valid_square:
        square = input('Entrez un choix de case (0-8) : ')
        try:
            val = int(square)
            if val not in game.available_moves():
                raise ValueError
            valid_square = True
        except ValueError:
            print("Case invalide, essayez à nouveau.")
    return val



if __name__ == "__main__":
    #main()
    t = TicTacToe()
    play(t, human_move, random.choice, print_game=True)
