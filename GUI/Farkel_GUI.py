import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import random
from gymnasium import spaces


class FarkleEnv:
    def __init__(self, num_players=2, target_score=10000):
        self.num_players = num_players
        self.target_score = target_score
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0] + [0] * num_players + [0] * 6 + [0]),
            high=np.array([num_players - 1, target_score, 6] + [target_score] * num_players + [6] * 6 + [1]),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(128)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.scores = [0] * self.num_players
        self.current_player = 0
        self.round_score = 0
        self.remaining_dice = 6
        self.dice_roll = self.roll_dice(self.remaining_dice)
        self.game_over = False
        self.last_action_stop = False
        return self.get_observation(), {}

    def get_observation(self):
        padded_dice = self.dice_roll + [0] * (6 - len(self.dice_roll))
        return np.array([self.current_player, self.round_score, self.remaining_dice] +
                        self.scores + padded_dice + [int(self.last_action_stop)])

    def roll_dice(self, num_dice):
        return [random.randint(1, 6) for _ in range(num_dice)]

    def get_valid_actions(self):
        valid_mask = np.zeros(128, dtype=np.int8)
        for action in range(128):
            binary = format(action, '07b')
            action_list = [int(b) for b in binary]
            if self._validate_dice_selection(self.dice_roll, action_list[:len(self.dice_roll)]):
                valid_mask[action] = 1
        return valid_mask

    def _validate_dice_selection(self, dice_roll, action):
        if len(action) < len(dice_roll):
            return False

        selected_dice = [d for i, d in enumerate(dice_roll) if action[i] == 1]
        if not selected_dice:
            return True

        if sorted(selected_dice) == [1, 2, 3, 4, 5, 6]:
            return True

        selected_counts = [selected_dice.count(i) for i in range(1, 7)]
        if len(selected_dice) == 6 and selected_counts.count(2) == 3:
            return True

        valid_singles = {1, 5}
        for value, count in enumerate(selected_counts, start=1):
            if count > 0:
                original_count = dice_roll.count(value)
                if value not in valid_singles and count < 3 and original_count < 3:
                    return False

        return True

    def _calculate_score(self, dice_roll):
        if not dice_roll:
            return 0

        counts = [dice_roll.count(i) for i in range(1, 7)]
        score = 0

        if sorted(dice_roll) == [1, 2, 3, 4, 5, 6]:
            return 1500
        if len(dice_roll) == 6 and counts.count(2) == 3:
            return 1000

        for value, count in enumerate(counts, start=1):
            if count >= 3:
                if value == 1:
                    score += 1000 * (2 ** (count - 3))
                else:
                    score += value * 100 * (2 ** (count - 3))

        ones_count = counts[0] % 3
        fives_count = counts[4] % 3
        score += ones_count * 100 + fives_count * 50

        return score

    def step(self, action):
        binary = format(action, '07b')
        action_list = [int(b) for b in binary]

        if not self._validate_dice_selection(self.dice_roll, action_list[:len(self.dice_roll)]):
            return self.get_observation(), -100, True, False, {"invalid_action": True}

        kept_dice = [self.dice_roll[i] for i in range(len(self.dice_roll)) if action_list[i] == 1]
        new_score = self._calculate_score(kept_dice)

        if new_score == 0:
            # Farkle - le joueur perd tous les points du tour
            lost_points = self.round_score
            self.round_score = 0
            self.next_player()
            return self.get_observation(), -lost_points, False, False, {"farkle": True, "lost_points": lost_points}

        self.round_score += new_score
        self.remaining_dice -= sum(action_list[:len(self.dice_roll)])

        if self.remaining_dice == 0:
            self.remaining_dice = 6

        self.last_action_stop = bool(action_list[-1])
        if self.last_action_stop:
            # Le joueur arrête son tour, on ajoute les points au score total
            self.scores[self.current_player] += self.round_score
            reward = self.round_score
            if self.scores[self.current_player] >= self.target_score:
                self.game_over = True
                return self.get_observation(), reward, True, False, {"win": True}
            self.next_player()
            return self.get_observation(), reward, False, False, {"stopped": True}

        self.dice_roll = self.roll_dice(self.remaining_dice)
        return self.get_observation(), new_score, False, False, {}

    def next_player(self):
        self.current_player = (self.current_player + 1) % self.num_players
        self.round_score = 0
        self.remaining_dice = 6
        self.dice_roll = self.roll_dice(self.remaining_dice)
        self.last_action_stop = False


class FarkleGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Farkle - Jeu de Dés")
        self.master.geometry("800x600")
        self.master.configure(bg='#E0E0E0')

        self.env = FarkleEnv(num_players=2)
        self.dice_images = {}
        self.selected_dice = []

        self.create_widgets()
        self.load_dice_images()
        self.update_display()

    def create_widgets(self):
        self.top_frame = tk.Frame(self.master, bg='#E0E0E0')
        self.top_frame.pack(pady=10)

        self.player_var = tk.StringVar()
        self.player_label = tk.Label(self.top_frame, textvariable=self.player_var,
                                     font=('Helvetica', 20), bg='#E0E0E0')
        self.player_label.pack()

        self.round_score_var = tk.StringVar()
        self.round_score_label = tk.Label(self.top_frame, textvariable=self.round_score_var,
                                          font=('Helvetica', 16), bg='#E0E0E0')
        self.round_score_label.pack()

        self.info_var = tk.StringVar()
        self.info_label = tk.Label(self.top_frame, textvariable=self.info_var,
                                   font=('Helvetica', 12), bg='#E0E0E0', fg='#666666')
        self.info_label.pack()

        self.dice_frame = tk.Frame(self.master, bg='#E0E0E0')
        self.dice_frame.pack(pady=20)

        self.dice_labels = []
        for i in range(6):
            label = tk.Label(self.dice_frame, bg='#E0E0E0')
            label.grid(row=0, column=i, padx=5)
            label.bind('<Button-1>', lambda e, i=i: self.toggle_dice(i))
            self.dice_labels.append(label)

        self.button_frame = tk.Frame(self.master, bg='#E0E0E0')
        self.button_frame.pack(pady=20)

        self.continue_button = tk.Button(self.button_frame, text="Continuer",
                                         command=lambda: self.take_action(False))
        self.continue_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(self.button_frame, text="Arrêter le tour",
                                     command=lambda: self.take_action(True))
        self.stop_button.pack(side=tk.LEFT, padx=10)

        self.scores_frame = tk.Frame(self.master, bg='#E0E0E0')
        self.scores_frame.pack(pady=10)

        self.score_vars = []
        for i in range(self.env.num_players):
            var = tk.StringVar()
            label = tk.Label(self.scores_frame, textvariable=var,
                             font=('Helvetica', 14), bg='#E0E0E0')
            label.pack()
            self.score_vars.append(var)

    def load_dice_images(self):
        for i in range(1, 7):
            img = Image.open(f"C:\\Users\\farin\\PycharmProjects\\5A-IABD-DRL\\Images\\dice{i}.png")
            img = img.resize((60, 60))
            self.dice_images[i] = ImageTk.PhotoImage(img)
            self.dice_images[f"{i}_selected"] = ImageTk.PhotoImage(img)

    def toggle_dice(self, index):
        if index < len(self.env.dice_roll):
            if index in self.selected_dice:
                self.selected_dice.remove(index)
            else:
                self.selected_dice.append(index)
            self.update_display()
            self.update_selection_info()

    def update_selection_info(self):
        if not self.selected_dice:
            self.info_var.set("Sélectionnez des dés qui rapportent des points")
            return

        action = self.get_action_from_selection(False)
        is_valid = self.env._validate_dice_selection(self.env.dice_roll, action[:len(self.env.dice_roll)])

        if is_valid:
            selected_dice = [self.env.dice_roll[i] for i in self.selected_dice]
            potential_score = self.env._calculate_score(selected_dice)
            self.info_var.set(f"Sélection valide - Points potentiels : {potential_score}")
        else:
            self.info_var.set("Sélection invalide")

    def get_action_from_selection(self, stop=False):
        action = [0] * 7
        for i in self.selected_dice:
            if i < len(self.env.dice_roll):
                action[i] = 1
        action[-1] = int(stop)
        return action

    def take_action(self, stop):
        action = self.get_action_from_selection(stop)
        action_int = int(''.join(map(str, action)), 2)

        observation, reward, done, _, info = self.env.step(action_int)

        if info.get("invalid_action", False):
            messagebox.showinfo("Action invalide", "Sélection de dés non valide")
        elif info.get("farkle", False):
            lost_points = info.get("lost_points", 0)
            messagebox.showinfo("Farkle", f"Pas de points! Vous perdez {lost_points} points. Tour terminé.")
        elif info.get("stopped", False):
            messagebox.showinfo("Tour terminé", f"Points marqués : {reward}")

        self.selected_dice = []
        self.update_display()

        if done:
            self.game_over()

    def update_display(self):
        self.player_var.set(f"Joueur {self.env.current_player + 1}")
        self.round_score_var.set(f"Score du tour: {self.env.round_score}")

        for i, label in enumerate(self.dice_labels):
            if i < len(self.env.dice_roll):
                dice_value = self.env.dice_roll[i]
                if i in self.selected_dice:
                    label.config(image=self.dice_images[f"{dice_value}_selected"])
                else:
                    label.config(image=self.dice_images[dice_value])
            else:
                label.config(image='')

        for i, var in enumerate(self.score_vars):
            var.set(f"Joueur {i + 1}: {self.env.scores[i]}")

    def game_over(self):
        winner = self.env.current_player + 1
        messagebox.showinfo("Fin du jeu",
                            f"Joueur {winner} gagne avec {self.env.scores[self.env.current_player]} points!")
        if messagebox.askyesno("Nouvelle partie", "Voulez-vous commencer une nouvelle partie?"):
            self.env.reset()
            self.selected_dice = []
            self.update_display()
        else:
            self.master.quit()


def main():
    root = tk.Tk()
    app = FarkleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()