import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps

from GUI.test import load_model_pkl, action_agent
from environment.FarkelEnv import FarkleEnv


class FarkleGUI:
    def __init__(self, master, players=2, agent=False, path_model=None, player_types=None):
        self.master = master
        self.master.title("Farkle - Jeu de Dés")
        self.master.geometry("800x600")
        self.master.configure(bg='#E0E0E0')

        self.agent = agent
        self.player_types = player_types or ["human", "random"]
        self.env = FarkleEnv(num_players=players)
        self.model = load_model_pkl(path_model)
        if self.model is None:
            print("Warning: Failed to load model, agent will use random actions")
        self.dice_images = {}
        self.selected_dice = []

        self.create_widgets()
        self.load_dice_images()
        self.update_display()

        # Démarrer automatiquement le tour si c'est un joueur non-humain
        self.master.after(1000, self.check_automatic_turn)

    def create_widgets(self):
        # Top Frame
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

        # Dice Frame
        self.dice_frame = tk.Frame(self.master, bg='#E0E0E0')
        self.dice_frame.pack(pady=20)

        self.dice_labels = []
        for i in range(6):
            label = tk.Label(self.dice_frame, bg='#E0E0E0')
            label.grid(row=0, column=i, padx=5)
            label.bind('<Button-1>', lambda e, i=i: self.toggle_dice(i))
            self.dice_labels.append(label)

        # Button Frame
        self.button_frame = tk.Frame(self.master, bg='#E0E0E0')
        self.button_frame.pack(pady=20)

        self.continue_button = tk.Button(self.button_frame, text="Continuer",
                                         command=lambda: self.take_action(stop=False))
        self.continue_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(self.button_frame, text="Arrêter le tour",
                                     command=lambda: self.take_action(stop=True))
        self.stop_button.pack(side=tk.LEFT, padx=10)

        # Scores Frame
        self.scores_frame = tk.Frame(self.master, bg='#E0E0E0')
        self.scores_frame.pack(pady=10)

        self.score_vars = []
        for i in range(self.env.num_players):
            var = tk.StringVar()
            label = tk.Label(self.scores_frame, textvariable=var,
                             font=('Helvetica', 14), bg='#E0E0E0')
            label.pack()
            self.score_vars.append(var)

        self.progress_bars = []
        for i in range(self.env.num_players):
            progress_label = tk.Label(self.scores_frame, text=f"Progression Joueur {i + 1} :", bg='#E0E0E0')
            progress_label.pack()
            progress_bar = ttk.Progressbar(self.scores_frame, orient="horizontal", length=300, mode="determinate",
                                           maximum=100)
            progress_bar.pack(pady=5)
            self.progress_bars.append(progress_bar)

    def load_dice_images(self):
        for i in range(1, 7):
            img = Image.open(f"C:/Users/farin/PycharmProjects/5A-IABD-DRL/Images/dice{i}.png")
            img = img.resize((60, 60))
            self.dice_images[i] = ImageTk.PhotoImage(img)
            selected_img = ImageOps.expand(img, border=3, fill='red')
            self.dice_images[f"{i}_selected"] = ImageTk.PhotoImage(selected_img)

    def check_automatic_turn(self):
        """Vérifie et exécute les tours automatiques pour les joueurs non-humains."""
        player_type = self.player_types[self.env.current_player]
        if player_type == "random":
            self.play_random()
        elif player_type == "agent":
            self.play_agent()

    def take_action(self, stop):
        """Gère l'action du joueur."""
        self.env.stop = stop
        player_type = self.player_types[self.env.current_player]

        if player_type == "human":
            if stop:
                action = self.get_stop_action(self.env.get_valid_actions())
            else:
                if not self.selected_dice:
                    messagebox.showinfo("Action invalide",
                                        "Vous devez sélectionner des dés valides avant de continuer.")
                    return
                action = self.get_action_from_selection()

            observation, reward, done, _, info = self.env.step(action)
            self.handle_action_feedback(info, reward, done)
            if not done:
                self.master.after(1000, self.check_automatic_turn)

    def play_random(self):
        """Exécute le tour du joueur aléatoire."""
        action = self.env.get_random_action()
        observation, reward, done, _, info = self.env.step(action)

        self.update_display()
        self.master.update()

        if done:
            self.game_over()
        elif info.get("farkle", False) or info.get("stopped", False):
            self.master.after(1000, self.check_automatic_turn)
        else:
            self.master.after(1000, self.play_random)

    def play_agent(self):
        """Exécute le tour de l'agent."""
        action = action_agent(self.env, self.model)
        observation, reward, done, _, info = self.env.step(action)

        self.update_display()
        self.master.update()

        if done:
            self.game_over()
        elif info.get("farkle", False) or info.get("stopped", False):
            self.master.after(1000, self.check_automatic_turn)
        else:
            self.master.after(1000, self.play_agent)

    def toggle_dice(self, index):
        if index < len(self.env.dice_roll):
            if index in self.selected_dice:
                self.selected_dice.remove(index)
            else:
                self.selected_dice.append(index)
            self.update_display()
            self.update_selection_info()

    def update_selection_info(self):
        if not self.env._validate_dice_selection(self.env.dice_roll, self.get_action_from_selection()):
            self.info_var.set("Sélectionnez des dés qui rapportent des points")
            self.continue_button.config(state=tk.DISABLED)
        else:
            selected_dice_values = [self.env.dice_roll[i] for i in self.selected_dice]
            potential_score = self.env._calculate_score(selected_dice_values)
            self.info_var.set(f"Score potentiel : {potential_score}")
            self.continue_button.config(state=tk.NORMAL)

    def get_stop_action(self, valid_actions):
        for i, va in enumerate(valid_actions[::-1]):
            if va == 1:
                return [int(b) for b in format(127 - i, '07b')]
        return [0] * 7

    def handle_action_feedback(self, info, reward, done):
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
        self.update_progress_bars()

        self.update_selection_info()

    def update_progress_bars(self):
        for i, progress_bar in enumerate(self.progress_bars):
            progress = (self.env.scores[i] / self.env.target_score) * 100
            progress_bar['value'] = progress

    def get_action_from_selection(self):
        action = [0] * 7
        for i in self.selected_dice:
            if i < len(self.env.dice_roll):
                action[i] = 1
        action[-1] = int(self.env.stop)
        return action

    def game_over(self):
        winner = self.env.current_player + 1
        messagebox.showinfo("Fin du jeu",
                            f"Joueur {winner} gagne avec {self.env.scores[self.env.current_player]} points!")
        if messagebox.askyesno("Nouvelle partie", "Voulez-vous commencer une nouvelle partie?"):
            self.env.reset()
            self.selected_dice = []
            self.update_display()
            self.master.after(1000, self.check_automatic_turn)
        else:
            self.master.quit()


def main_gui(player1_type="random", player2_type="random", path_model=None):
    root = tk.Tk()
    app = FarkleGUI(
        root,
        players=2,
        agent=(player1_type == "agent" or player2_type == "agent"),
        path_model=path_model,
        player_types=[player1_type, player2_type]
    )
    root.mainloop()

