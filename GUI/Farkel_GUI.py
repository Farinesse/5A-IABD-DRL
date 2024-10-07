import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageOps
from environment.FarkelEnv import FarkleEnv

class FarkleGUI:
    def __init__(self, master, players=2):
        self.master = master
        self.master.title("Farkle - Jeu de Dés")
        self.master.geometry("800x600")
        self.master.configure(bg='#E0E0E0')

        self.env = FarkleEnv(num_players=players)
        self.dice_images = {}
        self.selected_dice = []

        # Ajout de l'initialisation de action_received

        self.action_received = tk.BooleanVar(value=False)
        self.action = None

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
                                         command=lambda: self.take_action(False),
                                         state=tk.DISABLED)
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
            img = Image.open(f"Images/dice{i}.png")
            img = img.resize((60, 60))
            self.dice_images[i] = ImageTk.PhotoImage(img)

            # Créer une version "sélectionnée" de l'image avec un contour
            selected_img = ImageOps.expand(img, border=3, fill='red')
            self.dice_images[f"{i}_selected"] = ImageTk.PhotoImage(selected_img)

    def toggle_dice(self, index):
        if index < len(self.env.dice_roll):
            if index in self.selected_dice:
                self.selected_dice.remove(index)
            else:
                self.selected_dice.append(index)
            self.update_display()
            self.update_selection_info()

    def update_selection_info(self):
        if not self.selected_dice :
            self.info_var.set("Sélectionnez des dés qui rapportent des points")
            self.continue_button.config(state=tk.DISABLED)
            return

        action = self.get_action_from_selection()
        is_valid = self.env._validate_dice_selection(self.env.dice_roll, action[:len(self.env.dice_roll)])
        print(is_valid)

        if is_valid:
            selected_dice = [self.env.dice_roll[i] for i in self.selected_dice]
            potential_score = self.env._calculate_score(selected_dice)
            self.info_var.set(f"Sélection valide - Points potentiels : {potential_score}")
            self.continue_button.config(state=tk.NORMAL)

        else:
            self.info_var.set("Sélection invalide")
            self.continue_button.config(state=tk.DISABLED)

    def get_action_from_selection(self):
        action = [0] * 7
        for i in self.selected_dice:
            if i < len(self.env.dice_roll):
                action[i] = 1

        action[-1] = int(self.env.stop)
        if self.env.stop :
            action = [1] * 7
        return action

    def take_action(self, stop):
        self.env.stop = stop

        action = self.get_action_from_selection()


        if not stop and not self.selected_dice:
            messagebox.showinfo("Action invalide", "Vous devez sélectionner des dés valides avant de continuer.")
            return
        print(stop)
        observation, reward, done, _, info = self.env.step(action)

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

    def wait_for_action(self):
        self.action_received.set(False)
        self.continue_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        self.master.wait_variable(self.action_received)
        self.continue_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        return self.action


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

        self.update_selection_info()

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

def main_gui(players=2):
    root = tk.Tk()
    app = FarkleGUI(root,players)
    root.mainloop()

if __name__ == "__main__":
    main_gui()