import numpy as np
import tkinter as tk
from Qlearning import CheckForWin, GetLegalMoves, QLearnPlay

class TicTacToeUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Tic Tac Toe")
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.ai_player_id = 1
        self.human_player_id = 2
        self.human_score = 0
        self.ai_score = 0
        self.qPlay = QLearnPlay(self.ai_player_id,
                                'q_table.npy', 'q_table_keys.npy',
                                swap_player_ids=False)
        self.qPlay.set_epsilon(0.0)
        self.create_board()
        self.create_scoreboard()

    def create_board(self):
        for r in range(3):
            for c in range(3):
                button = tk.Button(self.master, text="", font=('normal', 40), width=5, height=2,
                                   command=lambda r=r, c=c: self.human_move(r, c))
                button.grid(row=r, column=c)
                self.buttons[r][c] = button

    def create_scoreboard(self):
        self.score_label = tk.Label(self.master, text="Human: 0   AI: 0", font=('normal', 20))
        self.score_label.grid(row=3, column=0, columnspan=3)

    def update_scoreboard(self):
        self.score_label.config(text=f"Human: {self.human_score}   AI: {self.ai_score}")

    def human_move(self, r, c):
        if self.board[r, c] == 0:
            self.board[r, c] = self.human_player_id
            self.buttons[r][c].config(text="O")
            if self.check_game_over():
                return
            self.master.after(500, self.ai_move)

    def ai_move(self):
        action, r, c = self.qPlay.get_move(self.board)
        if self.board[r, c] == 0:
            self.board[r, c] = self.ai_player_id
            self.buttons[r][c].config(text="X")
        else:
            print("AI player attempted illegal move")
        self.check_game_over()

    def check_game_over(self):
        win = CheckForWin(self.board)
        if win != 0:
            self.update_scores(win)
            self.display_winner(win)
            return True
        legal_moves_x, legal_moves_y = GetLegalMoves(self.board)
        if len(legal_moves_x) == 0:
            self.update_scores(0)
            self.display_winner(0)
            return True
        return False

    def update_scores(self, winner):
        if winner == self.human_player_id:
            self.human_score += 1
            self.ai_score -= 1
        elif winner == self.ai_player_id:
            self.ai_score += 1
            self.human_score -= 1
        # No score change for a draw
        self.update_scoreboard()

    def display_winner(self, winner):
        if winner == self.human_player_id:
            msg = "Human wins!"
        elif winner == self.ai_player_id:
            msg = "AI wins!"
        else:
            msg = "It's a draw!"
        result_window = tk.Toplevel(self.master)
        result_window.title("Game Over")
        tk.Label(result_window, text=msg, font=('normal', 20)).pack()
        tk.Button(result_window, text="Restart", command=lambda: self.restart_game(result_window)).pack()

    def restart_game(self, window):
        window.destroy()
        self.board = np.zeros((3, 3), dtype=np.int8)
        for r in range(3):
            for c in range(3):
                self.buttons[r][c].config(text="")

def main():
    root = tk.Tk()
    game = TicTacToeUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()