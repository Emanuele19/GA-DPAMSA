import os

import torch 
import config
import utils
from NSGA_II.nsga2 import NSGA2_GA

from DCNN_BDDQN.dqn import DQNAgent
from DCNN_BDDQN.env import Environment

class NSGA2_DCNN_GA(NSGA2_GA):
    """
    Questa classe eredita tutto il funzionamento Multiobiettivo (Torneo, Pareto, Crowding)
    da NSGA2_GA, ma sovrascrive il metodo mutation() per utilizzare il modello DCNN_BDDQN.
    """

    def mutation(self, model_path):
        """
        Override della mutazione per utilizzare DCNN_BDDQN.
        """
        # 1. Quanti individui mutare e quali
        num_individuals_to_mutate = max(
            1,
            round(len(self.population) * config.MUTATION_RATE)
        )

        if self.mode in {'sp', 'cs'}:
            sorted_population = sorted(
                enumerate(self.population_score),
                key=lambda x: x[1][1],
                reverse=True
            )
        else:
            sorted_population = sorted(
                enumerate(self.population_score),
                key=lambda x: (x[1][1], x[1][2]),
                reverse=True
            )

        positions_to_mutate = [
            pos for pos, _ in sorted_population[:num_individuals_to_mutate]
        ]

        # 2. Mutazione vera e propria con DCNN
        for index in positions_to_mutate:
            if index >= len(self.population):
                continue

            individual_to_mutate = self.population[index]

            # Trova la sub-board peggiore
            score, worst_fitted_range = utils.calculate_worst_fitted_sub_board(individual_to_mutate, self.mode)
            from_row, to_row, from_column, to_column = worst_fitted_range

            row_genes = individual_to_mutate[from_row:to_row]
            sub_board = []

            while len(row_genes) < config.AGENT_WINDOW_ROW:
                row_genes.append([5] * config.AGENT_WINDOW_COLUMN)

            for genes in row_genes:
                sub_genes = genes[from_column:to_column]
                while len(sub_genes) < config.AGENT_WINDOW_COLUMN:
                    sub_genes.append(5)
                sub_board.append(sub_genes)

            # --- START DCNN_BDDQN ---
            tensor_sub_board = torch.tensor(sub_board, dtype=torch.long)
            env = Environment(tensor_sub_board)
            
            # ✅ FIX PERFORMANCE: Salviamo l'agente in 'self' così sopravvive alle generazioni
            if getattr(self, 'agent', None) is None:
                self.agent = DQNAgent(n_sequences=env.N, vocab_size=6) 
                self.agent.load(model_path)
            
            state = env.reset()

            # Ciclo di risoluzione
            while True:
                # Usa self.agent invece di agent
                action = self.agent.predict(state)
                next_state, reward, done, is_stalling = env.step(action)
                state = next_state
                if done:
                    break

            aligned_sequences = env.get_alignment()
            # --- FINE INTEGRAZIONE DCNN_BDDQN ---

            # Reinserisci la board corretta
            for idx, sequence in enumerate(aligned_sequences):
                if idx < len(row_genes):
                    row_genes[idx][from_column:to_column] = sequence

            individual_to_mutate[from_row:to_row] = row_genes