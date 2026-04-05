import torch
import numpy as np
from typing import Tuple, List
import config

from dataset_module.encoding import SequenceDecoder

class Environment:
    def __init__(self, input_tensor: torch.Tensor | np.ndarray, 
                 fixed_size: int = 30, 
                 gap: str = config.GAP_CHARACTER,
                 pad: str = config.PAD_CHARACTER):
        
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor)
        
        self.N, self.L = input_tensor.shape
        self.fixed_size = fixed_size
        self.gap_idx = config.NUCLEOTIDE_ENCODING[gap]
        self.pad_idx = config.NUCLEOTIDE_ENCODING[pad]

        # Il reward si accumula per ogni episodio in due modi:
        # 1. Come somma di reward a coppie di nucleotidi nella colonna corrente
        # 2. Come somma dei reward calcolati nel punto 1. per tutte le colonne
        # Qui sto scalando i reward per colonna perché lo step è fatto una
        # colonna alla volta. Lo scaling porta tutti i reward tra [-1 e 1]
        # Va fratto al reward nella funzione di calcolo dei reward
        self.num_pairs = self.N * (self.N - 1) // 2
        self.reward_scaling_factor = 1.0 / (self.num_pairs * config.MATCH_REWARD)
        self.gap_column_action_penalty = -1.0 * config.MATCH_REWARD * self.num_pairs * config.PENALTY_MULTIPLIER
        
        # Padding iniziale se necessario
        if self.L < self.fixed_size:
            padding = torch.full((self.N, self.fixed_size - self.L), self.pad_idx, dtype=torch.long)
            self.initial_tensor = torch.cat([input_tensor, padding], dim=1)
        else:
            self.initial_tensor = input_tensor[:, :self.fixed_size]

        self.current_state = None
        self.done = False
        self.history = [] 

    def reset(self) -> torch.Tensor:
        self.current_state = self.initial_tensor.clone()
        self.done = False
        self.history = [] # Reset della memoria ad ogni nuovo episodio
        return self.current_state

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        if self.done:
            return self.current_state, 0.0, True, False

        # 1. Identifichiamo i valori della colonna 0
        original_column = self.current_state[:, 0].tolist()
        
        # 2. Costruiamo la colonna effettiva basandoci sull'azione
        # Questa è la colonna che farà parte dell'allineamento finale
        aligned_column = []
        next_state = torch.full_like(self.current_state, self.pad_idx)
        
        for i in range(self.N):
            is_gap = bool(action[i] == 1)
            if is_gap:
                aligned_column.append(self.gap_idx)
                next_state[i, :] = self.current_state[i, :]
            else:
                aligned_column.append(original_column[i])
                next_state[i, 0 : self.fixed_size - 1] = self.current_state[i, 1:]

        # 3. SALVATAGGIO NELLA STORIA
        # Registriamo la colonna solo se non è composta interamente da padding tecnico
        if not all(v == self.pad_idx for v in aligned_column):
            self.history.append(aligned_column)

        self.current_state = next_state

        is_stalling_action = all(c == 1 for c in action)
        if is_stalling_action:
            reward = self.gap_column_action_penalty * self.reward_scaling_factor
        else:
            reward = self.calc_reward(aligned_column)
        
        if torch.all(self.current_state == self.pad_idx):
            self.done = True

        # if not is_inference:
        #     self.done = self.done or all_gaps_column
            
        return self.current_state, reward, self.done, is_stalling_action

    def get_alignment(self) -> List[List[int]]:
        """
        Restituisce l'allineamento prodotto finora come lista di sequenze (righe).
        Converte la storia da [colonne] a [righe].
        """
        if not self.history:
            return [[] for _ in range(self.N)]
        
        # Trasponiamo la lista di liste (da colonne a righe)
        alignment = np.array(self.history).T.tolist()
        return alignment

    def get_alignment_as_strings(self, decoder: SequenceDecoder = SequenceDecoder(config.NUCLEOTIDE_DECODING)) -> List[str]:
        """
        Restituisce l'allineamento in formato testuale (es. ['AT-G', 'ATCG']).
        """            
        raw_alignment = self.get_alignment()
        return decoder.decode_batch(raw_alignment)

    def _calc_sp_reward(self, column: list) -> float:
        # Nota: usiamo direttamente aligned_column che contiene già i gap_idx decisi dall'azione
        score = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                c1, c2 = column[i], column[j]
                if c1 == self.pad_idx or c2 == self.pad_idx: continue
                if c1 == self.gap_idx or c2 == self.gap_idx:
                    score += config.GAP_PENALTY
                elif c1 == c2:
                    score += config.MATCH_REWARD
                else:
                    score += config.MISMATCH_PENALTY
        return score
    
    def calc_reward(self, column: list) -> float:
        # Simile ad SP ma fatto solo sui nucleotidi
        # Poi viene aggiunta una penalità di gap per ogni gap aggiunto
        score = 0.0
        gap_count = column.count(self.gap_idx)
        # if gap_count == self.N: # -1.5x the best reward the agent could get
        #     return -1.0 * config.MATCH_REWARD * self.num_pairs * config.PENALTY_MULTIPLIER
        
        score += gap_count * config.GAP_PENALTY
        for i in range(self.N):
            for j in range(i + 1, self.N):
                c1, c2 = column[i], column[j]
                if c1 == self.pad_idx or c2 == self.pad_idx: continue
                if c1 == self.gap_idx or c2 == self.gap_idx: continue
                elif c1 == c2:
                    score += config.MATCH_REWARD
                else:
                    score += config.MISMATCH_PENALTY
        return score * self.reward_scaling_factor
    
    def calculate_total_cs(self) -> float:
        """
        Calcola il Column Score totale dell'allineamento prodotto.
        Ritorna la somma delle colonne perfettamente conservate.
        """
        if not self.history:
            return 0.0

        perfect_columns = 0
        
        for column in self.history:
            # 1. Ignoriamo colonne che contengono padding (fine sequenza)
            if self.pad_idx in column:
                continue
                
            # 2. Ignoriamo colonne che contengono gap (non sono conservate per definizione)
            if self.gap_idx in column:
                continue
                
            # 3. Verifichiamo se tutti gli elementi sono uguali tra loro
            # (Se il set ha lunghezza 1, tutti gli elementi sono identici)
            if len(set(column)) == 1:
                perfect_columns += 1
                
        return float(perfect_columns) / len(self.history)
    

    def get_cs_percentage(self) -> float:
        """Ritorna la percentuale di colonne conservate rispetto alla lunghezza totale."""
        total_cs = self.calculate_total_cs()
        if not self.history: return 0.0
        return (total_cs / len(self.history)) * 100
    
    def calc_sp_score(self) -> float:
        """
        Calcola il Sum of Pairs (SP) score totale per l'intero allineamento.
        """
        total_score = 0.0
        for column in self.history:
            total_score += self._calc_sp_reward(column)
        return total_score
    
    def calc_exact_matched(self) -> int:
        """
        Calcola il numero di colonne esattamente allineate (exact matches).
        Una colonna è un "exact match" se tutti i nucleotidi sono identici
        e non contiene né gap né padding.
        """
        if not self.history:
            return 0

        exact_matches = 0
        for column in self.history:
            # Una colonna con gap o padding non può essere un exact match.
            if self.gap_idx in column or self.pad_idx in column:
                continue
            # Se la colonna ha un solo tipo di nucleotide, è un exact match.
            if len(set(column)) == 1:
                exact_matches += 1
        return exact_matches

    def calculate_metrics(self) -> dict:
        alignment_length = len(self.history)
        num_sequences = self.N
        cs_score = self.calculate_total_cs()
        sp_score = self.calc_sp_score()
        
        return {
            "AL": alignment_length,
            "QTY": num_sequences,
            "SP": sp_score,
            "CS": cs_score,
            "EM": self.calc_exact_matched()
        }