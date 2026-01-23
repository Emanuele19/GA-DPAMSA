import torch
import numpy as np

# --- CLASSE ENVIRONMENT (COPIATA PER TEST) ---
class Environment:
    def __init__(self, input_tensor: torch.Tensor, fixed_size: int = 5, gap_idx: int = 4, pad_idx: int = 5):
        self.N, self.L = input_tensor.shape
        self.fixed_size = fixed_size
        self.gap_idx = gap_idx
        self.pad_idx = pad_idx
        
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
        self.history = []
        return self.current_state

    def step(self, action: int):
        if self.done: return self.current_state, 0.0, True

        original_column = self.current_state[:, 0].tolist()
        aligned_column = []
        next_state = torch.full_like(self.current_state, self.pad_idx)
        
        for i in range(self.N):
            is_gap = (action >> i) & 1
            if is_gap:
                aligned_column.append(self.gap_idx)
                next_state[i, :] = self.current_state[i, :] # Resta fermo
            else:
                aligned_column.append(original_column[i])
                next_state[i, 0 : self.fixed_size - 1] = self.current_state[i, 1:] # Slitta

        if not all(v == self.pad_idx for v in aligned_column):
            self.history.append(aligned_column)

        self.current_state = next_state
        reward = self._calc_sp_reward(aligned_column)
        
        if torch.all(self.current_state == self.pad_idx):
            self.done = True
            
        return self.current_state, reward, self.done

    def _calc_sp_reward(self, column: list) -> float:
        score = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                c1, c2 = column[i], column[j]
                if c1 == self.pad_idx and c2 == self.pad_idx: continue
                if c1 == self.gap_idx or c2 == self.gap_idx: score += -0.5
                elif c1 == c2: score += 1.0
                else: score += -0.3
        return score

    def get_alignment(self):
        if not self.history: return [[] for _ in range(self.N)]
        return np.array(self.history).T.tolist()

# --- TEST SUITE ---

def test_environment():
    # Setup: 2 sequenze di lunghezza 3, finestra fissa 5
    # Seq 0: [1, 2, 3] (A, C, G)
    # Seq 1: [1, 1, 1] (A, A, A)
    input_data = torch.tensor([
        [1, 2, 3],
        [1, 1, 1]
    ], dtype=torch.long)
    
    env = Environment(input_data, fixed_size=5, gap_idx=4, pad_idx=5)
    
    print("--- TEST 1: RESET & INITIAL STATE ---")
    state = env.reset()
    print(f"Initial State:\n{state}")
    # Aspettativa: [[1, 2, 3, 5, 5], [1, 1, 1, 5, 5]]
    assert state[0, 0] == 1 and state[1, 0] == 1
    assert state.shape == (2, 5)
    print("Reset OK.\n")

    print("--- TEST 2: STEP 1 - BOTH MOVE (Action 0: 00 in binary) ---")
    # Entrambe avanzano. Colonna consumata: [1, 1] -> Match!
    next_state, reward, done = env.step(0)
    print(f"Action: Both Move | Reward: {reward} | Done: {done}")
    print(f"Next State:\n{next_state}")
    # Aspettativa: Reward 1.0, S0: [2, 3, 5, 5, 5], S1: [1, 1, 5, 5, 5]
    assert reward == 1.0
    assert next_state[0, 0] == 2 and next_state[1, 0] == 1
    print("Step 1 OK.\n")

    print("--- TEST 3: STEP 2 - SEQ 0 GAPS, SEQ 1 MOVES (Action 1: 01 in binary) ---")
    # Seq 0 mette GAP (resta su '2'), Seq 1 avanza (consuma '1', va sul prossimo '1')
    # Colonna consumata: [Gap, 1]
    next_state, reward, done = env.step(1)
    print(f"Action: S0 Gap, S1 Move | Reward: {reward} | Done: {done}")
    print(f"Next State:\n{next_state}")
    # Aspettativa: Reward -0.5, S0: [2, 3, 5, 5, 5], S1: [1, 5, 5, 5, 5]
    assert reward == -0.5
    assert next_state[0, 0] == 2 # S0 non si è mossa
    assert next_state[1, 0] == 1 # S1 si è mossa
    print("Step 2 OK.\n")

    print("--- TEST 4: HISTORY & ALIGNMENT ---")
    alignment = env.get_alignment()
    print(f"History (Vertical Columns): {env.history}")
    print(f"Final Alignment (Rows): {alignment}")
    # Aspettativa: Col 1: [1,1], Col 2: [Gap, 1] -> Seq0: [1, 4], Seq1: [1, 1]
    assert alignment[0] == [1, 4]
    assert alignment[1] == [1, 1]
    print("History OK.\n")

    print("--- TEST 5: REPEAT UNTIL DONE ---")
    # S0: [2, 3], S1: [1]
    env.step(0) # Entrambe avanzano -> Col [2, 1], Reward -0.3
    env.step(0) # Entrambe avanzano -> Col [3, 5], Reward -0.5 (S1 è già pad)
    _, _, done = env.step(0) # Ultimo shift per svuotare
    print(f"Final State (All padding?):\n{env.current_state}")
    print(f"Done: {done}")
    assert done == True
    print("Termination OK.\n")

    print("--- TEST 6: RESET AFTER DONE ---")
    env.reset()
    assert len(env.history) == 0
    assert env.current_state[0, 0] == 1
    print("Reset after done OK.\n")

if __name__ == "__main__":
    test_environment()
    print("TUTTI I TEST SUPERATI!")