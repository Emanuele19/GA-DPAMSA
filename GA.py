import copy
import random

import config
from DPAMSA.dqn import DQN
from DPAMSA.env import Environment
import utils

"""
Genetic Algorithm for DPAMSA Refinement (GA-DPAMSA)
---------------------------------------------------------
This module implements a Genetic Algorithm (GA) used to refine multiple sequence 
alignments (MSA) as part of the GA-DPAMSA framework.
The GA evolves a population of candidate alignments through iterative processes including mutation, selection, 
and horizontal crossover. Evaluation of candidates is based on alignment quality 
metrics such as Sum-of-Pairs (SP) score and Column Score (CS), or a combination of both 
in Multi-Objective (MO) mode. A Hall of Fame is maintained to preserve the best alignment 
found across all generations.

Key Features:
    - Population Initialization: Generates an initial set of candidate alignments using 
      exact copies and modified copies (with random gap insertions).
    - Fitness Evaluation: Computes fitness scores using SP and/or CS metrics after cleaning 
      unnecessary gap columns.
    - Mutation: Applies an RL-based mutation to the worst-performing sub-region of selected 
      individuals.
    - Selection: Chooses top-performing individuals based on fitness scores (or Pareto front in MO mode).
    - Horizontal Crossover: Combines portions (rows) from two parent alignments to create new offspring.
    - Hall of Fame: Stores the best alignment encountered during the evolution process.

Author: https://github.com/FLaTNNBio/GA-DPAMSA
"""

# Map nucleotide characters to numerical values (gaps are represented by 5)
nucleotides_map = {'A': 1, 'T': 2, 'C': 3, 'G': 4, 'a': 1, 't': 2, 'c': 3, 'g': 4, '-': 5,'N':6, 'n':6}


class GA:
    def __init__(self, sequences, mode):
        """
        Initializes the Genetic Algorithm with the given sequences and evaluation mode.

        Parameters:
        -----------
            sequences (list of str): Input nucleotide sequences to be aligned.
            mode (str): Evaluation mode. Options:
                        'sp'  -> Sum-of-Pairs,
                        'cs'  -> Column Score,
                        'mo'  -> Multi-Objective (combines SP and CS).
        """
        self.sequences = sequences
        self.mode = mode
        self.population_size = config.POPULATION_SIZE  # Number of individuals in the population
        self.population = []  # List of candidate alignments (each alignment is a list of sequences)
        self.population_score = []  # Fitness scores corresponding to each candidate alignment
        self.hall_of_fame = []  # Stores the best candidate alignment and its score
        self.current_iteration = 0  # Current generation count

    def generate_population(self):
        """
        Generates the initial population of candidate alignments.

        The population is built by:
            - Creating a fraction of exact copies (based on CLONE_RATE) from the input sequences.
            - Creating modified versions by inserting random gaps (based on GAP_RATE) into sequences.

        Returns:
        --------
            None: The resulting population is stored in self.population.
        """
        self.population = []  # Reset population

        # Determine the number of exact copies and modified individuals
        num_exact_copies = round(self.population_size * config.CLONE_RATE)
        num_modified = self.population_size - num_exact_copies

        # Add exact copies of dataset sequences
        for _ in range(num_exact_copies):
            self.population.append([[nucleotides_map[nuc] for nuc in seq] for seq in self.sequences])

        # Add modified sequences with random gaps
        for _ in range(num_modified):
            modified_individual = []
            for sequence in self.sequences:
                modified_seq = [nucleotides_map[nuc] for nuc in sequence]  # Convert to numerical form
                # Determine number of gaps to insert (at least one gap)
                num_gaps = max(1, round(len(modified_seq) * config.GAP_RATE))
                # Insert gaps at random positions
                for _ in range(num_gaps):
                    gap_pos = random.randint(0, len(modified_seq) - 1)
                    modified_seq.insert(gap_pos, 5)  # Insert gap (5)
                modified_individual.append(modified_seq)

            self.population.append(modified_individual)

    def update_hall_of_fame(self):
        """
        Updates the Hall of Fame (HoF) with the best candidate alignment found so far.

        Process:
            - If the HoF is empty, initialize it with the best individual from the current population.
            - Otherwise, combine the current population with the HoF individual, re-evaluate their fitness,
              and update the HoF with the absolute best candidate.

        Returns:
        --------
            None
        """
        # If Hall of Fame is empty, set it using the best current individual.
        if not self.hall_of_fame:
            best_idx = utils.get_index_of_the_best_fitted_individuals(self.population_score, num_individuals=1)[0]
            best_individual = copy.deepcopy(self.population[best_idx])
            best_score = copy.deepcopy(self.population_score[best_idx])
            self.hall_of_fame = (best_individual, best_score)
            return

        # Create deep copies of the current population and their scores.
        temp_population = copy.deepcopy(self.population)
        temp_scores = copy.deepcopy(self.population_score)

        # Retrieve the current HoF individual and add it to the temporary population.
        hof_individual, hof_score = copy.deepcopy(self.hall_of_fame)
        temp_population.append(hof_individual)

        # The following code avoids HoF update issues
        # Appending just hof_score leads to wrong HoF update due to the saved index in hof_score
        if len(temp_scores[0]) == 2:
            # If only one metric is used (SP or CS)
            temp_scores.append((len(temp_scores), hof_score[1]))
        else:
            # Multi-Objective mode
            temp_scores.append((len(temp_scores), hof_score[1], hof_score[2]))

        # Identify the absolute best candidate among both the current population and HoF individual.
        best_idx = utils.get_index_of_the_best_fitted_individuals(temp_scores, num_individuals=1)[0]
        self.hall_of_fame = (copy.deepcopy(temp_population[best_idx]), copy.deepcopy(temp_scores[best_idx]))

    def calculate_fitness_score(self):
        """
        Evaluates and assigns fitness scores to each candidate alignment in the population.

        For each candidate alignment:
            1. Determine the maximum sequence length and pad all sequences with gaps if necessary.
            2. Remove unnecessary gap columns.
            3. Compute fitness metrics:
                - SP score (if mode is 'sp' or 'mo')
                - CS score (if mode is 'cs' or 'mo')
            4. Store the fitness score and update the Hall of Fame.

        Returns:
        --------
            None
        """
        self.population_score = []  # Reset fitness scores

        for index_chromosome, chromosome in enumerate(self.population):
            # Get the maximum length of the sequences in the candidate alignment.
            max_length = max(map(len, chromosome))

            # Pad each sequence to ensure all have equal length (using gap value 5).
            for gene in chromosome:
                gene.extend([5] * (max_length - len(gene)))  # Extend with gaps instead of a loop

            # Clean columns that contain only gaps.
            utils.clean_unnecessary_gaps(chromosome)
            max_length = max(map(len, chromosome))  # Recompute length after cleaning

            # Compute fitness scores based on the current mode.
            sp_score = utils.get_sum_of_pairs(chromosome, 0, len(chromosome), 0, max_length) \
                if self.mode in {'sp', 'mo'} else None
            cs_score = utils.get_column_score(chromosome, 0, len(chromosome), 0, max_length) \
                if self.mode in {'cs', 'mo'} else None

            # Store scores as a tuple: (chromosome index, score) for single mode, or (index, sp, cs) for MO.
            if self.mode == 'mo':
                self.population_score.append((index_chromosome, sp_score, cs_score))
            else:
                self.population_score.append((index_chromosome, sp_score or cs_score))

        # Update the Hall of Fame after computing the fitness scores.
        self.update_hall_of_fame()

    def _find_pareto_front(self):
        """
        Identifies the Pareto Front of candidate alignments in Multi-Objective mode.

        A candidate is in the Pareto Front if no other candidate dominates it, where domination means:
            - Another candidate has both an equal or higher SP score and an equal or higher CS score,
              with at least one being strictly higher.

        Returns:
        --------
            list of tuples: Each tuple contains (individual index, SP score, CS score) for Pareto optimal candidates.
        """
        pareto_front = []

        # Compare each candidate alignment against every other candidate.
        for i, (idx1, sp1, cs1) in enumerate(self.population_score):
            dominated = False

            for j, (idx2, sp2, cs2) in enumerate(self.population_score):
                if i != j:
                    # Check if candidate j dominates candidate i.
                    if (sp2 >= sp1 and cs2 >= cs1) and (sp2 > sp1 or cs2 > cs1):
                        dominated = True
                        break  # If it is the case then we don't need it

            if not dominated:
                pareto_front.append((idx1, sp1, cs1))  # Add to the Pareto Front

        return pareto_front

    def selection(self):
        """
        Selects the top-performing candidate alignments to form the next generation.

        For single-objective modes ('sp' or 'cs'):
            - Sort candidates by fitness score in descending order and choose the top fraction (SELECTION_RATE).

        For Multi-Objective mode ('mo'):
            - Identify the Pareto Front.
            - If the Pareto Front has fewer candidates than required, supplement with top candidates
              sorted by SP or CS scores.
            - Select the top candidates based on these criteria.

        After selection, the population is updated with the selected candidates, and fitness scores are recalculated.

        Returns:
        --------
            None
        """
        top_n = int(self.population_size * config.SELECTION_RATE)

        if self.mode in {'sp', 'cs'}:
            # Sort candidates by single metric (SP or CS) in descending order.
            sorted_population = sorted(self.population_score, key=lambda x: x[1],
                                       reverse=True)  # Ordinamento decrescente
            top_indices = {ind for ind, score in sorted_population[:top_n]}

        elif self.mode == 'mo':
            # Identify the Pareto optimal candidates.
            pareto_front = self._find_pareto_front()
            pareto_indices = {ind for ind, _, _ in pareto_front}

            # If the Pareto Front does not yield enough candidates, supplement with top candidates.
            if len(pareto_indices) < top_n:
                sorted_sop = sorted(self.population_score, key=lambda x: x[1], reverse=True)  # sort by SP
                sorted_column = sorted(self.population_score, key=lambda x: x[2], reverse=True)  # sort by CS
                top_indices_sop = {ind for ind, _, _ in sorted_sop[:top_n]}
                top_indices_column = {ind for ind, _, _ in sorted_column[:top_n]}
                pareto_indices = pareto_indices.union(top_indices_sop).union(top_indices_column)
            else:
                sorted_pareto = sorted(pareto_front, key=lambda x: (x[1], x[2]), reverse=True)
                pareto_indices = {ind for ind, _, _ in sorted_pareto[:top_n]}

            top_indices = pareto_indices

        # Update the population to include only the selected candidates.
        self.population = [chromosome for idx, chromosome in enumerate(self.population) if idx in top_indices]
        self.calculate_fitness_score()

    def horizontal_crossover(self):
        """
        Performs horizontal crossover to generate new candidate alignments.

        Horizontal crossover creates a new candidate by combining parts of two parent candidates:
            - A random crossover point (row index) is chosen.
            - The child inherits the top part (up to the crossover point) from one parent and the
              bottom part from the other.
            - Deep copies are used to ensure that the child is independent of its parents.

        New candidates are generated until the population size reaches the target defined in config.

        Returns:
        --------
            None
        """
        new_individuals = []

        # Generate offspring until the desired population size is achieved.
        while len(self.population) + len(new_individuals) < config.POPULATION_SIZE:
            # Randomly select two distinct parent candidates.
            parent1 = random.choice(self.population)
            parent2 = random.choice(self.population)
            if parent1 is parent2:
                continue  # Ensure distinct parents

            num_seq = len(parent1)
            if num_seq < 2:
                continue  # Skip crossover if candidate contains only one sequence

            # Randomly select a crossover point (between 1 and num_seq-1).
            cut_index = random.randint(1, num_seq - 1)

            # Create a new candidate by combining deep copies of the two parents' segments.
            child = copy.deepcopy(parent1[:cut_index]) + copy.deepcopy(parent2[cut_index:])
            new_individuals.append(child)

        # Extend the population with the new offspring and recalculate fitness.
        self.population.extend(new_individuals)
        self.calculate_fitness_score()

    def mutation(self, model_path):
        """
        Applies mutation to selected candidate alignments by modifying their worst-performing sub-region.

        Process:
            1. Calculate the number of individuals to mutate based on MUTATION_RATE.
            2. Select the best candidates for mutation (using fitness scores).
            3. For each selected candidate, identify the worst-performing sub-board.
            4. Use a Deep Q-Network (DQN) agent (loaded from model_path) to iteratively mutate the sub-board.
            5. Replace the original sub-board with the mutated version in the candidate alignment.

        Parameters:
        -----------
            model_path (str): File path to the trained RL model used for mutation.

        Returns:
        --------
            None
        """
        # Determine how many individuals to mutate.
        num_individuals_to_mutate = round(
            config.POPULATION_SIZE * config.MUTATION_RATE)

        # Select the best individuals (indices) for mutation.
        best_fitted_individuals = utils.get_index_of_the_best_fitted_individuals(self.population_score,
                                                                                 num_individuals_to_mutate)

        for index in best_fitted_individuals:
            individual_to_mutate = self.population[index]

            # Identify the worst-performing sub-board (region) for mutation.
            score, worst_fitted_range = utils.calculate_worst_fitted_sub_board(individual_to_mutate, self.mode)
            from_row, to_row, from_column, to_column = worst_fitted_range

            # Extract the region (rows) to be mutated.
            row_genes = individual_to_mutate[from_row:to_row]
            sub_board = []

            # Ensure the extracted sub-board meets the required dimensions for the RL agent.
            while len(row_genes) < config.AGENT_WINDOW_ROW:
                row_genes.append([5] * config.AGENT_WINDOW_COLUMN)  # Fill missing rows with gaps (5)

            # For each row, extract and pad the sub-region.
            for genes in row_genes:
                sub_genes = genes[from_column:to_column]
                while len(sub_genes) < config.AGENT_WINDOW_COLUMN:
                    sub_genes.append(5)  # Fill missing columns with gaps (5)
                sub_board.append(sub_genes)

            # Create an Environment for the sub-board and load the RL agent.
            env = Environment(sub_board, convert_data=False)
            agent = DQN(env.action_number, env.row, env.max_len, env.max_len * env.max_reward)
            agent.load(model_path)
            state = env.reset()

            # Use the agent to mutate the sub-board until a termination condition is met.
            while True:
                action = agent.predict(state)
                _, next_state, done = env.step(action)
                state = next_state
                if done == 0:
                    break

            env.padding()  # Adjust alignment if needed after mutation

            # Replace the original sub-board in the candidate alignment with the mutated version.
            for idx, sequence in enumerate(env.aligned):
                if idx < len(row_genes):  # Avoid modifying padding rows beyond the original size
                    row_genes[idx][from_column:to_column] = sequence

            individual_to_mutate[from_row:to_row] = row_genes

    def print_hall_of_fame(self):
        """
        Prints the best individual currently stored in the Hall of Fame.

        - Displays the nucleotide sequence instead of numerical representation.
        - Shows the corresponding fitness score based on the GA mode.
        - Ensures visibility of the best solution found so far.
        """
        if not self.hall_of_fame:
            print("⚠️ Hall of Fame is empty! No best individual found yet.")
            return

        hof_individual, hof_score = self.hall_of_fame

        print("\n🏆 HALL OF FAME - BEST INDIVIDUAL SO FAR:")
        print("=" * 50)

        # Convert numerical sequence to nucleotides
        converted_hof = utils.get_nucleotides_seqs(hof_individual)

        for sequence in converted_hof:
            print(f"   🧬 {sequence}")

        # Display fitness scores based on mode
        if self.mode in {"sp", "cs"}:
            score_type = "SP" if self.mode == "sp" else "CS"
            print(f"   🎯 Best {score_type} Score: {hof_score[1]}")
        elif self.mode == "mo":
            print(f"   🎯 SP Score: {hof_score[1]}")
            print(f"   🎯 CS Score: {hof_score[2]}")

        print("=" * 50)

    def print_population(self):
        """
        Prints the current population along with their fitness scores.
        Provides a detailed view of the sequences and their fitness values.
        """
        print("\n📌 CURRENT POPULATION (Iteration {}):".format(self.current_iteration))
        print("=" * 50)

        if not self.population:
            print("⚠️ Empty Population! Make sure your population was generated correctly.")
            return

        for i, (individual, scores) in enumerate(zip(self.population, self.population_score), start=1):
            print(f"🔹 Individual {i}:")
            converted_individual = utils.get_nucleotides_seqs(individual)

            for sequence in converted_individual:
                print(f"   🧬 {sequence}")  # Print individual's sequences

            # Extract fitness scores based on mode
            if self.mode in {'sp', 'cs'}:
                score = scores[1]  # Single score (SP or CS)
                score_type = "SP" if self.mode == "sp" else "CS"
                print(f"   🎯 Fitness Score ({score_type}): {score}")

            elif self.mode == 'mo':
                sp_score, cs_score = scores[1], scores[2]
                print(f"   🎯 SP Score: {sp_score}")
                print(f"   🎯 CS Score: {cs_score}")

            print("-" * 50)

        self.print_hall_of_fame()
        print("\n✅ Population entirely printed.\n")

    def run(self, model_path, debug_mode=False):
        """
        Executes the complete Genetic Algorithm pipeline over multiple iterations.

        Process:
            1. Initialize the population.
            2. Calculate initial fitness scores.
            3. Validate that the RL model's window dimensions are compatible with the sequences.
            4. Iterate through GA steps:
                a. Apply mutation.
                b. Recalculate fitness scores.
                c. Perform selection.
                d. Execute horizontal crossover.
            5. Return the refined alignment from the best candidate in the Hall of Fame.

        Parameters:
        -----------
            model_path (str): Path to the RL model used for mutation.
            debug_mode (bool): If True, prints detailed log messages at each step.

        Returns:
        --------
            list: The nucleotide sequences (alignment) of the best candidate.
        """

        if debug_mode:
            print("🔵 Step 1: Initializing population...")

        self.generate_population()

        if debug_mode:
            print("✅ Population initialized!")

        self.calculate_fitness_score()

        if debug_mode:
            self.print_population()

        # Validate RL model window size
        first_individual = self.population[0]
        if config.AGENT_WINDOW_ROW > len(first_individual):
            raise ValueError("❌ AGENT_WINDOW_ROW is larger than the sequence count.")

        sequence_min_length = min(len(genes) for genes in first_individual)
        if config.AGENT_WINDOW_COLUMN > sequence_min_length:
            raise ValueError("❌ AGENT_WINDOW_COLUMN is larger than the shortest sequence length.")

        if debug_mode:
            print("🟢 Step 2: Starting Genetic Algorithm iterations...")

        # GA Iterations
        for i in range(config.GA_ITERATIONS):
            if debug_mode:
                print(f"🔄 Iteration {i + 1}/{config.GA_ITERATIONS}...")

            self.current_iteration += 1

            if debug_mode:
                print("  🔹 Applying mutation...")

            self.mutation(model_path)

            if debug_mode:
                print("  ✅ Mutation applied.")
                print("  🔹 Calculating fitness score...")

            self.calculate_fitness_score()

            if debug_mode:
                print("  ✅ Fitness score calculated.")
                self.print_population()

                print("  🔹 Performing selection...")

            self.selection()

            if debug_mode:
                print("  ✅ Selection completed.")
                self.print_population()

                print("  🔹 Performing horizontal crossover...")

            self.horizontal_crossover()

            if debug_mode:
                print("  ✅ Horizontal crossover completed.")
                self.print_population()

        if debug_mode:
            print("🟠 Step 3: Selecting the best chromosome...")

        best_chromosome, _ = self.hall_of_fame

        if debug_mode:
            print("✅ Best chromosome selected.")
            print("📜 Extracting aligned sequences...")

        aligned_seqs = utils.get_nucleotides_seqs(best_chromosome)

        if debug_mode:
            print("✅ Alignment completed!")

        return aligned_seqs

