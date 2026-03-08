import csv
import os
from tqdm import tqdm
import copy

import config
from DPAMSA.env import Environment
from GA import GA
import utils

class NSGA2_GA(GA):

    def non_dominated_sort(self, combined_scores):
        """
        combined_scores: list of tuples (index, sp, cs)
        Returns: list of fronts (each front is list of indices)
        """

        domination_count = {}
        dominated_set = {}
        fronts = [[]]

        for p in combined_scores:
            p_idx, p_sp, p_cs = p
            domination_count[p_idx] = 0
            dominated_set[p_idx] = []

            for q in combined_scores:
                q_idx, q_sp, q_cs = q
                if p_idx == q_idx:
                    continue

                # Check if p dominates q
                if (p_sp >= q_sp and p_cs >= q_cs) and (p_sp > q_sp or p_cs > q_cs):
                    dominated_set[p_idx].append(q_idx)

                # Check if q dominates p
                elif (q_sp >= p_sp and q_cs >= p_cs) and (q_sp > p_sp or q_cs > p_cs):
                    domination_count[p_idx] += 1

            if domination_count[p_idx] == 0:
                fronts[0].append(p_idx)

        i = 0
        while fronts[i]:
            next_front = []

            for p_idx in fronts[i]:
                for q_idx in dominated_set[p_idx]:
                    domination_count[q_idx] -= 1
                    if domination_count[q_idx] == 0:
                        next_front.append(q_idx)

            i += 1
            fronts.append(next_front)

        fronts.pop()  # remove last empty front
        return fronts

    def compute_crowding_distance(self, front, combined_scores):
        """
        front: list of indices
        combined_scores: list of (index, sp, cs)
        Returns: dict {index: distance}
        """

        distance = {idx: 0 for idx in front}

        if len(front) <= 2:
            for idx in front:
                distance[idx] = float('inf')
            return distance

        # Create lookup dictionary
        score_dict = {idx: (sp, cs) for idx, sp, cs in combined_scores}

        for objective in [0, 1]:  # 0 = SP, 1 = CS

            sorted_front = sorted(
                front,
                key=lambda idx: score_dict[idx][objective]
            )

            min_val = score_dict[sorted_front[0]][objective]
            max_val = score_dict[sorted_front[-1]][objective]

            distance[sorted_front[0]] = float('inf')
            distance[sorted_front[-1]] = float('inf')

            if max_val - min_val == 0:
                continue

            for i in range(1, len(sorted_front) - 1):
                prev_val = score_dict[sorted_front[i - 1]][objective]
                next_val = score_dict[sorted_front[i + 1]][objective]

                distance[sorted_front[i]] += (next_val - prev_val) / (max_val - min_val)

        return distance

    def nsga2_selection(self, parent_pop, offspring_pop, parent_scores, offspring_scores):

        combined_population = parent_pop + offspring_pop

        raw_scores = parent_scores + offspring_scores

        combined_scores = []

        for i, score in enumerate(raw_scores):

            if len(score) == 3:
                _, sp, cs = score
            else:
                sp, cs = score

            combined_scores.append((i, sp, cs))

        fronts = self.non_dominated_sort(combined_scores)

        new_population = []
        new_scores = []

        score_dict = {idx: (sp, cs) for idx, sp, cs in combined_scores}

        for front in fronts:

            if len(new_population) + len(front) > self.population_size:

                distance = self.compute_crowding_distance(front, combined_scores)

                sorted_front = sorted(
                    front,
                    key=lambda idx: distance[idx],
                    reverse=True
                )

                remaining = self.population_size - len(new_population)
                selected = sorted_front[:remaining]

                for idx in selected:
                    new_population.append(combined_population[idx])
                    sp, cs = score_dict[idx]
                    new_scores.append((idx, sp, cs))

                break

            else:
                for idx in front:
                    new_population.append(combined_population[idx])
                    sp, cs = score_dict[idx]
                    new_scores.append((idx, sp, cs))

        self.population = new_population
        self.population_score = new_scores

    def run(self, model_path, debug_mode=False):

        self.generate_population()
        self.calculate_fitness_score()

        for i in range(config.GA_ITERATIONS):

            # Save parents
            parent_population = copy.deepcopy(self.population)
            parent_scores = copy.deepcopy(self.population_score)

            # Generate offspring
            self.mutation(model_path)
            self.horizontal_crossover()
            self.calculate_fitness_score()

            offspring_population = copy.deepcopy(self.population)
            offspring_scores = copy.deepcopy(self.population_score)

            # NSGA-II selection
            self.nsga2_selection(
                parent_population,
                offspring_population,
                parent_scores,
                offspring_scores
            )

        best_chromosome, _ = self.hall_of_fame
        return utils.get_nucleotides_seqs(best_chromosome)