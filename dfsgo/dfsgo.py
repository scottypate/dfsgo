import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import math
from collections import Counter


class Optimizer(object):
    """
    The LineupOptimizer takes a set of data containing daily fantasy sports
    data and returns optimized lineup options to choose from. The
    pos_requirements dict contains the required lineup composition. A genetic
    search method is used to construct a population of lineup solutions.

    Args:
        df (pandas.DataFrame): Daily fantasy projections and salary data.
        position_col (String): The position data column in the df.
        projection_col (String): The projections data column in the df.
        salary_col (String): The salary data column in the df.
        name_col (String): The player name data column in the df.

    Returns:
        (pandas.DataFrame): A dataframe with each row representing a possible
            lineup that can be played with salary and predicted points.
    """

    def __init__(self, df, position_col, projection_col, salary_col, name_col,
                 salary_cap, pos_requirements):

        self.salary_cap = salary_cap
        self.pos_requirements = pos_requirements
        self.position_col = position_col
        self.salary_col = salary_col
        self.projection_col = projection_col
        self.name_col = name_col

        # Calculate the cost per point.
        df['point_cost'] = df[salary_col] / df[projection_col]
        self.input_df = df

    def _build_arrays(self):
        """
        Turn the dataframe into arrays for each position with the row indices.
        """
        self.pos_indices = {}

        for pos in self.pos_requirements:
            self.pos_indices[pos] = list(
                self.input_df[
                    self.input_df[self.position_col].astype(str).str.contains(
                        self.pos_requirements[pos]['pos_num']
                    )
                ].index
            )

    def _validate_positions(self, lineup):
        """
        Make sure the lineup contains all required positions.

        Args:
            lineup (pandas.DataFrame): Represents a single possible lineup.

        Returns:
            (List[Strings]): Position shortfalls with each shortfall represented
                by an entry in the list.
        """
        available = list(lineup['pos_filled'].astype(str))
        shortfalls = []
        for pos in self.pos_requirements:
            # Keep track of which positions have been used so that no position
            # is used more than once.
            used = []
            count = 0

            for entry in available:
                if pos in entry:
                    count += 1
                    used.append(entry)
                    if count == self.pos_requirements[pos]['n']:
                        # Once the required number has been found stop.
                        break

            if count < self.pos_requirements[pos]['n']:
                for i in range(self.pos_requirements[pos]['n'] - count):
                    shortfalls.append(pos)

            # Subtract the used positions from the available positions
            c1 = Counter(available)
            c2 = Counter(used)
            diff = c1 - c2
            available = list(diff.elements())

        return shortfalls

    def _validate_salary(self, lineup, salary_tolerance):
        """
        Validate that the total salary for the lineup is within tolerance.
        If the lineup salary is too high then high cost, low value players are
        removed according to their point_cost. If the lineup salary is too low
        then cheap players are removed.

        Args:
            lineup (pandas.DataFrame): Represents a single possible lineup.
            salary_tolerance (Float): The required percent of the salary cap
                that must be used for the lineup to be accepted. Between 0-1.

        Returns:
            (pandas.DataFrame): A lineup with appropriate removals made.
        """
        salary = sum(lineup[self.salary_col])

        if salary > self.salary_cap:
            lineup = lineup.sort_values('point_cost', ascending=True)[:-3]

        elif salary < (self.salary_cap * salary_tolerance):
            lineup = lineup.sort_values(self.salary_col, ascending=False)[:-2]

        lineup = lineup.drop_duplicates()

        return lineup

    def _initial_lineup(self):
        """
        Initialize a possible lineup solution that meets position requirements.
        The initial lineup needs to be refined by salary.

        Returns (pandas.DataFrame): A dataframe representing a possible lineup.
        """

        lineup_indices = []
        pos_filled = []

        for pos in self.pos_requirements:
            lineup_indices.extend(
                np.random.choice(
                    a=self.pos_indices[pos],
                    size=self.pos_requirements[pos]['n'],
                    replace=False
                )
            )
            pos_filled.append(pos)

        lineup = self.input_df.ix[lineup_indices, :]
        lineup = lineup.assign(pos_filled=pos_filled)

        return lineup

    def _fill_shortfalls(self, lineup, shortfalls):
        """
        Fill the given position shortfalls in a lineup.

        Args:
            lineup (pandas.DataFrame): Represents a single possible lineup.
            shortfalls (List[Strings]): position shortfalls for the lineup.

        Returns:
            (pandas.DataFrame): A possible lineup solution that meets position
                requirements.
        """

        lineup_indices = []
        need_filled = []

        for need in shortfalls:
            lineup_indices.extend(
                np.random.choice(
                    a=[
                        x for x in self.pos_indices[need]
                        if (x not in list(lineup.index)) and
                           (x not in list(lineup_indices))
                       ],
                    size=1,
                    replace=False
                )
            )
            need_filled.append(need)

        additions = self.input_df.ix[lineup_indices, :]
        additions = additions.assign(pos_filled=need_filled)

        return pd.concat([lineup, additions])

    def _generate_lineup(self, salary_tolerance):
        """
        Generates a refined lineup that meets both position requirements and
        is within salary tolerance.

        Args:
            salary_tolerance (Float): The required percent of the salary cap
                that must be used for the lineup to be accepted. Between 0-1.

        Returns:
            (pandas.DataFrame): A lineup solution that meets requirements.
        """

        self._build_arrays()
        lineup = self._initial_lineup()
        lineup = self._validate_salary(
            lineup=lineup,
            salary_tolerance=salary_tolerance
        )
        shortfalls = self._validate_positions(lineup=lineup)

        while len(shortfalls) > 0:
            lineup = self._fill_shortfalls(lineup=lineup, shortfalls=shortfalls)
            lineup = self._validate_salary(
                lineup=lineup,
                salary_tolerance=salary_tolerance
            )
            shortfalls = self._validate_positions(lineup=lineup)

        return lineup

    def initialize_population(self, pn, salary_tolerance, n_processes):
        """
        Generate a population of possible lineup solutions that have been
        adjusted to position and salary requirements.

        Args:
            pn (Int): The number of lineups in the population
            n_processes (Int): Number of concurrent processes to run.
            salary_tolerance (Float): The required percent of the salary cap
                that must be used for the lineup to be accepted. Between 0-1.

        Returns:
            (List[pandas.DataFrame]): Contains pn possible lineup solutions.

        """
        print('Initializing population...')

        initial_population = []
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = [
                executor.submit(
                    self._generate_lineup, salary_tolerance
                ) for i in range(pn)
            ]

            for f in as_completed(futures):
                initial_population.append(f.result())

        return initial_population

    def _evaluate_fitness(self, population, cut):
        """
        Evaluate the fitness of lineups in a given population.

        Args:
            population (List[pandas.DataFrame]): A population of lineups
            cut (Float): Percent value used to segment a population. A cut
                value of 0.2 means that the bottom and top 20% of the population
                will be segmented for evaluation and replacement.

        Returns:
            (List[Floats]): The evaluation metrics for the mean, top, and bottom
                fitness evaluation.
        """
        scores = []
        size = len(population)

        for solution in population:
            points = sum(solution[self.projection_col])
            scores.append(points)

        average = np.mean(scores)
        bottom = np.mean(
            sorted(scores)[0: math.floor(size * cut)]
        )
        top = np.mean(
            sorted(scores)[-math.floor(size * cut):]
        )

        return [average, bottom, top]

    def _crossover(self, combined_df):
        """
        Take the combined dataframe and refine it to meet position and salary
        requirements. The output is a new lineup solution.

        Args:
            combined_df (pandas.DataFrame): A combination of two parents seleted
                for generating a new lineup solution.

        Returns:
            (pandas.DataFrame): A new lineup solution.
        """

        lineup_indices = []
        pos_filled = []

        for pos in self.pos_requirements:

            used = []
            subset = combined_df[combined_df['pos_filled'] == pos]

            try:
                sampled_indices = subset[~subset.index.isin(lineup_indices)]\
                    .sample(
                        n=self.pos_requirements[pos]['n'],
                        replace=False
                    ).index

                lineup_indices.extend(sampled_indices)
                used.extend(sampled_indices)
                combined_df.drop(used, inplace=True)
                pos_filled.append(pos)

            except ValueError:
                return None

        lineup = self.input_df.ix[lineup_indices, :]
        lineup = lineup.assign(pos_filled=pos_filled)

        return lineup

    def _mutate(self, lineup, p_mutation):
        """
        Performs a random mutation on a lineup given a mutation probability.

        Args:
            lineup (pandas.DataFrame): Represents a single possible lineup.
            p_mutation (Float): Probability that a random mutation will occur.
                Must be between 0-1.

        Returns:
            (pandas.DataFrame): Returns a lineup that has been mutated if a
                randomly generated number is below p_mutation threshold.
                Otherwise it returns the original lineup with no mutation.
        """
        n = random.random()
        if n <= p_mutation:
            try:
                mutation_choice = lineup.sample(1)
                mutation_index = mutation_choice.index
                pos = mutation_choice['pos_filled'].values[0]

                lineup.drop(mutation_index, inplace=True)
                replacement = np.random.choice(
                        a=[
                            x for x in self.pos_indices[pos]
                            if (x not in list(lineup.index))
                        ],
                        size=1,
                        replace=False
                )
                mutation_df = self.input_df.ix[replacement, :]
                mutation_df = mutation_df.assign(pos_filled=pos)
                mutated_lineup = pd.concat([
                    lineup, mutation_df
                ])

                return mutated_lineup

            except ValueError:
                return lineup
        else:
            return lineup

    def _generate_child(self, p_mutation, parent_1, parent_2):
        """
        Generates new possible lineup from a population through combination,
        crossover, and mutation. If the new lineup is below the salary cap then
        it is added to a new population for blending.

        Args:
            p_mutation (Float): Probability that a random mutation will occur.
                Must be between 0-1.
            parent_1 (pandas.DataFrame): One of the parents to generate a child
            parent_2 (pandas.DataFrame): One of the parents to generate a child

        Returns:
            (pandas.DataFrame): New lineup possibility.
        """

        combined_df = pd.concat([
            parent_1, parent_2
        ])

        combined_df.drop_duplicates(inplace=True)

        i = 0
        while True:
            i += 1
            child = self._crossover(combined_df=combined_df)
            if child is not None:
                child = self._mutate(lineup=child, p_mutation=p_mutation)
                if sum(child[self.salary_col]) <= self.salary_cap:
                    return child
                    break

            elif child is None and i > 10:
                return parent_1
                break

    def _generate_offspring(self, n, p_mutation, population, n_processes):
        """
        Generates n new possible lineups from a population through combination,
        crossover, and mutation. If the new lineup is below the salary cap then
        it is added to a new population for blending.

        Args:
            n (Int): Number of new solutions to generate.
            p_mutation (Float): Probability that a random mutation will occur.
                Must be between 0-1.
            population (List[pandas.DataFrames]): A population of lineups.

        Returns:
            (List[pandas.DataFramse]): New lineup possibilies.
        """
        weights = np.array([sum(x[self.projection_col]) for x in population])
        normalized_weights = weights / weights.sum(axis=0)

        parents = []

        for i in range(n):
            parents.append(list(np.random.choice(
                    a=[i for i in range(len(population))],
                    size=2,
                    p=normalized_weights,
                    replace=False
            )))

        offspring = []

        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = [
                executor.submit(
                    self._generate_child, p_mutation,
                    population[parents[i][0]], population[parents[i][1]]
                ) for i in range(len(parents))
            ]

            for f in as_completed(futures):
                offspring.append(f.result())

        return offspring

    def _blend_generation(self, old_population, bottom, new_population, top):
        """
        Remove the worst solutions from the old_population and replace them with
        the top solutions from the new generation.

        Args:
            old_population (List[pandas.DataFrames]): Population of possible
                lineups that is going to be culled.
            bottom (Float): Cutoff point for culling the old_population.
            new_population (List[pandas.DataFrames]): Population of possible
                lineups that is going to replace poor performers in
                old_population.
            top (Float): Cutoff point for new lineups to be inserted into the
                old_population.

        Returns:
            (List[pandas.DataFrames]): New population of possible lineups.
        """

        blended_generation = []

        for solution in old_population:
            points = sum(solution[self.projection_col])

            if points > bottom:
                blended_generation.append(solution)

        for solution in new_population:
            points = sum(solution[self.projection_col])

            if points > top:
                blended_generation.append(solution)

        return blended_generation

    def _population_to_df(self, population, n_solutions):
        """
        Turns a population of possible lineups into a concatenated dataframe
        where each row represents a possible lineup with salary and cost.

        Args:
            population (List[pandas.DataFrames]): A population of lineups.
            n_solutions (Int): Number of solutions in the final dataframe

        Returns:
            (pandas.DataFrame): All solutions in the final population.
        """

        lineups_dict = {
            'players': [],
            'pts': [],
            'cost': []
        }

        for lineup in population:
            lineups_dict['players'].append(
                list(lineup[self.name_col].str.replace(',', '-'))
            )
            lineups_dict['pts'].append(sum(lineup[self.projection_col]))
            lineups_dict['cost'].append(sum(lineup[self.salary_col]))

        return_df = pd.DataFrame(lineups_dict)
        return_df['pts'] = return_df['pts'].astype(int)
        return_df.drop_duplicates(inplace=True)
        return_df.sort_values('pts', ascending=False, inplace=True)

        try:
            return return_df.ix[:n_solutions, :]
        except KeyError:
            return return_df

    def evolve(self, n_generations, population, n_offspring, p_mutation, cut,
               n_processes, convergence, n_solutions):
        """
        Run the process end-to-end.

        Args:
            n_generations (Int): The number of generations to develop.
            population (List[pandas.DataFrames]): The initial population to
                run the evolution process on.
            n_offspring (Int): The number of offspring to produce from each
                generation.
            p_mutation (Float): Probability that a random mutation will occur.
                Must be between 0-1.
            cut (Float): Percent value used to segment a population. A cut
                value of 0.2 means that the bottom and top 20% of the population
                will be segmented for evaluation and replacement.
            n_processes (Int): The number of concurrent processes to run.
            convergence (Int): The fitness difference between generations that
                will stop the evolution process early.
            n_solutions (Int): Number of solutions in the final dataframe.

        Returns:
            (pandas.DataFrame): All solutions in the final population.
        """
        self._build_arrays()
        for i in range(1, n_generations):
            bottom = self._evaluate_fitness(population=population, cut=cut)[1]
            offspring_population = self._generate_offspring(
                n=n_offspring,
                p_mutation=p_mutation,
                population=population,
                n_processes=n_processes
            )
            top = self._evaluate_fitness(offspring_population, cut=cut)[2]
            population = self._blend_generation(
                old_population=population,
                new_population=offspring_population,
                bottom=bottom,
                top=top
            )
            fitness = self._evaluate_fitness(population=population, cut=cut)

            evaluation = {
                'Generation': i,
                'Average fitness': fitness[0],
                'Bottom fitness': fitness[1],
                'Top fitness': fitness[2]
            }

            print(evaluation)

            if (fitness[0] - fitness[1]) < convergence:
                print('Solution converged. Stopping early...')
                break

        return self._population_to_df(population, n_solutions=n_solutions)
