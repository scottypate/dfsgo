# dfsgo
 
The dfsgo optimizer takes a set of data containing daily fantasy sports
data and returns optimized lineup options to choose from. The
pos_requirements dict contains the required lineup composition. A genetic
search method is used to construct a population of lineup solutions.

    Args:
        df (pandas.DataFrame): Daily fantasy projections and salary data.
        position_col (String): The position data column in the df.
        projection_col (String): The projections data column in the df.
        salary_col (String): The salary data column in the df.
        name_col (String): The player name data column in the df.
        salary_cap (Integer): The salary cap of the daily fantasy game.
        pos_requirements (Dict): The position strings and quantity constraints.
    
    Returns:
        (pandas.DataFrame): A dataframe with each row representing a possible
            lineup that can be played with salary and predicted points. 
            
### Installation
`pip install git+https://github.com/scottypate/dfsgo.git`

### Example

```python
from dfsgo.dfsgo import Optimizer

# Set position requirements
pos_requirements = {
    'PG': {'n': 1, 'pos_num': '1'},
    'SG': {'n': 1, 'pos_num': '2'},
    'PF': {'n': 1, 'pos_num': '4'},
    'SF': {'n': 1, 'pos_num': '3'},
    'C': {'n': 1, 'pos_num': '5'},
    'G': {'n': 1, 'pos_num': '12'},
    'F': {'n': 1, 'pos_num': '34'},
    'Util': {'n': 1, 'pos_num': ''}
}

# Setup the optimizer
optimizer = Optimizer(
    df=df,
    position_col='pos',
    name_col='name',
    projection_col='prediction',
    salary_col='salary',
    salary_cap=50000,
    pos_requirements=pos_requirements
)

# pn is the number of lineups in the initial population.
# salary_tolerance is the percent of total salary cap that must be used
# in each lineup
initial_population = optimizer.initialize_population(
    pn=20000,
    salary_tolerance=0.9,
    n_processes=100
)

# The evolution process will run n_generations unless it converges on a
# solution earlier in which case it will stop early.
lineups_df = optimizer.evolve(
    population=initial_population,
    n_generations=15,
    n_offspring=2000,
    p_mutation=0.12,
    cut=0.5,
    n_processes=5,
    convergence=3,
    n_solutions=1000
)
```