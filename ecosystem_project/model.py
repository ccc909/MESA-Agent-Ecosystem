import math

from mesa import Model
from mesa.datacollection import DataCollector
from .agents import Vegetation, Herbivore, Carnivore, Fire, Tornado, WeatherSystem
from mesa.experimental.cell_space import OrthogonalVonNeumannGrid
from mesa.experimental.devs import ABMSimulator


class EcosystemModel(Model):
    """Evolving Ecosystem Model.

    A model for simulating an ecosystem with evolving agent behaviors and natural disasters.
    """

    description = "A model for simulating an ecosystem with evolving agent behaviors and natural disasters."

    def __init__(
        self,
        width=20,
        height=20,
        initial_herbivores=80,
        initial_carnivores=15,
        herbivore_reproduce=0.04,
        carnivore_reproduce=0.03,
        carnivore_gain_from_food=20,
        vegetation=True,
        vegetation_regrowth_time=15,
        herbivore_gain_from_food=4,
        mutation_rate=0.05,
        evolution_method="genetic",
        enable_disasters=True,
        disaster_frequency=0.02,
        enable_weather=True,
        seed=None,
        simulator: ABMSimulator = None,
    ):
        """Create a new Ecosystem model with the given parameters.

        Args:
            height: Height of the grid
            width: Width of the grid
            initial_herbivores: Number of herbivores to start with
            initial_carnivores: Number of carnivores to start with
            herbivore_reproduce: Probability of each herbivore reproducing each step
            carnivore_reproduce: Probability of each carnivore reproducing each step
            carnivore_gain_from_food: Energy a carnivore gains from eating a herbivore
            vegetation: Whether to have vegetation for herbivores to eat
            vegetation_regrowth_time: How long it takes for vegetation to regrow
            herbivore_gain_from_food: Energy herbivores gain from vegetation
            mutation_rate: Rate at which genes mutate during reproduction
            evolution_method: Method of evolution ("genetic" or "reinforcement")
            enable_disasters: Whether to enable natural disasters
            disaster_frequency: Probability of a disaster occurring each step
            enable_weather: Whether to enable weather patterns affecting vegetation
            seed: Random seed
            simulator: ABMSimulator instance for event scheduling
        """
        super().__init__(seed=seed)
        self.simulator = simulator
        self.simulator.setup(self)

        self.height = height
        self.width = width
        self.vegetation = vegetation
        self.mutation_rate = mutation_rate
        self.evolution_method = evolution_method
        self.enable_disasters = enable_disasters
        self.disaster_frequency = disaster_frequency
        self.enable_weather = enable_weather
        self.base_vegetation_regrowth_time = vegetation_regrowth_time

        self.grid = OrthogonalVonNeumannGrid(
            [self.height, self.width],
            torus=True,
            capacity=5,
            random=self.random,
        )
        
        self.max_herbivores = self.width * self.height * 2
        self.max_carnivores = self.width * self.height // 2

        model_reporters = {
            "Carnivores": lambda m: len(m.agents_by_type[Carnivore]),
            "Herbivores": lambda m: len(m.agents_by_type[Herbivore]),
            "Average Carnivore Energy": lambda m: self._avg_energy(Carnivore),
            "Average Herbivore Energy": lambda m: self._avg_energy(Herbivore),
            "Average Carnivore Foraging": lambda m: self._avg_gene_value(Carnivore, "foraging_efficiency"),
            "Average Herbivore Foraging": lambda m: self._avg_gene_value(Herbivore, "foraging_efficiency"),
            "Average Herbivore Risk Aversion": lambda m: self._avg_gene_value(Herbivore, "risk_aversion"),
            "Fires": lambda m: len(m.agents_by_type[Fire]) if Fire in m.agents_by_type else 0,
            "Tornadoes": lambda m: len(m.agents_by_type[Tornado]) if Tornado in m.agents_by_type else 0,
            "Weather State": lambda m: m.weather_system.state if hasattr(m, "weather_system") else "normal",
            "Learning Method": lambda m: m.evolution_method,
        }
        
        if vegetation:
            model_reporters["Vegetation"] = lambda m: len(
                m.agents_by_type[Vegetation].select(lambda a: a.fully_grown)
            )

        self.datacollector = DataCollector(model_reporters)

        initial_herbivores = min(initial_herbivores, self.max_herbivores)
        initial_carnivores = min(initial_carnivores, self.max_carnivores)
        
        available_cells = list(self.grid.all_cells.cells)
        self.random.shuffle(available_cells)
        
        if len(available_cells) < initial_herbivores + initial_carnivores:
            initial_herbivores = int(len(available_cells) * 0.8)
            initial_carnivores = len(available_cells) - initial_herbivores
        
        herbivore_cells = self.random.sample(available_cells, initial_herbivores)
        available_cells = [cell for cell in available_cells if cell not in herbivore_cells]
        carnivore_cells = self.random.sample(available_cells, min(initial_carnivores, len(available_cells)))
        
        Herbivore.create_agents(
            self,
            initial_herbivores,
            energy=self.rng.random((initial_herbivores,)) * 2 * herbivore_gain_from_food,
            p_reproduce=herbivore_reproduce,
            energy_from_food=herbivore_gain_from_food,
            cell=herbivore_cells,
            mutation_rate=mutation_rate,
        )
        
        Carnivore.create_agents(
            self,
            initial_carnivores,
            energy=self.rng.random((initial_carnivores,)) * 2 * carnivore_gain_from_food,
            p_reproduce=carnivore_reproduce,
            energy_from_food=carnivore_gain_from_food,
            cell=carnivore_cells,
            mutation_rate=mutation_rate,
        )

        if vegetation:
            possibly_fully_grown = [True, False]
            for cell in self.grid:
                fully_grown = self.random.choice(possibly_fully_grown)
                countdown = (
                    0 if fully_grown else self.random.randrange(0, vegetation_regrowth_time)
                )
                Vegetation(self, countdown, vegetation_regrowth_time, cell)
        
        if enable_weather:
            self.weather_system = WeatherSystem(self, "normal", cycle_length=30)
        
        if enable_disasters and self.random.random() < 0.1:
            self._generate_random_disaster()

        self.running = True
        self.datacollector.collect(self)
    
    def _generate_random_disaster(self):
        """Generate a random disaster event in the ecosystem."""
        available_cells = [cell for cell in list(self.grid) 
                          if len(cell.agents) < self.grid.capacity]
        
        if not available_cells:
            return
            
        random_cell = self.random.choice(available_cells)
        
        if self.random.random() < 0.7:
            Fire(self, random_cell)
        else:
            Tornado(self, random_cell)
    
    def _avg_energy(self, agent_class):
        """Calculate the average energy of a specific agent type.
        
        Args:
            agent_class: Class of agents to average over
            
        Returns:
            Average energy or 0 if no agents exist
        """
        agents = self.agents_by_type[agent_class]
        if not agents:
            return 0
        return sum(agent.energy for agent in agents) / len(agents)
    
    def _avg_gene_value(self, agent_class, gene_name):
        """Calculate the average value of a specific gene across agent population.
        
        Args:
            agent_class: Class of agents to average over
            gene_name: Name of the gene to average
            
        Returns:
            Average gene value or 0 if no agents exist
        """
        agents = self.agents_by_type[agent_class]
        if not agents:
            return 0
        return sum(agent.genes.values[gene_name] for agent in agents) / len(agents)
        
    def step(self):
        """Execute one step of the model."""
        if self.evolution_method == "swarm" and hasattr(self, 'swarm_system'):
            self.swarm_system.update_signals()
            
        self.agents_by_type[Herbivore].shuffle_do("step")
        self.agents_by_type[Carnivore].shuffle_do("step")
        
        if Fire in self.agents_by_type:
            self.agents_by_type[Fire].shuffle_do("step")
        if Tornado in self.agents_by_type:
            self.agents_by_type[Tornado].shuffle_do("step")
        
        if self.enable_weather and hasattr(self, "weather_system"):
            self.weather_system.step()
        
        if self.enable_disasters and self.random.random() < self.disaster_frequency:
            self._generate_random_disaster()

        self.datacollector.collect(self)
        
        if len(self.agents_by_type[Herbivore]) == 0 or len(self.agents_by_type[Carnivore]) == 0:
            if not self.vegetation:
                self.running = False
