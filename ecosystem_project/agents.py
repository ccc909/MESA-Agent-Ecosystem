from mesa.experimental.cell_space import CellAgent, FixedAgent


class Gene:
    """Represents a gene that controls agent behavior."""

    def __init__(self, model, gene_values=None):
        """Initialize a gene with random or specified values.
        
        Args:
            model: Model instance for random number generation
            gene_values: Dictionary of gene values or None for random initialization
        """
        self.model = model
        
        if gene_values is None:
            self.values = {
                "movement_speed": model.random.uniform(0.5, 1.5),
                "directedness": model.random.uniform(0.0, 1.0),
                
                "foraging_efficiency": model.random.uniform(0.5, 1.5),
                "detection_range": model.random.uniform(0.8, 1.2),
                
                "reproduction_threshold": model.random.uniform(0.8, 1.2),
                
                "metabolism_rate": model.random.uniform(0.8, 1.2),
                "risk_aversion": model.random.uniform(0.0, 1.0),
            }
        else:
            self.values = gene_values

    def mutate(self, mutation_rate):
        """Apply random mutations to genes based on mutation rate.
        
        Args:
            mutation_rate: Probability of each gene mutating
        
        Returns:
            A new Gene instance with potentially mutated values
        """
        new_values = {}
        for gene, value in self.values.items():
            if self.model.random.random() < mutation_rate:
                change = self.model.random.uniform(-0.2, 0.2)
                new_values[gene] = max(0.1, min(2.0, value + change))
            else:
                new_values[gene] = value
                
        return Gene(self.model, new_values)
        
    def crossover(self, other_gene):
        """Create a new gene by combining this gene with another.
        
        Args:
            other_gene: Another Gene instance to crossover with
            
        Returns:
            A new Gene instance with mixed values from both parents
        """
        new_values = {}
        for gene in self.values:
            if self.model.random.random() < 0.5:
                new_values[gene] = self.values[gene]
            else:
                new_values[gene] = other_gene.values[gene]
                
        return Gene(self.model, new_values)


class Animal(CellAgent):
    """Base class for all animals in the ecosystem."""

    def __init__(
        self, model, energy=10, p_reproduce=0.04, energy_from_food=4, 
        genes=None, cell=None, mutation_rate=0.05
    ):
        """Initialize an animal.

        Args:
            model: Model instance
            energy: Starting amount of energy
            p_reproduce: Base probability of reproduction
            energy_from_food: Base energy obtained from 1 unit of food
            genes: Gene instance or None for random initialization
            cell: Cell in which the animal starts
            mutation_rate: Rate of mutation for offspring
        """
        super().__init__(model)
        self.energy = energy
        self.base_p_reproduce = p_reproduce
        self.base_energy_from_food = energy_from_food
        self.cell = cell
        self.age = 0
        self.mutation_rate = mutation_rate
        self.memory = []
        self.reward_history = []
        
        self.genes = genes if genes is not None else Gene(model)
        
        self.learning_method = model.evolution_method
        
        if self.learning_method == "reinforcement":
            from .reinforcement_learning import QLearning
            self.q_learner = QLearning(model)
        
        self.update_traits()
        
    def update_traits(self):
        """Calculate effective traits based on current genes and learning method."""
        self.p_reproduce = self.base_p_reproduce * self.genes.values["reproduction_threshold"]
        
        self.energy_from_food = self.base_energy_from_food * self.genes.values["foraging_efficiency"]
        
        self.metabolism = self.genes.values["metabolism_rate"]
        
        if self.learning_method != "genetic":
            if hasattr(self, 'neural_net'):
                if len(self.memory) > 5:
                    self.neural_net.train(batch_size=min(10, len(self.memory)))
                    inputs = [self.energy/50, self.age/30, self.metabolism, 
                              self.p_reproduce, self.genes.values["risk_aversion"]]
                    outputs = self.neural_net.forward(inputs)
                    
                    self.metabolism *= max(0.8, min(1.2, outputs[0, 0] * 2))
                    self.energy_from_food *= max(0.8, min(1.2, outputs[0, 1] * 2))
            
            elif self.learning_method == "reinforcement" and hasattr(self, 'q_learner'):
                if len(self.reward_history) > 5:
                    avg_reward = self.q_learner.get_average_reward()
                    reward_factor = max(0.8, min(1.3, 1.0 + avg_reward / 10))
                    self.energy_from_food *= reward_factor
                    self.metabolism *= max(0.9, min(1.1, 1.0 - avg_reward / 20))
    
    def spawn_offspring(self, mate=None):
        """Create offspring with genetic inheritance.
        
        Args:
            mate: Optional mate for sexual reproduction
        """
        if isinstance(self, Herbivore) and len(self.model.agents_by_type[Herbivore]) >= self.model.max_herbivores:
            return
        elif isinstance(self, Carnivore) and len(self.model.agents_by_type[Carnivore]) >= self.model.max_carnivores:
            return
            
        if len(self.cell.agents) >= self.model.grid.capacity:
            available_cells = [cell for cell in self.cell.neighborhood 
                               if len(cell.agents) < self.model.grid.capacity]
            if not available_cells:
                return
            target_cell = self.model.random.choice(available_cells)
        else:
            target_cell = self.cell
            
        self.energy /= 2
        
        if mate is not None and isinstance(mate, self.__class__):
            offspring_genes = self.genes.crossover(mate.genes)
            offspring_genes = offspring_genes.mutate(self.mutation_rate)
        else:
            offspring_genes = self.genes.mutate(self.mutation_rate)
        
        self.__class__(
            self.model,
            self.energy,
            self.base_p_reproduce,
            self.base_energy_from_food,
            offspring_genes,
            target_cell,
            self.mutation_rate,
        )

    def feed(self):
        """Abstract method to be implemented by subclasses."""
        pass

    def move(self):
        """Abstract method to be implemented by subclasses."""
        pass
        
    def find_mate(self):
        """Find a suitable mate in the current cell."""
        potential_mates = [
            agent for agent in self.cell.agents 
            if isinstance(agent, self.__class__) and agent != self and agent.energy > 10
        ]
        return self.model.random.choice(potential_mates) if potential_mates else None

    def step(self):
        """Execute one step of the animal's behavior."""
        self.age += 1
        
        self.move()

        self.energy -= 1 * self.metabolism

        self.feed()

        if self.energy < 0:
            self.remove()
        elif (self.energy > 20 * self.genes.values["reproduction_threshold"] and 
              self.model.random.random() < self.p_reproduce):
            mate = self.find_mate()
            self.spawn_offspring(mate)


class Herbivore(Animal):
    """A herbivore that eats vegetation and can be eaten by carnivores."""

    def feed(self):
        """If possible, eat vegetation at current location."""
        vegetation_patch = next(
            (obj for obj in self.cell.agents if isinstance(obj, Vegetation)), None
        )
        if vegetation_patch and vegetation_patch.fully_grown:
            self.energy += self.energy_from_food
            vegetation_patch.fully_grown = False

    def move(self):
        """Move based on genes, balancing foraging needs with predator avoidance."""
        nearby_cells = self.cell.neighborhood
        
        if self.learning_method == "reinforcement" and hasattr(self, 'q_learner'):
            current_energy = self.energy / 50
            
            vegetation_present_current = any(
                isinstance(agent, Vegetation) and agent.fully_grown 
                for agent in self.cell.agents
            )
            
            predators_nearby = False
            for cell in nearby_cells:
                if any(isinstance(agent, Carnivore) for agent in cell.agents):
                    predators_nearby = True
                    break
            
            state = {
                'energy': current_energy, 
                'danger': 1 if predators_nearby else 0,
                'food': 1 if vegetation_present_current else 0
            }
            
            action = self.q_learner.select_action(state)
            
            if not nearby_cells:
                return
                
            if action < len(nearby_cells):
                new_cell = list(nearby_cells)[action % len(nearby_cells)]
                
                if len(new_cell.agents) < self.model.grid.capacity:
                    self.cell = new_cell
                
                reward = 0
                
                if any(isinstance(agent, Vegetation) and agent.fully_grown for agent in new_cell.agents):
                    reward += 2
                
                if any(isinstance(agent, Carnivore) for agent in new_cell.agents):
                    reward -= 5
                
                reward -= 0.1
                
                new_vegetation_present = any(
                    isinstance(agent, Vegetation) and agent.fully_grown 
                    for agent in new_cell.agents
                )
                new_predators_nearby = any(
                    isinstance(agent, Carnivore) for agent in new_cell.agents
                )
                new_state = {
                    'energy': self.energy / 50,
                    'danger': 1 if new_predators_nearby else 0,
                    'food': 1 if new_vegetation_present else 0
                }
                
                self.q_learner.update_q_value(state, action, reward, new_state)
                self.reward_history.append(reward)
                if len(self.reward_history) > 100:
                    self.reward_history.pop(0)
                
                return
        
        cell_scores = {}
        
        for cell in nearby_cells:
            score = 0
            
            vegetation_present = any(
                isinstance(agent, Vegetation) and agent.fully_grown 
                for agent in cell.agents
            )
            if vegetation_present:
                score += 2 * self.genes.values["foraging_efficiency"]
            
            predators_present = any(
                isinstance(agent, Carnivore) for agent in cell.agents
            )
            if predators_present:
                score -= 5 * self.genes.values["risk_aversion"]
            
            cell_scores[cell] = score
        
        randomness = 1.0 - self.genes.values["directedness"]
        
        for cell in cell_scores:
            cell_scores[cell] += self.model.random.uniform(0, 5) * randomness
        
        if not cell_scores:
            return
            
        best_cells = [
            cell for cell, score in cell_scores.items() 
            if score == max(cell_scores.values()) and len(cell.agents) < self.model.grid.capacity
        ]
        
        if best_cells:
            self.cell = self.model.random.choice(best_cells)


class Carnivore(Animal):
    """A carnivore that hunts and eats herbivores."""

    def feed(self):
        """Hunt and eat a herbivore if present in current cell."""
        herbivores = [
            agent for agent in self.cell.agents if isinstance(agent, Herbivore)
        ]
        
        if herbivores:
            hunting_success = min(1.0, self.genes.values["foraging_efficiency"])
            
            if self.model.random.random() < hunting_success:
                prey = self.model.random.choice(herbivores)
                self.energy += self.energy_from_food
                prey.remove()

    def move(self):
        """Move based on genes, focusing on hunting herbivores."""
        extended_neighborhood = self.cell.neighborhood
        
        if self.learning_method == "reinforcement" and hasattr(self, 'q_learner'):
            current_energy = self.energy / 50
            
            prey_counts = {}
            max_prey = 0
            
            for cell in extended_neighborhood:
                prey_count = sum(1 for agent in cell.agents if isinstance(agent, Herbivore))
                prey_counts[cell] = prey_count
                
                if prey_count > max_prey:
                    max_prey = prey_count
            
            prey_presence = min(1.0, max_prey / 3)
            
            state = {
                'energy': current_energy,
                'food': prey_presence
            }
            
            action = self.q_learner.select_action(state)
            
            if not extended_neighborhood:
                return
                
            if action < len(extended_neighborhood):
                new_cell = list(extended_neighborhood)[action % len(extended_neighborhood)]
                
                if len(new_cell.agents) < self.model.grid.capacity:
                    self.cell = new_cell
                
                reward = 0
                
                prey_in_new_cell = sum(1 for agent in new_cell.agents if isinstance(agent, Herbivore))
                reward += prey_in_new_cell * 3
                
                reward -= 0.2
                
                new_prey_count = sum(1 for agent in new_cell.agents if isinstance(agent, Herbivore))
                new_prey_presence = min(1.0, new_prey_count / 3)
                
                new_state = {
                    'energy': self.energy / 50,
                    'food': new_prey_presence
                }
                
                self.q_learner.update_q_value(state, action, reward, new_state)
                self.reward_history.append(reward)
                if len(self.reward_history) > 100:
                    self.reward_history.pop(0)
                
                return
        
        cell_scores = {}
        
        for cell in extended_neighborhood:
            herbivore_count = sum(
                1 for agent in cell.agents if isinstance(agent, Herbivore)
            )
            
            cell_scores[cell] = herbivore_count * self.genes.values["foraging_efficiency"]
        
        randomness = 1.0 - self.genes.values["directedness"]
        
        for cell in cell_scores:
            cell_scores[cell] += self.model.random.uniform(0, 3) * randomness
        
        if not cell_scores:
            return
            
        best_score = max(cell_scores.values())
        best_cells = [
            cell for cell, score in cell_scores.items() 
            if score == best_score and len(cell.agents) < self.model.grid.capacity
        ]
        
        if best_cells and self.model.random.random() < self.genes.values["movement_speed"]:
            self.cell = self.model.random.choice(best_cells)


class Vegetation(FixedAgent):
    """A patch of vegetation that grows at a fixed rate and can be eaten by herbivores."""

    @property
    def fully_grown(self):
        """Whether the vegetation patch is fully grown."""
        return self._fully_grown

    @fully_grown.setter
    def fully_grown(self, value: bool) -> None:
        """Set vegetation growth state and schedule regrowth if eaten."""
        self._fully_grown = value

        if not value:
            self.model.simulator.schedule_event_relative(
                setattr,
                self.regrowth_time,
                function_args=[self, "fully_grown", True],
            )

    def __init__(self, model, countdown, regrowth_time, cell):
        """Create a new patch of vegetation.

        Args:
            model: Model instance
            countdown: Time until vegetation is fully grown again
            regrowth_time: Time needed to regrow after being eaten
            cell: Cell to which this vegetation patch belongs
        """
        super().__init__(model)
        self._fully_grown = countdown == 0
        self.regrowth_time = regrowth_time
        self.cell = cell

        if not self.fully_grown:
            self.model.simulator.schedule_event_relative(
                setattr, countdown, function_args=[self, "fully_grown", True]
            )


class Fire(FixedAgent):
    """A fire that spreads, destroys vegetation, and can harm animals."""

    def __init__(self, model, cell, duration=10, spread_rate=0.3, damage=5):
        """Create a new fire disaster.

        Args:
            model: Model instance
            cell: Cell where the fire starts
            duration: How long the fire lasts before burning out
            spread_rate: Probability of spreading to adjacent cells each step
            damage: Amount of damage (energy reduction) to animals in the same cell
        """
        super().__init__(model)
        self.cell = cell
        self.duration = duration
        self.spread_rate = spread_rate
        self.damage = damage
        self.age = 0

        self.model.simulator.schedule_event_relative(
            self.remove, duration
        )

    def step(self):
        """Execute one step of fire behavior: cause damage and potentially spread."""
        self.age += 1
        
        for agent in list(self.cell.agents):
            if isinstance(agent, Vegetation):
                agent.fully_grown = False
            elif isinstance(agent, Animal):
                agent.energy -= self.damage
                if agent.energy <= 0:
                    agent.remove()
        
        if self.age < self.duration - 1:
            for cell in self.cell.neighborhood:
                if (self.model.random.random() < self.spread_rate and 
                    not any(isinstance(agent, Fire) for agent in cell.agents) and
                    len(cell.agents) < self.model.grid.capacity):
                    has_vegetation = any(
                        isinstance(agent, Vegetation) and agent.fully_grown 
                        for agent in cell.agents
                    )
                    if (has_vegetation or 
                        self.model.random.random() < 0.2):
                        new_duration = max(2, self.duration - 2)
                        new_spread = max(0.1, self.spread_rate - 0.1)
                        Fire(self.model, cell, new_duration, new_spread, self.damage)


class Tornado(CellAgent):
    """A moving tornado that damages everything in its path and relocates animals."""

    def __init__(self, model, cell, lifetime=15, damage=3):
        """Create a new tornado disaster.

        Args:
            model: Model instance
            cell: Cell where the tornado starts
            lifetime: How many steps the tornado lasts
            damage: Energy reduction to animals hit by the tornado
        """
        super().__init__(model)
        self.cell = cell
        self.lifetime = lifetime
        self.damage = damage
        self.age = 0
        self.path = []
        
        self.direction = self.model.random.randint(0, 3)

    def step(self):
        """Execute one step of tornado behavior: move, cause damage, relocate animals."""
        self.age += 1
        
        for agent in list(self.cell.agents):
            if isinstance(agent, Vegetation):
                if self.model.random.random() < 0.7:
                    agent.fully_grown = False
            elif isinstance(agent, Animal):
                agent.energy -= self.damage
                
                if self.model.random.random() < 0.8:
                    available_cells = [cell for cell in list(self.model.grid) 
                                      if len(cell.agents) < self.model.grid.capacity]
                    if available_cells:
                        random_cell = self.model.random.choice(available_cells)
                        agent.cell = random_cell
        
        self.path.append(self.cell)
        
        if self.age < self.lifetime:
            if self.model.random.random() < 0.2:
                self.direction = self.model.random.randint(0, 3)
                
            neighbors = list(self.cell.neighborhood)
            
            if neighbors:
                
                if self.model.random.random() < 0.3:
                    self.direction = self.model.random.randint(0, 3)
                
                target_cells = [cell for cell in neighbors if len(cell.agents) < self.model.grid.capacity]
                
                if target_cells:
                    self.cell = self.model.random.choice(target_cells)
        else:
            self.remove()


class WeatherSystem:
    """A system that controls weather patterns affecting vegetation growth."""
    
    def __init__(self, model, initial_state="normal", cycle_length=30):
        """Initialize weather system.
        
        Args:
            model: Model instance
            initial_state: Starting weather state
            cycle_length: Average length of a weather cycle
        """
        self.model = model
        self.states = ["drought", "normal", "rainy"]
        self.state = initial_state
        self.cycle_length = cycle_length
        self.days_in_state = 0
        
        self.regrowth_multipliers = {
            "drought": 2.5,
            "normal": 1.0,
            "rainy": 0.5
        }
        
        self._schedule_weather_change()
    
    def _schedule_weather_change(self):
        """Schedule the next weather change."""
        variance = 0.5 if self.state == "normal" else 0.8
        
        base_time = self.cycle_length
        change_time = int(base_time * (1.0 + self.model.random.uniform(-variance, variance)))
        
        self.model.simulator.schedule_event_relative(
            self._change_weather, change_time
        )
    
    def _change_weather(self):
        """Change to a new weather state."""
        current_idx = self.states.index(self.state)
        
        if self.state == "normal":
            if self.model.random.random() < 0.5:
                next_idx = 0
            else:
                next_idx = 2
        else:
            if self.model.random.random() < 0.7:
                next_idx = 1
            else:
                next_idx = current_idx
        
        self.state = self.states[next_idx]
        self.days_in_state = 0
        
        self._apply_weather_effects()
        
        self._schedule_weather_change()
    
    def _apply_weather_effects(self):
        """Apply effects of current weather to the model."""
        multiplier = self.regrowth_multipliers[self.state]
        
        for agent in self.model.agents_by_type[Vegetation]:
            agent.regrowth_time = int(self.model.base_vegetation_regrowth_time * multiplier)
        
        if self.state == "drought" and self.model.random.random() < 0.1:
            random_cell = self.model.random.choice(list(self.model.grid))
            Fire(self.model, random_cell)
        
        if self.state == "rainy" and self.model.random.random() < 0.05:
            random_cell = self.model.random.choice(list(self.model.grid))
            Tornado(self.model, random_cell)
    
    def step(self):
        """Update weather state each step."""
        self.days_in_state += 1
