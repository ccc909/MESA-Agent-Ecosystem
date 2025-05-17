from .agents import Vegetation, Herbivore, Carnivore, Fire, Tornado
from .model import EcosystemModel
from mesa.experimental.devs import ABMSimulator
import random
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
    make_space_component,
)


def ecosystem_portrayal(agent):
    if agent is None:
        return

    portrayal = {
        "size": 25,
        "stroke_color": "#000000",
        "stroke_width": 1,
        "zorder": 1,
    }

    if isinstance(agent, Carnivore):
        size = 20 + (agent.energy / 5)
        
        portrayal["size"] = size
        portrayal["color"] = (0.8, 0.0, 0.0)
        portrayal["opacity"] = 0.9
        portrayal["zorder"] = 3
        portrayal["stroke_width"] = 1.5
        
        portrayal["text"] = f"{agent.energy:.0f}"
        portrayal["text_color"] = "white"
        
        if agent.age > 30:
            portrayal["stroke_color"] = "#FF0000"
            portrayal["stroke_width"] = 2
    
    elif isinstance(agent, Herbivore):
        size = 15 + (agent.energy / 8)
        
        portrayal["size"] = size
        portrayal["color"] = (0.0, 0.0, 0.8)
        portrayal["opacity"] = 0.9
        portrayal["zorder"] = 2
        
        if agent.energy > 15:
            portrayal["text"] = f"{agent.energy:.0f}"
            portrayal["text_color"] = "white"
            
        if agent.genes.values["foraging_efficiency"] > 1.3:
            portrayal["stroke_color"] = "#FFFF00"
            portrayal["stroke_width"] = 2
    
    elif isinstance(agent, Vegetation):
        if agent.fully_grown:
            height = 1.0
            color = (0.0, 180/255, 0.0)
            size = 75
        else:
            height = 0.5
            color = (150/255, 100/255, 0.0)
            size = 60
        
        portrayal["shape"] = "vegetation"
        portrayal["size"] = size
        portrayal["color"] = color
        portrayal["opacity"] = 0.8 + (height * 0.2)
        portrayal["zorder"] = 1
        portrayal["stroke_width"] = 0
    
    elif isinstance(agent, Fire):
        portrayal["shape"] = "circle"
        portrayal["size"] = 40 + (agent.age * 2)
        portrayal["color"] = (1.0, min(1.0, 0.2 + agent.age * 0.05), 0)
        portrayal["opacity"] = 0.8
        portrayal["zorder"] = 4
        portrayal["stroke_color"] = "#FF0000"
        portrayal["stroke_width"] = 2
        
    elif isinstance(agent, Tornado):
        portrayal["shape"] = "circle"
        portrayal["size"] = 35
        portrayal["color"] = (0.7, 0.7, 0.9)
        portrayal["opacity"] = 0.7
        portrayal["zorder"] = 5
        portrayal["stroke_color"] = "#000080"
        portrayal["stroke_width"] = 2
        portrayal["text"] = "⌀"
        portrayal["text_color"] = "white"

    return portrayal


model_params = {
    "seed": {
        "type": "InputText",
        "value": int(random.random() * 100),
        "label": "Random Seed",
    },
    "width": Slider("Grid Width", 20, 10, 50, 1),
    "height": Slider("Grid Height", 20, 10, 50, 1),
    "vegetation": {
        "type": "Select",
        "value": True,
        "values": [True, False],
        "label": "Vegetation Growth Enabled?",
    },
    "vegetation_regrowth_time": Slider("Vegetation Regrowth Time", 15, 1, 50),
    "initial_herbivores": Slider("Initial Herbivore Population", 80, 10, 300),
    "herbivore_reproduce": Slider("Herbivore Reproduction Rate", 0.04, 0.01, 1.0, 0.01),
    "initial_carnivores": Slider("Initial Carnivore Population", 15, 5, 100),
    "carnivore_reproduce": Slider(
        "Carnivore Reproduction Rate",
        0.03,
        0.01,
        1.0,
        0.01,
    ),
    "carnivore_gain_from_food": Slider("Carnivore Energy from Prey", 20, 1, 50),
    "herbivore_gain_from_food": Slider("Herbivore Energy from Vegetation", 4, 1, 10),
    "mutation_rate": Slider("Behavior Mutation Rate", 0.05, 0.0, 0.5, 0.01),
    "evolution_method": {
        "type": "Select",
        "value": "genetic",
        "values": ["genetic", "reinforcement"],
        "label": "Evolution Method",
    },
    "enable_disasters": {
        "type": "Select",
        "value": True,
        "values": [True, False],
        "label": "Enable Natural Disasters",
    },
    "disaster_frequency": Slider("Disaster Frequency", 0.02, 0.0, 0.2, 0.01),
    "enable_weather": {
        "type": "Select",
        "value": True,
        "values": [True, False],
        "label": "Enable Weather Patterns",
    },
}


def post_process_space(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    
    if hasattr(model, "weather_system"):
        weather_state = model.weather_system.state
        if weather_state == "drought":
            weather_text = "Weather: Drought (Slow Growth)"
            color = "orange"
        elif weather_state == "rainy":
            weather_text = "Weather: Rainy (Fast Growth)"
            color = "blue"
        else:
            weather_text = "Weather: Normal"
            color = "black"
            
        ax.text(0.5, 1.02, weather_text, transform=ax.transAxes,
                ha='center', va='bottom', color=color, fontweight='bold')


def post_process_lines(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))


space_component = make_space_component(
    ecosystem_portrayal, draw_grid=False, post_process=post_process_space
)

population_plot = make_plot_component(
    {"Carnivores": "tab:red", "Herbivores": "tab:blue", "Vegetation": "tab:green"},
    post_process=lambda ax: (post_process_lines(ax), ax.set_title("Population Dynamics")),
)

energy_plot = make_plot_component(
    {"Average Carnivore Energy": "tab:red", "Average Herbivore Energy": "tab:blue"},
    post_process=lambda ax: (post_process_lines(ax), ax.set_title("Average Energy Levels")),
)

gene_plot = make_plot_component(
    {
        "Average Carnivore Foraging": "tab:red", 
        "Average Herbivore Foraging": "tab:blue",
        "Average Herbivore Risk Aversion": "tab:purple"
    },
    post_process=lambda ax: (post_process_lines(ax), ax.set_title("Evolving Traits")),
)

disaster_plot = make_plot_component(
    {
        "Fires": "tab:orange",
        "Tornadoes": "tab:gray",
    },
    post_process=lambda ax: (post_process_lines(ax), ax.set_title("Natural Disasters")),
)

simulator = ABMSimulator()
model = EcosystemModel(simulator=simulator, vegetation=True, enable_disasters=True, enable_weather=True)

page = SolaraViz(
    model,
    components=[space_component, population_plot, energy_plot, gene_plot, disaster_plot],
    model_params=model_params,
    name="Evolving Ecosystem Simulation with Disasters & Weather",
    simulator=simulator,
)
page
