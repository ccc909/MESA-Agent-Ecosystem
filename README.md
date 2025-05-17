# Ecosystem Simulation Project

An interactive ecosystem simulation with evolutionary behaviors, natural disasters, and environmental dynamics using the Mesa framework.

## Overview

This project simulates an ecosystem with:

- **Herbivores**: Consume vegetation and avoid predators
- **Carnivores**: Hunt herbivores for sustenance
- **Vegetation**: Grows and regrows when consumed
- **Natural Disasters**: Fires and tornadoes that disrupt the environment
- **Weather System**: Affects vegetation growth rates

The simulation includes evolving agent behaviors through:
- Genetic inheritance
- Reinforcement learning

## Features

- **Genetic Evolution**: Agents inherit and mutate traits affecting movement, foraging efficiency, metabolism, and reproduction
- **Reinforcement Learning**: Agents can learn optimal behaviors through Q-learning
- **Dynamic Environment**: Weather patterns and natural disasters create a changing landscape
- **Interactive Visualization**: Real-time visualization of agent populations and evolving traits

## Agent Types

### Animals
Animals inherit their traits through a genetic system and can learn optimal behaviors. Each animal has:

- **Herbivores**
  - Consume vegetation patches for energy
  - Avoid predators based on risk aversion genes
  - Make movement decisions balancing food needs and safety
  - Reproduction when energy reserves are high enough

- **Carnivores**
  - Hunt and consume herbivores for energy
  - Use foraging efficiency genes to determine hunting success
  - Track prey through the environment
  - Reproduction when energy reserves are high enough

### Plants
- **Vegetation**
  - Fixed agents that grow and regrow at rates affected by weather
  - Provide energy to herbivores when consumed
  - Regrow after a set time period determined by environmental conditions

### Disasters
- **Fire**
  - Spreads through the environment at a defined rate
  - Destroys vegetation in its path
  - Can cause damage to animals
  - Burns out after a set duration

- **Tornado**
  - Mobile disaster that moves through the environment
  - Can damage or kill animals in its path
  - Destroys vegetation
  - Dissipates after a set duration

### Genetic Traits
All animals possess genes that control:
- Movement speed and directedness
- Foraging efficiency and detection range
- Reproduction thresholds
- Metabolism rates
- Risk aversion (particularly important for herbivores)

## Requirements

- Python 3.x
- Mesa (Experimental features)
- Solara (for visualization)

## Installation

1. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```
   pip install mesa solara
   ```

## Usage

Copy the "ecosystem_project" folder somewhere.

Run the simulation !from its parent directory! with:

```
python -m solara run ecosystem_project.app
```

The web interface will allow you to:
- Adjust simulation parameters (population sizes, mutation rates, etc.)
- Toggle features (weather, disasters)
- Observe population dynamics and evolutionary trends in real-time

## Project Structure

- `agents.py`: Defines all agent types (animals, vegetation, disasters)
- `model.py`: Contains the main ecosystem model
- `reinforcement_learning.py`: Q-learning implementation
- `app.py`: Visualization and UI components

## Simulation Parameters

- **Grid Size**: Controls the environment dimensions
- **Population Size**: Initial number of herbivores and carnivores
- **Reproduction Rates**: Probability of reproduction per step
- **Energy Values**: Energy gained from food and initial energy
- **Mutation Rate**: Controls how quickly genes evolve
- **Evolution Method**: Choose between genetic or reinforcement learning approaches
- **Weather & Disasters**: Toggle and adjust environmental effects
