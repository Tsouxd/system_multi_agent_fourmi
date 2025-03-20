# Importations
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from mesa.datacollection import DataCollector
import logging
from mesa.visualization.modules import TextElement
from mesa.visualization.UserParam import Slider


# Configuration du logger
logging.basicConfig(level=logging.DEBUG)

# Classe Maison
class Home(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.food_storage = 0  # Stock de nourriture dans la maison

class Pheromone(Agent):
    def __init__(self, unique_id, model, position, food_memory, lifespan=40):
        super().__init__(unique_id, model)
        self.food_memory = food_memory
        self.position = position
        self.age = 0
        self.lifespan = lifespan

        if self.position:
            # Place l'agent uniquement si la position est valide
            self.model.grid.place_agent(self, self.position)
        else:
            raise ValueError("Position cannot be None when creating a Pheromone")

    def step(self):
        self.age += 1
        if self.age > self.lifespan:
            if self.position:
                self.model.grid.remove_agent(self)
                self.position = None  # Réinitialisez manuellement self.position

class QueenAnt(Agent):
    def __init__(self, unique_id, model, home, reproduction_rate=10, move_rate=30, food_threshold=5, population_limit=1000):
        super().__init__(unique_id, model)
        self.reproduction_rate = reproduction_rate
        self.move_rate = move_rate
        self.food_threshold = food_threshold  # Seuil de nourriture requis pour la reproduction
        self.population_limit = population_limit  # Limite de population avant d'arrêter la reproduction
        self.age = 0
        self.home = home
        self.target_home = home
        self.pos = home.pos
        self.model.grid.place_agent(self, self.pos)
        self.ants_created = 0

    def step(self):
        logging.debug(f"{self.__class__.__name__} {self.unique_id} étape")
        self.age += 1

        # Vérifie si la maison a assez de nourriture pour la reproduction
        if self.home.food_storage >= self.food_threshold:
            self.create_ant()
            self.home.food_storage -= self.food_threshold  # Réduit la nourriture utilisée pour la reproduction

        if self.age % self.move_rate == 0:
            self.move_to_new_home()

    def create_ant(self):
        """Crée une nouvelle fourmi."""
        current_population = len([ant for ant in self.model.schedule.agents if isinstance(ant, AntAgent)])
        
        if current_population < self.population_limit:  
            new_id = self.model.next_id()
            
            # Vérifie que l'ID n'est pas déjà utilisé
            existing_ids = {agent.unique_id for agent in self.model.schedule.agents}
            if new_id in existing_ids:
                print(f"⚠️ Problème d'ID: {new_id} existe déjà!")
                return  # Empêche l'ajout d'un doublon
            
            # Attribuer un rôle de manière aléatoire ou alternée
            roles = ["eclaireuse", "ouvriere", "soldat"]
            role = self.random.choice(roles)
            
            new_ant = AntAgent(new_id, self.model, self.home, role)
            self.model.schedule.add(new_ant)
            self.model.grid.place_agent(new_ant, self.home.pos)
            self.ants_created += 1
            self.model.total_ants_born += 1
            print(f"✅ Une nouvelle fourmi est née avec ID: {new_id} et rôle: {role}. Total: {self.ants_created}")
        else:
            print(f"🚫 Limite atteinte ({self.population_limit} fourmis).")
            
    def move_to_new_home(self):
        """Déplace la reine vers une autre maison."""
        available_homes = [home for home in self.model.homes if home != self.home]

        if available_homes:
            self.target_home = self.random.choice(available_homes)  # Choisir une nouvelle maison
            self.model.grid.move_agent(self, self.target_home.pos)
            self.home = self.target_home  # Mise à jour de la maison actuelle


# Classe Fourmi
class AntAgent(Agent):
    def __init__(self, unique_id, model, home, role):
        super().__init__(unique_id, model)
        self.role = role  # "eclaireuse", "ouvriere", "soldat"
        self.state = "recherche_nourriture"
        self.food_memory = None
        self.alive = True
        self.home = home  # Maison associée à la fourmi
        self.hunger = 50  # Durée de vie initiale

    def step(self):
        if not self.alive:
            return

        self.hunger -= 1
        if self.hunger <= 0:
            self.alive = False
            self.model.grid.remove_agent(self)
            return
        
        if self.role == "eclaireuse":
            self.eclaireuse_step()
        elif self.role == "ouvriere":
            self.ouvriere_step()
        elif self.role == "soldat":
            self.soldat_step()
            
    def eclaireuse_step(self):
        if self.state == "recherche_nourriture":
            self.se_diriger_vers_nourriture()
        elif self.state == "retour_maison":
            self.retourner_maison()
            
    def ouvriere_step(self):
        if self.state == "recherche_nourriture":
            self.suivre_pheromones()
        elif self.state == "retour_maison":
            self.retourner_maison()

    def soldat_step(self):
        self.patrouiller()
        self.proteger_ouvrieres()

    def se_diriger_vers_nourriture(self):
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        valid_moves = [
            pos for pos in neighborhood
            if self.is_position_valid(pos)
        ]
        if valid_moves:
            food_positions = [
                pos for pos in valid_moves
                if any(isinstance(obj, FoodSource) for obj in self.model.grid.get_cell_list_contents([pos]))
            ]
            if food_positions:
                new_position = self.random.choice(food_positions)
                self.state = "retour_maison"
                self.food_memory = new_position
                food_source = next(
                    obj for obj in self.model.grid.get_cell_list_contents([new_position])
                    if isinstance(obj, FoodSource)
                )
                food_source.quantity -= 1
                if food_source.quantity <= 0:
                    self.model.grid.remove_agent(food_source)
            else:
                new_position = self.random.choice(valid_moves)
            self.model.grid.move_agent(self, new_position)

    def suivre_pheromones(self):
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        valid_moves = [
            pos for pos in neighborhood
            if self.is_position_valid(pos)
        ]
        if valid_moves:
            pheromone_positions = [
                pos for pos in valid_moves
                if any(isinstance(obj, Pheromone) for obj in self.model.grid.get_cell_list_contents([pos]))
            ]
            if pheromone_positions:
                new_position = self.random.choice(pheromone_positions)
                self.model.grid.move_agent(self, new_position)
            else:
                new_position = self.random.choice(valid_moves)
                self.model.grid.move_agent(self, new_position)

    def patrouiller(self):
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        valid_moves = [
            pos for pos in neighborhood
            if self.is_position_valid(pos)
        ]
        if valid_moves:
            new_position = self.random.choice(valid_moves)
            self.model.grid.move_agent(self, new_position)

    def proteger_ouvrieres(self):
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        for pos in neighborhood:
            cell_contents = self.model.grid.get_cell_list_contents([pos])
            for obj in cell_contents:
                if isinstance(obj, Predator):
                    self.combattre_predateur(obj)

    def combattre_predateur(self, predator):
        if predator.alive:
            predator.alive = False
            self.model.grid.remove_agent(predator)

    def retourner_maison(self):
        if self.pos == self.home.pos:
            self.home.food_storage += 1
            self.hunger = 50
            self.state = "recherche_nourriture"
        else:
            neighborhood = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=False
            )
            valid_moves = [
                pos for pos in neighborhood
                if self.is_position_valid(pos)
            ]
            if valid_moves:
                best_move = min(
                    valid_moves, key=lambda pos: self.distance_to_home(pos)
                )
                self.model.grid.move_agent(self, best_move)

                # Créer une phéromone avec un ID unique
                pheromone_id = self.model.next_id() + 10000  
                pheromone = Pheromone(pheromone_id, self.model, self.pos, self.food_memory)
                self.model.grid.place_agent(pheromone, best_move)
                self.model.schedule.add(pheromone)

    def distance_to_home(self, pos):
        return abs(pos[0] - self.home.pos[0]) + abs(pos[1] - self.home.pos[1])

    def is_position_valid(self, pos):
        cell_contents = self.model.grid.get_cell_list_contents([pos])
        return not any(isinstance(obj, Obstacle) for obj in cell_contents)


# Classe Obstacle
class Obstacle(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

# Classe Source de Nourriture
class FoodSource(Agent):
    def __init__(self, unique_id, model, quantity):
        super().__init__(unique_id, model)
        self.quantity = quantity  # Quantité de nourriture disponible

# Classe Prédateur
class Predator(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.hunger = 100  # Durée de vie initiale
        self.alive = True

    def step(self):
        if not self.alive:
            return

        self.hunger -= 1  # Réduction de la faim à chaque étape
        if self.hunger <= 0:
            self.alive = False
            self.model.grid.remove_agent(self)
            return

        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        valid_moves = [
            pos for pos in neighborhood
            if self.is_position_valid(pos)
        ]
        if valid_moves:
            new_position = self.random.choice(valid_moves)
            self.model.grid.move_agent(self, new_position)

            # Vérifier les fourmis sur la nouvelle position
            cell_contents = self.model.grid.get_cell_list_contents([new_position])
            for obj in cell_contents:
                if isinstance(obj, AntAgent) and obj.alive:
                    obj.alive = False
                    self.model.grid.remove_agent(obj)
                    self.hunger = 100  # Recharge la faim après avoir mangé une fourmi

    def is_position_valid(self, pos):
        cell_contents = self.model.grid.get_cell_list_contents([pos])
        return not any(isinstance(obj, Obstacle) for obj in cell_contents)  # Vérifie s'il y a un obstacle

    

# Classe Modèle
# Mise à jour de la classe AntModel pour inclure la collecte des données
class AntModel(Model):
    def __init__(self, width, height, nb_fourmis, nb_obstacles, nb_maisons, nb_predateurs, nb_reines, evaporation_rate=0.95):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        self.evaporation_rate = evaporation_rate
        self.current_id = 0  # Initialisation du compteur d'ID
        self.nb_reines = nb_reines  # Stocker le nombre de reines
        self.all_ants = []
        self.total_ants_born = 0 
        
        # Initialisation de la collecte des données
        self.datacollector = DataCollector(
            model_reporters={
                "Fourmis Vivantes": self.count_alive_ants,  # Utilisation de la méthode de classe
                
                # Collecte des stocks de nourriture des maisons
                **{f"Maison {i}": (lambda m, i=i: m.homes[i].food_storage if i < len(m.homes) else 0)
                   for i in range(nb_maisons)},
            
                # Autres données globales
                "Nombre de Fourmis": lambda m: len([a for a in m.schedule.agents if isinstance(a, AntAgent)]),
                "Nombre de Fourmis Mortes": lambda m: len([a for a in m.schedule.agents if isinstance(a, AntAgent) and not a.alive]),
                "Nombre de Prédateurs": lambda m: len([a for a in m.schedule.agents if isinstance(a, Predator)]),
                "Nombre de Prédateurs Mortes": lambda m: len([a for a in m.schedule.agents if isinstance(a, Predator) and not a.alive]),
                "Nourriture Apportée": lambda m: sum(home.food_storage for home in m.homes),
                "Nourriture Disponible": lambda m: sum(food.quantity for food in m.schedule.agents if isinstance(food, FoodSource)),
            }
        )
        
        # Clear existing agents from grid and schedule
        for agent in list(self.schedule.agents):  # Iterate over a copy
            self.grid.remove_agent(agent)
            self.schedule.remove(agent)

        # Ajouter les maisons
        self.homes = []
        for i in range(nb_maisons):
            home = Home(i, self)
            self.schedule.add(home)
            x, y = self.find_empty_cell()
            self.grid.place_agent(home, (x, y))
            self.homes.append(home)

        # Ajouter les fourmis avec des rôles
        self.ants = []
        for i in range(nb_fourmis):
            home = self.random.choice(self.homes)
            roles = ["eclaireuse", "ouvriere", "soldat"]
            role = self.random.choice(roles)
            ant = AntAgent(i + nb_maisons, self, home, role)
            self.schedule.add(ant)
            self.grid.place_agent(ant, home.pos)
            self.ants.append(ant)

        # Ajouter les reines
        self.queens = []
        for i in range(nb_reines):
            if i < len(self.homes):  # Associer chaque reine à une maison si possible
                home = self.homes[i]
                queen = QueenAnt(i, self, home=home)
                if not any(agent.unique_id == queen.unique_id for agent in self.schedule.agents):
                    self.schedule.add(queen)
                self.grid.place_agent(queen, home.pos)
                self.queens.append(queen)

        # Ajouter une reine dans la première maison
        self.queen = QueenAnt(self.next_id(), self, home=self.homes[0])
        if self.queen.unique_id not in [agent.unique_id for agent in self.schedule.agents]:
            self.schedule.add(self.queen)
            self.grid.place_agent(self.queen, self.homes[0].pos)

        # Ajouter les obstacles
        for i in range(nb_obstacles):
            obstacle = Obstacle(i + nb_fourmis + nb_maisons, self)
            self.schedule.add(obstacle)
            x, y = self.find_empty_cell()
            self.grid.place_agent(obstacle, (x, y))

        # Ajouter les sources de nourriture
        for i in range(10):  # Par exemple, 10 sources de nourriture
            food = FoodSource(i + nb_fourmis + nb_obstacles + nb_maisons, self, quantity=100)
            self.schedule.add(food)
            x, y = self.find_empty_cell()
            self.grid.place_agent(food, (x, y))

        # Ajouter les prédateurs
        for i in range(nb_predateurs):
            predator = Predator(i + nb_fourmis + nb_obstacles + nb_maisons + 10, self)
            self.schedule.add(predator)
            x, y = self.find_empty_cell()
            self.grid.place_agent(predator, (x, y))

    def update_parameters(self, nb_fourmis, nb_obstacles, nb_maisons, nb_predateurs):
        """Updates model parameters and re-initializes agents."""
        self.nb_fourmis = nb_fourmis
        self.nb_obstacles = nb_obstacles
        self.nb_maisons = nb_maisons
        self.nb_predateurs = nb_predateurs

    def find_empty_cell(self):
        while True:
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            if self.grid.is_cell_empty((x, y)):
                return x, y

    def step(self):
        # Faire avancer tous les agents
        self.schedule.step()

        # Collecte des données pour chaque étape
        self.datacollector.collect(self)

        # Vérifier si toutes les fourmis sont mortes (en utilisant count_alive_ants)
        if self.count_alive_ants() == 0:
            self.running = False
            print("Toutes les fourmis sont mortes. Simulation terminée.")
            
        # Créer des fourmis via la reine (si applicable)
        for queen in self.queens:
            queen.step()  # Appeler la méthode step de la reine, qui créera des fourmis directement

    def next_id(self):
        new_id = self.current_id
        self.current_id += 1
        return new_id

    def count_alive_ants(self):
        """Compte le nombre de fourmis vivantes."""
        return sum(1 for agent in self.schedule.agents if isinstance(agent, AntAgent) and agent.alive)

# Visualisation
def agent_portrayal(agent):
    if isinstance(agent, AntAgent):
        if agent.alive:
            if agent.role == "eclaireuse":
                return {
                    "Shape": "/home/tsou/projet python/Fourmi/System_Multi_Agent/resources/fourmi_eclaireuse.png",
                    "Layer": 1,
                    "scale": 0.6,
                }
            elif agent.role == "ouvriere":
                return {
                    "Shape": "/home/tsou/projet python/Fourmi/System_Multi_Agent/resources/fourmi_ouvrier.png",
                    "Layer": 1,
                    "scale": 0.6,
                }
            elif agent.role == "soldat":
                return {
                    "Shape": "/home/tsou/projet python/Fourmi/System_Multi_Agent/resources/fourmi_guerrier.png",
                    "Layer": 1,
                    "scale": 0.6,
                }
        else:
            return {
                "Shape": "/home/tsou/projet python/Fourmi/System_Multi_Agent/resources/resources/fourmi_mort.png",
                "Layer": 1,
                "scale": 0.6,
            }
    elif isinstance(agent, Predator):  # Ajout du prédateur
        return {
            "Shape": "/home/tsou/projet python/Fourmi/System_Multi_Agent/resources/predator.png",
            "Layer": 1,
            "scale": 0.8,
        }
    elif isinstance(agent, Home):
        return {
            "Shape": "/home/tsou/projet python/Fourmi/System_Multi_Agent/resources/maison.png",
            "Layer": 1,
            "scale": 1,
        }
    elif isinstance(agent, Obstacle):
        return {
            "Shape": "/home/tsou/projet python/Fourmi/System_Multi_Agent/resources/obstacle.png",
            "Layer": 1,
            "scale": 0.8,
        }
    elif isinstance(agent, FoodSource):
        return {
            "Shape": "/home/tsou/projet python/Fourmi/System_Multi_Agent/resources/food1.png",
            "Layer": 1,
            "scale": 0.6,
        }
    elif isinstance(agent, QueenAnt):
        return {"Shape": "/home/tsou/projet python/Fourmi/System_Multi_Agent/resources/reine_fourmi.png",
                "Filled": "true",
                "Color": "purple",  # Couleur pour la Reine
                "Layer": 1,
                "r": 0.8}  # Taille plus grande pour la Reine
    elif isinstance(agent, Pheromone):
        # Intensité de couleur basée sur la durée de vie restante
        intensity = int((1 - agent.age / agent.lifespan) * 255)
        return {
            "Color": f"rgb({intensity}, {intensity // 2}, 0)",  # Dégradé de orange
            "Shape": "circle",
            "Layer": 1,
            "r": 0.3,
        }
    return {}

class AliveAntsText(TextElement):
    def render(self, model):
        alive_ants = model.count_alive_ants()
        return f"Fourmis vivantes : {alive_ants} | Total des fourmis nées: {model.total_ants_born}" # Display the correct count

class StatisticsElement(TextElement):
    def render(self, model):
        return f"Total des Fourmis: {len([a for a in model.schedule.agents if isinstance(a, AntAgent)])} | " \
               f"Fourmis Mortes: {len([a for a in model.schedule.agents if isinstance(a, AntAgent) and not a.alive])} | " \
               f"Prédateurs: {len([a for a in model.schedule.agents if isinstance(a, Predator)])} | " \
               f"Prédateurs Mortes: {len([a for a in model.schedule.agents if isinstance(a, Predator) and not a.alive])} | " \
               f"Nourriture Apportée: {sum(home.food_storage for home in model.homes)} | " \
               f"Nourriture Disponible: {sum(food.quantity for food in model.schedule.agents if isinstance(food, FoodSource))}"

# Liste des couleurs définies manuellement pour chaque maison
# Vous pouvez modifier ces couleurs comme vous le souhaitez
manual_colors = [
    "#FF0000",  # Maison 0 - Rouge
    "#00FF00",  # Maison 1 - Vert
    #"#0000FF",  # Maison 2 - Bleu
    #"#FFFF00",  # Maison 3 - Jaune
    #"#FF00FF",  # Maison 4 - Magenta
    #"#00FFFF",  # Maison 5 - Cyan
    # Ajoutez d'autres couleurs si nécessaire
]

# Ajout du ChartModule à la visualisation
chart = ChartModule(
    [{"Label": f"Maison {i}", "Color": manual_colors[i]} for i in range(len(manual_colors))],  # Utilisation des couleurs définies manuellement
    data_collector_name="datacollector"
)

# Ajout du ChartModule à la visualisation
chart_stats = ChartModule(
    [
        {"Label": "Nombre de Fourmis", "Color": "#FF5733"},
        {"Label": "Nombre de Fourmis Mortes", "Color": "#8E44AD"},
        {"Label": "Nombre de Prédateurs", "Color": "#C0392B"},
        {"Label": "Nombre de Prédateurs Mortes", "Color": "#2980B9"},
        {"Label": "Nourriture Apportée", "Color": "#27AE60"},
        {"Label": "Nourriture Disponible", "Color": "#2980B9"},
    ],
    data_collector_name="datacollector"
)


grid = CanvasGrid(agent_portrayal, 40, 40, 800, 800)

# Create sliders for parameters
fourmis_slider = Slider("Nombre de Fourmis", 30, 1, 1000, 1)
obstacles_slider = Slider("Nombre d'Obstacles", 10, 0, 200, 1)
maisons_slider = Slider("Nombre de Maisons", 2, 1, 5, 1)  # Example range
predateurs_slider = Slider("Nombre de Prédateurs", 2, 0, 100, 1)

server = ModularServer(
    AntModel,
    [grid, StatisticsElement(), AliveAntsText(), chart, chart_stats],
    "Simulation de Fourmis",
    {
        "width": 40,
        "height": 40,
        "nb_fourmis": fourmis_slider,
        "nb_obstacles": obstacles_slider,
        "nb_maisons": maisons_slider,
        "nb_predateurs": predateurs_slider,
        "nb_reines": 2,
        "evaporation_rate": 0.95
    }
)

def update_model_params(model):
    model.update_parameters(
        model.nb_fourmis.value,  # Access the slider's value
        model.nb_obstacles.value,
        model.nb_maisons.value,
        model.nb_predateurs.value
    )

if __name__ == "__main__":
    server.launch()