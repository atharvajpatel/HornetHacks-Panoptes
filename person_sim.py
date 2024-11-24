# imports
import numpy as np


grid_dimensions = 200
heatmap_width = 64
base_heat = 37

class Person:
    def __init__(self, name, is_ally):
        """
        Initialize a person with a name and their status (enemy or ally).
        
        Args:
            name (str): The name of the person.
            is_ally (bool): True if the person is an ally, False if they are an enemy.
        """
        self.id = name
        self.ally = is_ally
        #self.heatmap = np.zeros((heatmap_width, heatmap_width))
        self.radio_signals = []
        self.x = np.random.uniform(0, grid_dimensions)
        self.y = np.random.uniform(0, grid_dimensions)
        #self.base_heat = 37 # human body temperature in celsius
        self.temperature = base_heat
        self.environmental_temp = 25


    def generate_radio_signal(self, duration=1.0, sampling_rate=1000):
        """
        Generate a simulated radio signal.
        
        Args:
            duration (float): Duration of the signal in seconds.
            sampling_rate (int): Number of samples per second.

        Returns:
            np.ndarray: The generated radio signal.
        """
        if self.ally:
            frequency = np.random.uniform(175, 200)
            amplitude = np.random.uniform(5, 10) 
            phase = np.random.uniform(np.pi, 2 * np.pi)
        else:
            # enemy
            frequency = np.random.uniform(150, 175)
            amplitude = np.random.uniform(1, 5) 
            phase = np.random.uniform(-1*np.pi, np.pi)

        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        #frequency = np.random.uniform(50, 150)  # Frequency in Hz
        #amplitude = np.random.uniform(1, 5)  # Signal amplitude
          # Phase in radians
        noise = np.random.normal(0, 0.5, size=t.shape)  # Add Gaussian noise

        # Generate the radio signal
        signal = amplitude * np.sin(2 * np.pi * frequency * t + phase) + noise
        
        # Store the signal
        self.radio_signals.append(signal)
        return self.radio_signals

    
    
    def emit_heat(self, activity_level=1.0, environment_temp=25):
        """
        Simulate the heat emitted by the person.

        Args:
            activity_level (float): Multiplier for physical activity (1.0 = normal, >1.0 = higher activity, <1.0 = lower).
            environment_temp (float): The ambient temperature in degrees Celsius.

        Returns:
            float: The heat emitted in degrees Celsius.
        """
        # Heat generation depends on base heat, activity level, and environmental temperature
        metabolic_heat = base_heat * activity_level
        environmental_effect = 0.1 * (base_heat - environment_temp) # probably cools the person down
        noise = np.random.normal(0, 0.5)  # Add some randomness to simulate real-world variation
        # Total heat emission
        total_heat = metabolic_heat + environmental_effect + noise
        self.temperature = total_heat
        
    
    def walk(self, environment_temp=25):
        """
        Simulate the person walking, increasing their activity level.

        Args:
            environment_temp (float): The ambient temperature in degrees Celsius.

        Returns:
            float: The heat emitted during walking.
        """
        intensity = 1.5  # Walking increases activity level
        x_step = np.random.uniform(1, 5)
        y_step = np.random.uniform(1, 5)
        self.x += x_step
        self.y += y_step
        self.emit_heat(activity_level=intensity, environment_temp=environment_temp)



    def rest(self, environment_temp=25):
        """
        Simulate the person resting, decreasing their activity level.

        Args:
            environment_temp (float): The ambient temperature in degrees Celsius.

        Returns:
            float: The heat emitted during resting.
        """
        intensity = 0.8  # Resting decreases activity level
        self.emit_heat(activity_level=intensity, environment_temp=environment_temp)


    
    def __str__(self):
        """String representation of the person."""
        status = "Ally" if self.ally else "Enemy"
        summary = (f"Person {self.id} is located at {self.x}, {self.y}. They are an {status}. The temperature is {self.temperature}")
        return summary
    
    

# Example usage
# Create instances of Person
alice = Person("Alice", is_ally=True)
alice.walk()
eve = Person("Eve", is_ally=False) # eve is an enemy
eve.rest()
signal_alice = alice.generate_radio_signal(duration=2.0)
signal_eve = eve.generate_radio_signal(duration=2.0)


# Generate radio signals

# Display details
print(alice)
print(eve)

# print(eve.radio_signals[0])
# print(len(eve.radio_signals[0]))





import matplotlib.pyplot as plt

def plot(person):
    signal = person.radio_signals[0]
    duration = 2
    t = np.linspace(0, duration, len(signal), endpoint=False)  # Time array for plotting. x axis
    #print(len(t))
    #print(len(person.radio_signals[0]))
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, label=person.id+str(person.ally))

    #plt.plot(t, person.radio_signals[0], label="Eve (Enemy)")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Signal Amplitude")
    plt.title("Simulated Radio Signals")
    plt.grid()
    plt.show()


plot(alice)
plot(eve)

# right now the plot only shows one by one. probably overlay them soon


# def create_enemy_signal():
#     fs = 1e6  # Sampling frequency (Hz)
#     T = 1e-3  # Pulse duration (s)
#     f_start = 77e9  # Start frequency (Hz)
#     f_end = 77.5e9  # End frequency (Hz)
#     t = np.linspace(0, T, int(fs*T))

#     # Generate chirp
#     chirp_signal = np.sin(2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) / T * t**2))

#     # Plot
#     plt.plot(t, chirp_signal)
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude")
#     plt.title("Chirp Signal")
#     plt.show()
#     pass

# def create_ally_signal():
#     fs = 1e6  # Sampling frequency (Hz)
#     T = 1e-3  # Pulse duration (s)
#     f_start = 79e9  # Start frequency (Hz)
#     f_end = 78.5e9  # End frequency (Hz)
#     t = np.linspace(0, T, int(fs*T))

#     # Generate chirp
#     chirp_signal = np.sin(2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) / T * t**2))

#     # Plot
#     plt.plot(t, chirp_signal)
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude")
#     plt.title("Chirp Signal")
#     plt.show()
#     pass

# # testing
# create_enemy_signal()
# create_ally_signal()

# def spawn_batch_enemies():
#     pass

# def spawn_group_of_allies():
#     pass
