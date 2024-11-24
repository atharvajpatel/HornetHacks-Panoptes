# imports
import numpy as np

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
        self.radio_signals = []

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
        return signal

    def status(self):
        """Return the status of the person (Enemy or Ally)."""
        return "Ally" if self.ally else "Enemy"

    def __str__(self):
        """String representation of the person."""
        return f"Person(name={self.id}, status={self.status()})"

# Example usage
if __name__ == "__main__":
    # Create instances of Person
    alice = Person("Alice", is_ally=True)
    eve = Person("Eve", is_ally=False)

    # Generate radio signals
    signal_alice = alice.generate_radio_signal(duration=2.0)
    signal_eve = eve.generate_radio_signal(duration=2.0)

    # Display details
    print(alice)
    print(eve)

    # Optionally plot the signals
    import matplotlib.pyplot as plt

    t = np.linspace(0, 2.0, 2000, endpoint=False)  # Time array for plotting
    plt.figure(figsize=(10, 6))

    plt.plot(t, signal_alice, label="Alice (Ally)")
    plt.plot(t, signal_eve, label="Bob (Enemy)")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Signal Amplitude")
    plt.title("Simulated Radio Signals")
    plt.grid()
    plt.show()




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
