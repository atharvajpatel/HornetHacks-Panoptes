import plotly.graph_objects as go
import numpy as np



def draw():
    # Create spheres (inner Earth and outer celestial sphere)
    phi = np.linspace(0, 2*np.pi, 100)
    theta = np.linspace(-np.pi/2, np.pi/2, 100)
    phi, theta = np.meshgrid(phi, theta)

    # Inner sphere (Earth)
    x_earth = np.cos(theta) * np.cos(phi)
    y_earth = np.cos(theta) * np.sin(phi)
    z_earth = np.sin(theta)

    # Outer sphere (1.5x radius for celestial sphere)
    x_celestial = 1.5 * x_earth
    y_celestial = 1.5 * y_earth
    z_celestial = 1.5 * z_earth

    # Generate random background stars
    num_bg_stars = 1000
    np.random.seed(42)  # For reproducibility
    bg_stars = []
    for _ in range(num_bg_stars):
        # Generate random spherical coordinates
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))
        r = 2  # Radius larger than celestial sphere
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        brightness = np.random.uniform(0.2, 1.0)
        bg_stars.append((x, y, z, brightness))

    # Star data
    star_coords = [
        (108.63333333333334, 147.36666666666667, 247),
        (608.4137931034483, 358.7241379310345, 247),
        (308.5, 52.5, 245),
        (364.4117647058824, 278.11764705882354, 245),
        (349.0, 314.5, 245),
        (434.26666666666665, 531.9333333333333, 240),
        (375.0, 239.5, 237),
        (438.0, 375.5, 228),
        (446.75, 388.25, 224),
        (597.0, 256.5, 223),
        (753.875, 553.125, 214),
        (453.6666666666667, 259.3333333333333, 207),
        (189.5, 5.5, 206),
        (371.75, 322.0, 203),
        (567.5, 338.5, 201),
        (760.3333333333334, 86.33333333333333, 192),
        (123.4, 504.8, 191),
        (840.3333333333334, 501.6666666666667, 184)
    ]

    def normalize_to_celestial_sphere(coords):
        x, y, brightness = coords
        x_norm = (x - np.min([c[0] for c in star_coords])) / (np.max([c[0] for c in star_coords]) - np.min([c[0] for c in star_coords])) * 2 - 1
        y_norm = (y - np.min([c[1] for c in star_coords])) / (np.max([c[1] for c in star_coords]) - np.min([c[1] for c in star_coords])) * 2 - 1
        
        r = np.sqrt(x_norm**2 + y_norm**2)
        if r > 1:
            x_norm /= r
            y_norm /= r
        
        z_norm = np.sqrt(1 - x_norm**2 - y_norm**2)
        # Scale to celestial sphere radius
        return 1.5 * x_norm, 1.5 * y_norm, 1.5 * z_norm, brightness

    # Create figure
    fig = go.Figure()

    # Add Earth surface
    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        surfacecolor=np.ones_like(x_earth) * 0.1,
        colorscale=[
            [0, 'rgb(5, 5, 20)'],      # Deep night blue
            [0.5, 'rgb(10, 10, 40)'],  # Midnight blue
            [1, 'rgb(20, 20, 60)']     # Dark blue with slight variation
        ],
        showscale=False,
        opacity=1,
        lighting=dict(
            ambient=0.3,
            diffuse=0.8,
            fresnel=0.2,
            specular=0.5,
            roughness=0.5
        )
    ))

    # Add transparent celestial sphere
    fig.add_trace(go.Surface(
        x=x_celestial, y=y_celestial, z=z_celestial,
        surfacecolor=np.ones_like(x_celestial) * 0.1,
        colorscale=[[0, 'rgb(20, 20, 40)'], [1, 'rgb(20, 20, 40)']],
        showscale=False,
        opacity=0.1,
        hoverinfo='skip'
    ))

    # Add background ambient stars
    for star in bg_stars:
        x, y, z, brightness = star
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(
                size=brightness*0.5,
                color='rgba(255, 255, 255, {})'.format(brightness * 0.2),
                symbol='circle'
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add main stars on celestial sphere
    for star in star_coords:
        x_star, y_star, z_star, brightness = normalize_to_celestial_sphere(star)
        marker_size = (brightness - min([s[2] for s in star_coords])) / (max([s[2] for s in star_coords]) - min([s[2] for s in star_coords])) * 6 + 1
        
        fig.add_trace(go.Scatter3d(
            x=[x_star], y=[y_star], z=[z_star],
            mode='markers',
            marker=dict(
                size=marker_size,
                color='white',
                symbol='circle',
                opacity=0.9
            ),
            showlegend=False
        ))

    # Add observer position
    observer_coords = (-0.2262, 0.0541, 0.9726)
    fig.add_trace(go.Scatter3d(
        x=[observer_coords[0]],
        y=[observer_coords[1]],
        z=[observer_coords[2]],
        mode='markers',
        marker=dict(
            size=8,
            color='red',
            symbol='circle'
        ),
        name='Observer'
    ))

    # Update layout with enhanced lighting and atmosphere
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2, y=2, z=1.5)
            ),
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            bgcolor='rgb(0,0,0)'
        ),
        paper_bgcolor='black',
        showlegend=True,
        title=dict(
            text='Earth at Night with Celestial Sphere',
            font=dict(color='white', size=24),
            x=0.5,
            y=0.95
        ),
        margin=dict(t=50, b=0, l=0, r=0)
    )

    fig.show()