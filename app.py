import streamlit as st
import numpy as np
from matplotlib.figure import Figure
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm, jv
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Constants
h = 6.62607015e-34  # Planck constant (J⋅s)
hbar = 1.054571817e-34  # reduced Planck constant
m0 = 9.1093837015e-31  # electron mass (kg)
eV_to_J = 1.602176634e-19

def main():
    st.set_page_config(
        page_title="Particle in a Box - Quantum Mechanics Explorer",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Particle in a Box - Quantum Mechanics Explorer")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Model Controls")
        
        # Model selection
        model = st.radio(
            "Box Geometry:",
            ["1D", "3D", "Spherical"],
            index=0
        )
        
        # Mass scaling
        mass_scale = st.slider(
            "Mass Scaling (×m₀):",
            min_value=0.01,
            max_value=2.0,
            value=1.0,
            step=0.01,
            format="%.2f"
        )
        
        # Model-specific parameters
        if model == "1D":
            st.subheader("1D Box Parameters")
            n = st.number_input("Quantum number n:", min_value=1, max_value=20, value=1, step=1)
            L = st.number_input("Box length L (nm):", min_value=0.1, max_value=10.0, value=1.0, step=0.1, format="%.1f")
            
        elif model == "3D":
            st.subheader("3D Box Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                nx = st.number_input("nx:", min_value=1, max_value=10, value=1, step=1)
                Lx = st.number_input("Lx (nm):", min_value=0.1, max_value=10.0, value=1.0, step=0.1, format="%.1f")
            with col2:
                ny = st.number_input("ny:", min_value=1, max_value=10, value=1, step=1)
                Ly = st.number_input("Ly (nm):", min_value=0.1, max_value=10.0, value=1.0, step=0.1, format="%.1f")
            with col3:
                nz = st.number_input("nz:", min_value=1, max_value=10, value=1, step=1)
                Lz = st.number_input("Lz (nm):", min_value=0.1, max_value=10.0, value=1.0, step=0.1, format="%.1f")
                
        elif model == "Spherical":
            st.subheader("Spherical Box Parameters")
            n = st.number_input("n (radial):", min_value=1, max_value=10, value=1, step=1)
            l = st.number_input("l (angular):", min_value=0, max_value=10, value=0, step=1)
            m = st.number_input("m (magnetic):", min_value=-10, max_value=10, value=0, step=1)
            R = st.number_input("Sphere radius R (nm):", min_value=0.1, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    
    # Main content area
    mass = m0 * mass_scale
    
    # Calculate and display energy
    if model == "1D":
        E_joules = (n**2 * h**2) / (8 * mass * (L * 1e-9)**2)
        E_eV = E_joules / eV_to_J
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Energy", f"{E_eV:.3f} eV")
        with col2:
            st.metric("Energy", f"{E_joules:.2e} J")
            
        # Plot 1D wavefunction
        plot_1d_wavefunction(n, L, E_eV)
        
    elif model == "3D":
        E_joules = (h**2 / (8 * mass)) * (
            (nx / (Lx * 1e-9))**2 + (ny / (Ly * 1e-9))**2 + (nz / (Lz * 1e-9))**2
        )
        E_eV = E_joules / eV_to_J
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Energy", f"{E_eV:.3f} eV")
        with col2:
            st.metric("Energy", f"{E_joules:.2e} J")
            
        # Plot 3D wavefunction
        plot_3d_wavefunction(nx, ny, nz, Lx, Ly, Lz, E_eV)
        
    elif model == "Spherical":
        try:
            E_joules = energy_spherical_box(n, l, R * 1e-9, mass)
            E_eV = E_joules / eV_to_J
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Energy", f"{E_eV:.3f} eV")
            with col2:
                st.metric("Energy", f"{E_joules:.2e} J")
        except:
            st.error("Invalid quantum state combination")
            E_eV = float('inf')
            
        # Plot spherical wavefunction
        plot_spherical_wavefunction(n, l, m, R, E_eV)

def plot_1d_wavefunction(n, L, E_eV):
    """Plot 1D wavefunction and probability density"""
    st.subheader("Wavefunction Visualization")
    
    # Generate wavefunction
    L_nm = L  # Already in nm
    x = np.linspace(0, L_nm, 1000)
    norm = np.sqrt(2 / L_nm)
    psi = norm * np.sin(n * np.pi * x / L_nm)
    prob_density = np.abs(psi)**2
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f'1D Particle in Box: n={n}, L={L:.1f} nm - Energy = {E_eV:.3f} eV',
            f'Probability Density: n={n} ({n-1} internal nodes)'
        ],
        vertical_spacing=0.1
    )
    
    # Wavefunction plot
    fig.add_trace(
        go.Scatter(x=x, y=psi, mode='lines', name=f'ψ_{n}(x)', 
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Fill areas for positive and negative parts
    fig.add_trace(
        go.Scatter(x=x, y=psi, mode='lines', fill='tonexty', 
                  fillcolor='rgba(0,0,255,0.3)', line=dict(color='blue', width=2),
                  showlegend=False),
        row=1, col=1
    )
    
    # Mark nodes
    if n > 1:
        node_positions = [k * L / n for k in range(1, n)]
        for node_x in node_positions:
            fig.add_vline(x=node_x, line_dash="dot", line_color="black", 
                         opacity=0.5, row=1, col=1)
    
    # Probability density plot
    fig.add_trace(
        go.Scatter(x=x, y=prob_density, mode='lines', name=f'|ψ_{n}(x)|²',
                  line=dict(color='red', width=2), fill='tonexty',
                  fillcolor='rgba(255,0,0,0.3)'),
        row=2, col=1
    )
    
    # Mark nodes on probability density
    if n > 1:
        for node_x in node_positions:
            fig.add_vline(x=node_x, line_dash="dot", line_color="black", 
                         opacity=0.5, row=2, col=1)
    
    # Update layout
    fig.update_xaxes(title_text="Position (nm)", row=2, col=1)
    fig.update_yaxes(title_text="Wavefunction ψ(x)", row=1, col=1)
    fig.update_yaxes(title_text="Probability Density |ψ(x)|²", row=2, col=1)
    
    fig.update_layout(height=800, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_3d_wavefunction(nx, ny, nz, Lx, Ly, Lz, E_eV):
    """Plot 3D wavefunction"""
    st.subheader("3D Wavefunction Visualization")
    
    # Create coordinate grids
    n_points = 30
    x = np.linspace(0, Lx, n_points)
    y = np.linspace(0, Ly, n_points)
    z = np.linspace(0, Lz, n_points)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Calculate wavefunction
    norm = np.sqrt(8 / (Lx * Ly * Lz * 1e-27))  # Convert nm^3 to m^3
    psi = norm * np.sin(nx * np.pi * X / Lx) * \
          np.sin(ny * np.pi * Y / Ly) * \
          np.sin(nz * np.pi * Z / Lz)
    
    # Create isosurface plot
    fig = go.Figure()
    
    # Define isosurface levels
    max_val = np.max(np.abs(psi))
    level_positive = 0.3 * max_val
    level_negative = -0.3 * max_val
    
    # Add positive isosurface
    fig.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=psi.flatten(),
        isomin=level_positive,
        isomax=max_val,
        surface_count=1,
        colorscale='Blues',
        opacity=0.7,
        name='Positive regions'
    ))
    
    # Add negative isosurface if it exists
    if np.min(psi) < level_negative:
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=psi.flatten(),
            isomin=np.min(psi),
            isomax=level_negative,
            surface_count=1,
            colorscale='Reds',
            opacity=0.7,
            name='Negative regions'
        ))
    
    # Add box outline
    add_box_outline(fig, Lx, Ly, Lz)
    
    fig.update_layout(
        title=f'Ψ_{nx},{ny},{nz}(x,y,z) - 3D Particle in a Box<br>'
              f'Quantum numbers: nx={nx}, ny={ny}, nz={nz}<br>'
              f'Energy: {E_eV:.2f} eV',
        scene=dict(
            xaxis_title='x (nm)',
            yaxis_title='y (nm)',
            zaxis_title='z (nm)',
            aspectmode='data'
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_spherical_wavefunction(n, l, m, R, E_eV):
    """Plot spherical wavefunction"""
    st.subheader("Spherical Wavefunction Visualization")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'Radial Wavefunction: n={n}, l={l}',
            f'Angular Probability |Y_l^m|²: l={l}, m={m}',
            f'3D Angular Shape Y_{l}^{m}',
            'Energy Level Information'
        ],
        specs=[[{"type": "scatter"}, {"type": "scatterpolar"}],
               [{"type": "scatter3d"}, {"type": "table"}]]
    )
    
    # Radial wavefunction
    r = np.linspace(0, R, 1000)
    try:
        R_nl = particle_in_spherical_box_radial(r * 1e-9, n, l, R * 1e-9)
        fig.add_trace(
            go.Scatter(x=r, y=R_nl, mode='lines', name=f'R_{n}{l}(r)',
                      line=dict(color='blue', width=2), fill='tonexty',
                      fillcolor='rgba(0,0,255,0.3)'),
            row=1, col=1
        )
        fig.update_xaxes(title_text="r (nm)", row=1, col=1)
        fig.update_yaxes(title_text="R_nl(r)", row=1, col=1)
    except:
        fig.add_annotation(
            text="Radial function not available",
            x=0.5, y=0.5, xref="x domain", yref="y domain",
            showarrow=False, row=1, col=1
        )
    
    # Angular probability (polar plot)
    try:
        theta = np.linspace(0, np.pi, 100)
        phi = 0  # For m=0, phi doesn't matter
        Y_lm = sph_harm(m, l, phi, theta)
        angular_prob = np.abs(Y_lm)**2
        
        fig.add_trace(
            go.Scatterpolar(r=angular_prob, theta=theta*180/np.pi, 
                           mode='lines', name=f'|Y_{l}^{m}|²',
                           line=dict(color='red', width=2)),
            row=1, col=2
        )
    except:
        fig.add_annotation(
            text="Angular function not available",
            x=0.5, y=0.5, xref="x domain", yref="y domain",
            showarrow=False, row=1, col=2
        )
    
    # 3D angular shape
    try:
        theta_3d = np.linspace(0, np.pi, 30)
        phi_3d = np.linspace(0, 2*np.pi, 30)
        THETA, PHI = np.meshgrid(theta_3d, phi_3d)
        
        Y_lm_3d = sph_harm(m, l, PHI, THETA)
        Y_real = np.real(Y_lm_3d)
        
        R_surf = np.abs(Y_real)
        X = R_surf * np.sin(THETA) * np.cos(PHI)
        Y = R_surf * np.sin(THETA) * np.sin(PHI)
        Z = R_surf * np.cos(THETA)
        
        fig.add_trace(
            go.Surface(x=X, y=Y, z=Z, surfacecolor=Y_real,
                      colorscale='RdBu', opacity=0.8,
                      name=f'Y$_{l}^{m}$'),
            row=2, col=1
        )
    except:
        fig.add_annotation(
            text="3D shape not available",
            x=0.5, y=0.5, xref="x domain", yref="y domain",
            showarrow=False, row=2, col=2
        )
    
    # Energy level info table
    info_data = {
        'Parameter': ['n (radial)', 'l (angular)', 'm (magnetic)', 'Radius (nm)', 'Mass (×m₀)'],
        'Value': [n, l, m, f'{R:.1f}', f'{1.0:.2f}']
    }
    
    if E_eV != float('inf'):
        info_data['Parameter'].extend(['Energy (eV)', 'Energy (J)', 'Degeneracy'])
        info_data['Value'].extend([f'{E_eV:.3f}', f'{E_eV * eV_to_J:.2e}', f'{2*l+1}-fold'])
    else:
        info_data['Parameter'].append('Status')
        info_data['Value'].append('Invalid state')
    
    fig.add_trace(
        go.Table(
            header=dict(values=list(info_data.keys()), fill_color='lightblue'),
            cells=dict(values=list(info_data.values()), fill_color='white')
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def add_box_outline(fig, Lx, Ly, Lz):
    """Add box outline to 3D plot"""
    # Define box vertices
    vertices = [
        [0, 0, 0], [Lx, 0, 0], [Lx, Ly, 0], [0, Ly, 0],  # bottom face
        [0, 0, Lz], [Lx, 0, Lz], [Lx, Ly, Lz], [0, Ly, Lz]   # top face
    ]
    
    # Define edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    
    for edge in edges:
        points = np.array([vertices[edge[0]], vertices[edge[1]]])
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='lines', line=dict(color='black', width=2),
            showlegend=False
        ))

def energy_spherical_box(n, l, R, m):
    """Calculate energy for particle in spherical box"""
    zeros = find_spherical_bessel_zeros(l, num_zeros=max(10, n+2))
    
    if n > len(zeros):
        raise ValueError(f"Not enough zeros found for n={n}, l={l}")
    
    alpha_nl = zeros[n-1]
    E = (hbar**2 * alpha_nl**2) / (2 * m * R**2)
    return E

def particle_in_spherical_box_radial(r, n, l, R):
    """Radial part of the wavefunction for particle in spherical box"""
    zeros = find_spherical_bessel_zeros(l, num_zeros=max(10, n+2))
    
    if n > len(zeros):
        raise ValueError(f"Not enough zeros found for n={n}, l={l}")
    
    k_nl = zeros[n-1] / R
    A = np.sqrt(2 / R**3)
    kr = k_nl * r
    R_nl = A * spherical_bessel_j(l, kr)
    return R_nl

def spherical_bessel_j(n, x):
    """Spherical Bessel function j_n(x)"""
    if hasattr(x, '__iter__'):
        result = np.zeros_like(x)
        nonzero = x != 0
        result[nonzero] = np.sqrt(np.pi / (2 * x[nonzero])) * jv(n + 0.5, x[nonzero])
        return result
    else:
        if x == 0:
            return 1.0 if n == 0 else 0.0
        return np.sqrt(np.pi / (2 * x)) * jv(n + 0.5, x)

def find_spherical_bessel_zeros(n, num_zeros=10):
    """Find zeros of spherical Bessel functions"""
    x_start = np.pi * (np.arange(num_zeros) + n/2 + 1)
    zeros = []
    
    for x0 in x_start:
        x_range = np.linspace(x0 - 0.5, x0 + 0.5, 1000)
        j_vals = spherical_bessel_j(n, x_range)
        
        sign_changes = np.where(np.diff(np.sign(j_vals)))[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            x1, x2 = x_range[idx], x_range[idx + 1]
            j1, j2 = j_vals[idx], j_vals[idx + 1]
            zero = x1 - j1 * (x2 - x1) / (j2 - j1)
            zeros.append(zero)
    
    return np.array(zeros[:num_zeros])

if __name__ == "__main__":
    main()
