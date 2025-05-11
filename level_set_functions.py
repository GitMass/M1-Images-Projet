import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import laplace, gaussian_filter


def compute_phi_gradient(phi):
    """
    Compute the gradient of the level set function φ.

    Parameters:
    - phi: 2D array, the level set function

    Returns:
    - phi_x: derivative of φ in x direction (axis=1, columns)
    - phi_y: derivative of φ in y direction (axis=0, rows)
    - grad_phi_mag: gradient magnitude sqrt(φ_x^2 + φ_y^2)
    """
    grad_phi_y, grad_phi_x = np.gradient(phi)  # order: rows (y), columns (x)
    grad_phi_mag = np.sqrt(grad_phi_x**2 + grad_phi_y**2) + 1e-10  # small epsilon to avoid division by 0
    return grad_phi_x, grad_phi_y, grad_phi_mag

def div(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """Calculate divergence of vector field (nx, ny)"""
    [_, nxx] = np.gradient(nx)
    [nyy, _] = np.gradient(ny)
    return nxx + nyy

def dirac(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    Dirac delta function approximation
    """
    f = (1 / 2 / sigma) * (1 + np.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b

def neumann_bound_cond(f):
    """
    Apply Neumann boundary condition to function f
    """
    g = f.copy()

    g[np.ix_([0, -1], [0, -1])] = g[np.ix_([2, -3], [2, -3])]
    g[np.ix_([0, -1]), 1:-1] = g[np.ix_([2, -3]), 1:-1]
    g[1:-1, np.ix_([0, -1])] = g[1:-1, np.ix_([2, -3])]
    return g






# Init phi
def initialize_level_set_binary_square(img_shape, c0=1, square_size_ratio=0.8):
    """
    Initialize a level set function as a binary square.
    
    Parameters:
    -----------
    img_shape : tuple
        Shape of the image (height, width)
    square_size_ratio : float, optional
        Size of the square as a ratio of the image dimensions (default: 0.4)
        
    Returns:
    --------
    phi_init : ndarray
        Binary level set function with 1 inside the square and 0 outside
    """
    phi_init = np.zeros(img_shape)
    phi_init[:,:] = c0
    
    # Calculate square dimensions
    h, w = img_shape
    center_y, center_x = h // 2, w // 2
    square_size_y = int(h * square_size_ratio / 2)
    square_size_x = int(w * square_size_ratio / 2)
    
    # Set square region to 1
    y_min = max(0, center_y - square_size_y)
    y_max = min(h, center_y + square_size_y)
    x_min = max(0, center_x - square_size_x)
    x_max = min(w, center_x + square_size_x)
    
    phi_init[y_min:y_max, x_min:x_max] = -c0
    
    return phi_init







def dist_reg_p2(phi):
    """
    Compute the distance regularization term with the double-well potential p2
    """
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    a = (s >= 0) & (s <= 1)
    b = (s > 1)
    
    # Compute first order derivative of the double-well potential
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)
    
    # Compute d_p(s)=p'(s)/s with careful handling of s=0 case
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))
    
    return div(dps * phi_x - phi_x, dps * phi_y - phi_y) + laplace(phi, mode='nearest')


def compute_length_term(phi, g, epsilon=1.5):
    """
    Computes the edge-based term for the DRLSE model, combining both length and area terms.
    
    This function calculates the force exerted by image edges on the evolving level set function
    through two components:
    1. The geodesic length term (weighted by g): dirac_phi * g * curvature
    2. The advection term: dirac_phi * (vx * n_x + vy * n_y)
    
    Parameters:
    -----------
    phi : ndarray
        The level set function (LSF)
    g : ndarray
        The edge indicator function with low values at edges and high values in homogeneous regions
    epsilon : float
        Width parameter for the Dirac delta function approximation
        
    Returns:
    --------
    edge_term : ndarray
        Combined force from edge-based terms that drives the LSF toward object boundaries
    """
    [vy, vx] = np.gradient(g)  # Calculate gradient of edge indicator function

    [phi_y, phi_x] = np.gradient(phi)  # Calculate gradient of level set function
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))  # Gradient magnitude

    # Add a small positive number to avoid division by zero
    delta = 1e-10
    n_x = phi_x / (s + delta)  # Normalized x component of gradient
    n_y = phi_y / (s + delta)  # Normalized y component of gradient

    curvature = div(n_x, n_y)  # Calculate curvature

    # Calculate Dirac delta of level set function
    dirac_phi = dirac(phi, epsilon)

    # Calculate edge term (combination of advection and curvature terms)
    edge_term = dirac_phi * (vx * n_x + vy * n_y) + dirac_phi * g * curvature
    
    return edge_term

"""
def compute_length_term(phi, g, sigma):
    # Calculate gradient of phi
    phi_y, phi_x = np.gradient(phi)
    
    # Calculate gradient magnitude (add small epsilon to avoid division by zero)
    grad_norm = np.sqrt(phi_x**2 + phi_y**2 + 1e-10)
    
    # Normalize gradient vectors
    nx = phi_x / grad_norm
    ny = phi_y / grad_norm
    
    # Weight by edge indicator function
    weighted_nx = g * nx
    weighted_ny = g * ny
    
    # Calculate divergence of weighted normalized gradient
    div_y, div_x = np.gradient(weighted_nx)
    div_y2, div_x2 = np.gradient(weighted_ny)
    div = div_x + div_y2
    
    # Multiply by regularized Dirac delta
    dirac_phi = dirac(phi, sigma)
    length_term = dirac_phi * div
    
    return length_term
"""

def compute_area_term(phi, g, epsilon) :
    # Calculate Dirac delta of level set function
    dirac_phi = dirac(phi, epsilon)

    area_term = dirac_phi * g  # balloon/pressure force

    return area_term






def drlse_display(image, phi, g, epsilon, mu, lmda, alfa, i):

    # compute informations 
    
    reg_term = dist_reg_p2(phi)

    length_term = compute_length_term(phi, g, epsilon)

    area_term = compute_area_term(phi, g, epsilon) 


    # Display
    plt.figure(figsize=(20, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.contour(phi, levels=[0], colors='r')
    plt.title('Zero LS Contour')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(phi, cmap='coolwarm', vmin=-2, vmax=2)
    plt.colorbar()
    plt.title('Level Set Function')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    phi_x, phi_y, grad_phi_mag = compute_phi_gradient(phi)
    plt.imshow(grad_phi_mag, cmap='coolwarm', vmin=0, vmax=2)
    plt.colorbar()
    plt.title('grad_phi_mag')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(mu*reg_term, cmap='coolwarm', vmin=-0.2, vmax=0.2)
    plt.colorbar()
    plt.title('mu*regularisation_term')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(lmda*length_term, cmap='coolwarm', vmin=-0.2, vmax=0.2)
    plt.colorbar()
    plt.title('lmda*length_term')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(alfa*area_term, cmap='coolwarm', vmin=-0.2, vmax=0.2)
    plt.colorbar()
    plt.title('alfa*area_term')
    plt.axis('off')
    
    plt.suptitle(f"Iteration {i}")
    plt.tight_layout()
    plt.show()