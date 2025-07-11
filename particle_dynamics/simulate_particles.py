# -*- coding: utf-8 -*-
# Este script simula la evolución de partículas en una cámara de plasma usando campos eléctricos y magnéticos.
# Se utilizan métodos de física computacional y visualización para analizar trayectorias y densidades.

import numpy as np
from tqdm import tqdm
import scipy.constants as cte
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from dolfinx.fem import Function, assemble_scalar, form, functionspace
from multiprocessing import Pool
import socket
import dolfinx
from dolfinx.io import XDMFFile
from mpi4py import MPI
import time
import csv
  
# Leer la malla y los campos eléctricos desde archivos externos
with XDMFFile(MPI.COMM_WORLD, "plasma_fields.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()

u_values = np.load("E_dg_graf.npy")
V = functionspace (mesh, ("DG", 0, (3,)))
gu_save = Function(V)
gu_save.x.array[:] = u_values.ravel()

# Genera posición y velocidad aleatoria para una partícula
# m: masa, W: energía, rad: radio, long: longitud
# Devuelve posición (x0, y0, z0) y velocidad (vx0, vy0, vz0)
def pos_vel_random(m,W,rad,long): 

    gamma = (W+m*cte.c**2)/(m*cte.c**2)
    v = cte.c * np.sqrt((gamma**2-1)/gamma**2)

    vx0,vy0,vz0 = np.random.randn(3)
    norm = np.sqrt(vx0**2+vy0**2+vz0**2)
    vx0,vy0,vz0 = vx0/norm,vy0/norm,vz0/norm

    theta = np.random.uniform(0, 2 * np.pi)
    r = np.sqrt(np.random.uniform(0, rad**2))  
    x0 = r * np.cos(theta)
    y0 = r * np.sin(theta)

    z0 = np.random.uniform(0, long)
    return x0, y0, z0, vx0*v, vy0*v, vz0*v

# Obtiene el campo eléctrico en una posición (x, y, z)
def getfields(x, y, z):
    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    points = np.array([[x, y, z]], dtype=mesh.geometry.x.dtype)
    potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, potential_colliding_cells, points)

    cells = []
    if len(colliding_cells.links(0)) > 0:
        cells.append(colliding_cells.links(0)[0])

    points_on_proc = np.array(points, dtype=np.float64)
    cells = np.array(cells, dtype=np.int32)
    E_values = gu_save.eval(points_on_proc, cells)
    E_values[-1] = 0
    return E_values


host_actual = socket.gethostname()

# Configuración de entorno: local o remoto
if host_actual == "ESSB145":
    print("Corriendo en local...")
    import imageio.v2 as imageio
else : 
    print("Corriendo en remoto...")

# Inicialización de parámetros físicos y simulación
periodo = 2 * np.pi * cte.electron_mass/(0.09645 * cte.elementary_charge)
W, R, L1, B0 = 0.04 * cte.elementary_charge, 0.04, 0.0938, 0.09645
t0, tf, num, phi, om = 0.0, 6*periodo, 100, 0.0, 1/periodo * 2* np.pi
dt = (tf - t0) / num
tao = np.linspace(t0, tf, num)
m1, m2, m3, me, charge = cte.electron_mass, cte.electron_mass, cte.electron_mass, cte.electron_mass, cte.elementary_charge
# m1, m2, m3, me, charge = cte.proton_mass, 3.347047356*1e-27, 5.008267523387676*1e-27, cte.electron_mass, cte.elementary_charge
chrono = []
E_final_electron = []

npart = 500000
nproc = 24
charge = cte.elementary_charge

# Algoritmo de integración de Boris para evolución de partículas
# fx: estado de la partícula, tao_val: tiempo actual
def megaboris1(args): 
    fx, tao_val = args
    if fx[6] == cte.electron_mass:
        charge = - cte.elementary_charge
    else: 
        charge = cte.elementary_charge
    radio1 = np.sqrt(fx[0]**2 + fx[1]**2)
    if (0.0 <= radio1 <= R) and (0.0 <= fx[2] <= L1):
        m = fx[6]
        elec = getfields(fx[0], fx[1, fx[2]]) * np.cos(om * tao_val + phi)
        elec[2] = 0.0
        B = np.array([0,0,B_ext])  
        v_minus = fx[3:6] + (charge * elec / m) * (dt / 2)
        t = (charge * B / m) * (dt / 2)
        t_mag2 = np.dot(t, t)
        s = 2 * t / (1 + t_mag2)
        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)
        fx[3:6] = v_plus + (charge * elec / m) * (dt / 2)
        v = np.array(fx[3:6])
        fx[0:3] = list(np.array(fx[0:3]) + v * dt)
    return fx

# Calcula histogramas de densidad y energía en diferentes planos
# tag: 'xy' o 'xz', type: tipo de partícula
def histograma(x,y,z,E,tag,type):
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)
    zmin = min(z)
    zmax = max(z)

    if tag == "xz":
        npart_hist, xedges, qedges = np.histogram2d(x, z, bins=100, range=[[xmin, xmax], [zmin, zmax]])
        energy_hist, xedges, qedges = np.histogram2d(x, z, weights=E, bins=100, range=[[xmin, xmax], [zmin, zmax]])

        dx = np.diff(xedges) 
        dz = np.diff(qedges)
        volumen_celdas = 2 * np.sqrt(R**2 - xedges[:-1]**2) * dx  * dz * 1e6    # cm

        densidad_npart = npart_hist / volumen_celdas[:, np.newaxis]  
        densidad_energy = np.divide(energy_hist, npart_hist, out=np.zeros_like(energy_hist), where=npart_hist != 0)

        densidad_npart = densidad_npart / np.max(densidad_npart) 

    if tag == "xy":
        mask = (x**2 + y**2) <= R**2  
        x = x[mask]
        y = y[mask]
        E = E[mask]
        npart_hist, xedges, qedges = np.histogram2d(x, y, bins=100, range=[[xmin, xmax], [ymin, ymax]])
        energy_hist, xedges, qedges = np.histogram2d(x, y, weights=E, bins=100, range=[[xmin, xmax], [ymin, ymax]])

        dx = np.diff(xedges)  
        dy = np.diff(qedges)  
        volumen_celdas = dx[:, np.newaxis] * dy  * L1  * 1e6

        densidad_npart = npart_hist / volumen_celdas  
        densidad_energy = np.divide(energy_hist, npart_hist, out=np.zeros_like(energy_hist), where=npart_hist != 0)

        densidad_npart = densidad_npart / np.max(densidad_npart)  

    return npart_hist, energy_hist, densidad_npart, densidad_energy, xedges, qedges

def get_histogram(x, y, z, vx, vy, vz, xs):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)

    # Procesa los histogramas para cada tipo de partícula y plano
    # Divide los datos en 4 grupos (3 tipos de iones y electrones)
    x1, y1, z1, vx1, vy1, vz1 = x[:npart//4], y[:npart//4], z[:npart//4], vx[:npart//4], vy[:npart//4], vz[:npart//4]
    x2, y2, z2, vx2, vy2, vz2 = x[npart//4:2*npart//4], y[npart//4:2*npart//4], z[npart//4:2*npart//4], vx[npart//4:2*npart//4], vy[npart//4:2*npart//4], vz[npart//4:2*npart//4]
    x3, y3, z3, vx3, vy3, vz3 = x[2*npart//4:3*npart//4], y[2*npart//4:3*npart//4], z[2*npart//4:3*npart//4], vx[2*npart//4:3*npart//4], vy[2*npart//4:3*npart//4], vz[2*npart//4:3*npart//4]
    x4, y4, z4, vx4, vy4, vz4 = x[3*npart//4:], y[3*npart//4:], z[3*npart//4:], vx[3*npart//4:], vy[3*npart//4:], vz[3*npart//4:]

    # Calcula energía relativista para cada grupo
    v1 = (vx1**2 + vy1**2 + vz1**2)
    v2 = (vx2**2 + vy2**2 + vz2**2)
    v3 = (vx3**2 + vy3**2 + vz3**2)
    v4 = (vx4**2 + vy4**2 + vz4**2)

    gamma1 = 1/np.sqrt(1-v1/(cte.c)**2)
    gamma2 = 1/np.sqrt(1-v2/(cte.c)**2)
    gamma3 = 1/np.sqrt(1-v3/(cte.c)**2)
    gamma4 = 1/np.sqrt(1-v4/(cte.c)**2)

    E1 = (gamma1 - 1) * m1 * (cte.c)**2 / (cte.elementary_charge)  # eV
    E2 = (gamma2 - 1) * m2 * (cte.c)**2 / (cte.elementary_charge)
    E3 = (gamma3 - 1) * m3 * (cte.c)**2 / (cte.elementary_charge)
    E4 = (gamma4 - 1) * me * (cte.c)**2 / (cte.elementary_charge)

    with Pool(8) as pool:
        resultados = pool.starmap(histograma, [
            (x1, y1, z1, E1, "xy","H1+"),  
            (x2, y2, z2, E2, "xy","H2+"),  
            (x3, y3, z3, E3, "xy","H3+"), 
            (x4, y4, z4, E4, "xy","e"), 
            (x1, y1, z1, E1, "xz","H1+"),  
            (x2, y2, z2, E2, "xz","H2+"),  
            (x3, y3, z3, E3, "xz","H3+"), 
            (x4, y4, z4, E4, "xz","e")  
        ])
        
    npart_hist, energy_hist, densidad_npart, densidad_energy, xedges, qedges = zip(*resultados)

    npart_xy = npart_hist[:3]
    npart_xz = npart_hist[4:7]
    mean_npart_xy = densidad_npart[:3]
    mean_npart_xz = densidad_npart[4:7]
       
    energy_xy = energy_hist[:3]
    energy_xz = energy_hist[4:7]
    mean_energy_xy = densidad_energy[:3]
    mean_energy_xz = densidad_energy[4:7]

    xedges_xy = xedges[:3]
    xedges_xz = xedges[4:7]
    yedges = qedges[:3]
    zedges = qedges[4:7]

    mean_epart =[densidad_npart[3],densidad_npart[7]]
    mean_energy_e =[densidad_energy[3],densidad_energy[7]]

    # Visualización de resultados: densidad y energía en diferentes planos
    fig1_xz, axes1_xz = plt.subplots(1, 3, figsize=(18, 6))
    for i, density in enumerate(mean_npart_xz):
        ax = axes1_xz[i]
        c = ax.imshow(density.T, origin='lower', extent=[xedges_xz[i][0], xedges_xz[i][-1], zedges[i][0], zedges[i][-1]], cmap='plasma', aspect='auto')
        ax.set_title(f'H{i+1}+')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        fig1_xz.colorbar(c, ax=ax, label='Densidad de partículas')

    fig1_xy, axes1_xy = plt.subplots(1, 3, figsize=(18, 6))
    for i, density in enumerate(mean_npart_xy):
        ax = axes1_xy[i]
        c = ax.imshow(density.T, origin='lower', extent=[xedges_xy[i][0], xedges_xy[i][-1], yedges[i][0], yedges[i][-1]], cmap='plasma', aspect='auto')
        ax.set_title(f'H{i+1}+')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig1_xy.colorbar(c, ax=ax, label='Densidad de partículas')

    fig1e, axes1e = plt.subplots(1, 2, figsize=(18, 6))
    for i, density in enumerate(mean_epart):
        ax = axes1e[i]
        if i == 1:
            xed = xedges[7]
            qed = qedges[7]
            ax.set_ylabel('z')
        else: 
            xed = xedges[3]
            qed = qedges[3]
            ax.set_ylabel('y')
        c = ax.imshow(density.T, origin='lower', extent=[xed[0], xed[-1], qed[0], qed[-1]], cmap='plasma', aspect='equal')
        ax.set_title(f'e')
        ax.set_xlabel('x')
        fig1e.colorbar(c, ax=ax, label='Densidad de partículas')


    fig2_xz, axes2_xz = plt.subplots(1, 3, figsize=(18, 6))
    for i, energy in enumerate(mean_energy_xz):
        ax = axes2_xz[i]
        c = ax.imshow(energy.T, origin='lower', extent=[xedges_xz[i][0], xedges_xz[i][-1], zedges[i][0], zedges[i][-1]], cmap='plasma', aspect='auto')
        ax.set_title(f'H{i+1}+')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        fig2_xz.colorbar(c, ax=ax, label='Densidad de energía')

    fig2_xy, axes2_xy = plt.subplots(1, 3, figsize=(18, 6))
    for i, energy in enumerate(mean_energy_xy):
        ax = axes2_xy[i]
        c = ax.imshow(energy.T, origin='lower', extent=[xedges_xy[i][0], xedges_xy[i][-1], yedges[i][0], yedges[i][-1]], cmap='plasma', aspect='auto')
        ax.set_title(f'H{i+1}+')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig2_xy.colorbar(c, ax=ax, label='Densidad de energía')

    fig2e, axes2e = plt.subplots(1, 2, figsize=(18, 6))
    for i, energy in enumerate(mean_energy_e):
        ax = axes2e[i]
        if i == 1:
            xed = xedges[7]
            qed = qedges[7]
            ax.set_ylabel('z')
        else: 
            xed = xedges[3]
            qed = qedges[3]
            ax.set_ylabel('y')
        c = ax.imshow(energy.T, origin='lower', extent=[xed[0], xed[-1], qed[0], qed[-1]], cmap='plasma', aspect='equal')
        ax.set_title(f'e')
        ax.set_xlabel('x')
        fig2e.colorbar(c, ax=ax, label='Densidad de energía')

    # Guarda las figuras en la carpeta correspondiente
    carpeta_salida = "/home/lucas/fenicsx/Pillbox/densidades2/" if host_actual == "ESSB145" else "./"
    fig1_xy.suptitle(f"Distribución de densidad de partículas en el plano XY. Número de partículas: {npart}. Campo B = {B_ext} T",
                    fontsize=16, fontweight='bold')
    fig1_xz.suptitle(f"Distribución de densidad de partículas en el plano XZ. Número de partículas: {npart}. Campo B = {B_ext} T",
                    fontsize=16, fontweight='bold')
    fig1e.suptitle(f"Distribución de densidad de electrones en las secciones XZ y XY. Número de partículas: {npart}. Campo B = {B_ext} T",
                    fontsize=16, fontweight='bold')
    fig2_xy.suptitle(f"Distribución de densidad de energía promedio en el plano XY. Número de partículas: {npart}. Campo B = {B_ext} T",
                fontsize=16, fontweight='bold')
    fig2_xz.suptitle(f"Distribución de densidad de energía promedio en el plano XY. Número de partículas: {npart}. Campo B = {B_ext} T",
                    fontsize=16, fontweight='bold')
    fig2e.suptitle(f"Distribución de densidad de energía promedio de electrones en las secciones XZ y. Número de partículas: {npart}. Campo B = {B_ext} T",
                    fontsize=16, fontweight='bold')

    plt.tight_layout()
    
    fig1_xy.savefig(f"{carpeta_salida}xy_densidad_B{xs}.png", dpi=300)
    fig1_xz.savefig(f"{carpeta_salida}xz_densidad_B{xs}.png", dpi=300)
    fig1e.savefig(f"{carpeta_salida}e_densidad_B{xs}.png", dpi=300)
    fig2_xy.savefig(f"{carpeta_salida}xy_energia_B{xs}.png", dpi=300)
    fig2_xz.savefig(f"{carpeta_salida}xz_energia_B{xs}.png", dpi=300)
    fig2e.savefig(f"{carpeta_salida}e_energia_B{xs}.png", dpi=300)

    plt.close(fig1_xy)
    plt.close(fig1_xz)
    plt.close(fig1e)
    plt.close(fig2_xy)
    plt.close(fig2_xz)
    plt.close(fig2e)

def get_histogram_electron(x, y, z, vx, vy, vz, xs):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)

    # Calcular velocidad y energía relativista para todas las partículas (electrones)
    v = (vx**2 + vy**2 + vz**2)
    gamma = 1 / np.sqrt(1 - v / (cte.c)**2)
    E = (gamma - 1) * me * (cte.c)**2 / cte.elementary_charge  # en eV

    with Pool(2) as pool:
        resultados = pool.starmap(histograma, [
            (x, y, z, E, "xy", "e"),
            (x, y, z, E, "xz", "e"),
        ])

    npart_hist, energy_hist, densidad_npart, densidad_energy, xedges, qedges = zip(*resultados)

    mean_epart = densidad_npart
    mean_energy_e = densidad_energy

    # Visualización y guardado de histogramas para electrones
    fig_xy, ax_xy = plt.subplots(figsize=(9, 6))
    c_xy = ax_xy.imshow(mean_epart[0].T, origin='lower',
                        extent=[xedges[0][0], xedges[0][-1], qedges[0][0], qedges[0][-1]],
                        cmap='plasma', aspect='auto')
    ax_xy.set_title("Densidad de electrones (XY)")
    ax_xy.set_xlabel('x')
    ax_xy.set_ylabel('y')
    fig_xy.colorbar(c_xy, ax=ax_xy, label='Densidad de partículas')

    fig_xz, ax_xz = plt.subplots(figsize=(9, 6))
    c_xz = ax_xz.imshow(mean_epart[1].T, origin='lower',
                        extent=[xedges[1][0], xedges[1][-1], qedges[1][0], qedges[1][-1]],
                        cmap='plasma', aspect='auto')
    ax_xz.set_title("Densidad de electrones (XZ)")
    ax_xz.set_xlabel('x')
    ax_xz.set_ylabel('z')
    fig_xz.colorbar(c_xz, ax=ax_xz, label='Densidad de partículas')

    fig_e_xy, ax_e_xy = plt.subplots(figsize=(9, 6))
    e_xy = ax_e_xy.imshow(mean_energy_e[0].T, origin='lower',
                        extent=[xedges[0][0], xedges[0][-1], qedges[0][0], qedges[0][-1]],
                        cmap='plasma', aspect='auto')
    ax_e_xy.set_title("Densidad de energía promedio (XY)")
    ax_e_xy.set_xlabel('x')
    ax_e_xy.set_ylabel('y')
    fig_e_xy.colorbar(e_xy, ax=ax_e_xy, label='Densidad de energía')

    fig_e_xz, ax_e_xz = plt.subplots(figsize=(9, 6))
    e_xz = ax_e_xz.imshow(mean_energy_e[1].T, origin='lower',
                        extent=[xedges[1][0], xedges[1][-1], qedges[1][0], qedges[1][-1]],
                        cmap='plasma', aspect='auto')
    ax_e_xz.set_title("Densidad de energía promedio (XZ)")
    ax_e_xz.set_xlabel('x')
    ax_e_xz.set_ylabel('z')
    fig_e_xz.colorbar(e_xz, ax=ax_e_xz, label='Densidad de energía')
    # Guardar figuras
    carpeta_salida = "/home/lucas/fenicsx/Pillbox/densidades2/" if host_actual == "ESSB145" else "./"

    fig_xy.suptitle(f"Densidad de electrones en el plano XY. N = {npart}. B = {B_ext} T", fontsize=14, fontweight='bold')
    fig_xz.suptitle(f"Densidad de electrones en el plano XZ. N = {npart}. B = {B_ext} T", fontsize=14, fontweight='bold')
    fig_e_xy.suptitle(f"Energía media de electrones en el plano XY. N = {npart}. B = {B_ext} T", fontsize=14, fontweight='bold')
    fig_e_xz.suptitle(f"Energía media de electrones en el plano XZ. N = {npart}. B = {B_ext} T", fontsize=14, fontweight='bold')

    fig_xy.savefig(f"{carpeta_salida}electron_xy_densidad_B{xs}.png", dpi=300)
    fig_xz.savefig(f"{carpeta_salida}electron_xz_densidad_B{xs}.png", dpi=300)
    fig_e_xy.savefig(f"{carpeta_salida}electron_xy_energia_B{xs}.png", dpi=300)
    fig_e_xz.savefig(f"{carpeta_salida}electron_xz_energia_B{xs}.png", dpi=300)

    plt.close(fig_xy)
    plt.close(fig_xz)
    plt.close(fig_e_xy)
    plt.close(fig_e_xz)

def get_paths(e_path,p_path,tao,electricfield,xs,B_ext):

    e_path, p_path, tao = np.array(e_path), np.array(p_path), np.array(tao)

    xe, ye, ze = e_path[:,0], e_path[:,1], e_path[:,2]
    xp, yp, zp = p_path[:,0], p_path[:,1], p_path[:,2]

    vxe, vye, vze = e_path[:,3], e_path[:,4], e_path[:,5]

    vmod = vxe**2 + vye**2 + vze**2
    gamma = 1/np.sqrt(1-vmod/(cte.c)**2)
    E = (gamma - 1) * me * (cte.c)**2 / (cte.elementary_charge)

    E_final_electron.append(E[-1])

    # Trayectoria electrón y protón en 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc1 = ax.scatter(xe, ye, ze, c=tao, cmap='viridis', marker='o', s=30, label='Electrón', alpha=0.7)
    ax.plot(xe, ye, ze, 'b-', linewidth=1, alpha=0.3)  

    sc2 = ax.scatter(xp, yp, zp, c=tao, cmap='plasma', marker='^', s=30, label='Protón', alpha=0.7)
    ax.plot(xp, yp, zp, 'r--', linewidth=1, alpha=0.3)  


    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Trayectoria del electrón y protón desde el mismo punto inicial', fontweight='bold', pad=20)
    ax.legend(loc='upper right')

    # Trayectoria electrón y protón en 2D
    fig2, axes = plt.subplots(1, 2, figsize=(18, 6))
    planes = [('XY', 'Y'), ('XZ', 'Z')]  
    for i, (plane, ylabel) in enumerate(planes):
        qe, be = (xe, ye) if plane == 'XY' else (xe, ze)
        qp, bp = (xp, yp) if plane == 'XY' else (xp, zp)
        
        axes[i].scatter(qe, be, c='blue', marker='o', s=50, alpha=0.7, label='Electrón')
        axes[i].scatter(qp, bp, c='red', marker='^', s=50, alpha=0.7, label='Protón')
        
        axes[i].plot(qe, be, 'b-', alpha=0.2, linewidth=0.5)
        axes[i].plot(qp, bp, 'r-', alpha=0.2, linewidth=0.5)
        
        axes[i].set_xlabel('X', fontsize=12)
        axes[i].set_ylabel(ylabel, fontsize=12)
        axes[i].set_title(f'Plano {plane}', fontweight='bold')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    fig2.suptitle('Comparación de Trayectorias: Electrón vs Protón', fontsize=16, y=1.02)

    # Trayectoria protón en 3D
    fig3 = plt.figure(figsize=(10, 7))
    axi = fig3.add_subplot(111, projection='3d')
    sc = axi.scatter(xp, yp, zp, c=tao, cmap='plasma', s=50)
    cbar = plt.colorbar(sc, ax=axi)
    cbar.set_label('Tiempo')
    axi.set_xlabel('X')
    axi.set_ylabel('Y')
    axi.set_zlabel('Z')
    axi.set_title('Trayectoria de una partícula en 3D')


    # Trayectoria protón en 2D
    fig4, axes4 = plt.subplots(1, 2, figsize=(18, 6))
    planes = [('XY', 'Y'), ('XZ', 'Z')]  
    for i, (plane, ylabel) in enumerate(planes):
        qp, bp = (xp, yp) if plane == 'XY' else (xp, zp)
        
        axes4[i].scatter(qp, bp, c='red', marker='^', s=50, alpha=0.7, label='Protón')
        
        axes4[i].plot(qp, bp, 'r-', alpha=0.2, linewidth=0.5)
        
        axes4[i].set_xlabel('X', fontsize=12)
        axes4[i].set_ylabel(ylabel, fontsize=12)
        axes4[i].set_title(f'Plano {plane}', fontweight='bold')
        axes4[i].legend(loc='upper right')
        axes4[i].grid(True, linestyle='--', alpha=0.5)
        

    E_mean = np.mean(E)
    print(f"Energía promedio en el tiempo <E>(t) = {E_mean} eV")

    # Energia respecto del tiempo
    fig5 = plt.figure(figsize=(8, 5))
    plt.plot(tao, E, label='E vs t', color='royalblue')
    plt.xlabel('t')
    plt.ylabel('E')
    plt.title(f'Energía del electrón respecto del tiempo. <E>(t) = {E_mean}.')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()




    carpeta_salida = "/home/lucas/fenicsx/Pillbox/densidades2/" if host_actual == "ESSB145" else "./"

    plt.tight_layout()

    fig.savefig(f"{carpeta_salida}path_e_p_B{xs}.png", dpi=300)
    fig2.savefig(f"{carpeta_salida}path2d_e_p_B{xs}.png", dpi=300)
    fig3.savefig(f"{carpeta_salida}path_p_B{xs}.png", dpi=300)
    fig4.savefig(f"{carpeta_salida}path2d_p_B{xs}.png", dpi=300)
    fig5.savefig(f"{carpeta_salida}Evst_B{xs}.png", dpi=300)



    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)


def getfieldsgraph(Ex,Ey,Ez,tao,xs):
    fig6 = plt.figure(figsize=(10, 6))
    plt.plot(tao, Ex, label='$E_x$', color='r')
    plt.plot(tao, Ey, label='$E_y$', color='g')
    plt.plot(tao, Ez, label='$E_z$', color='b')

    plt.xlabel('Tiempo (s)')
    plt.ylabel('Campo Eléctrico (V/m)')
    plt.title('Componentes del Campo Eléctrico en función del Tiempo')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    carpeta_salida = "/home/lucas/fenicsx/Pillbox/densidades2/" if host_actual == "ESSB145" else "./"   
    fig6.savefig(f"{carpeta_salida}electric_vs_t_B{xs}.png", dpi=300)


    plt.close(fig6)

# La función getresonance grafica la energía final de las partículas en función del campo magnético aplicado
# (B_ext_values). Recibe una lista de valores de campo magnético y una lista de energías finales (E_final),
# que corresponden a simulaciones independientes con las mismas condiciones iniciales pero diferentes B_ext.
# Así se puede analizar cómo varía la energía final del conjunto de partículas según el campo aplicado.
def getresonance(B_ext_values,E_final):
    fig7 = plt.figure(figsize=(10, 6))
    plt.plot(B_ext_values, E_final)

    plt.xlabel('B [T]')
    plt.ylabel('E final [eV]')
    plt.title('Energía final del electrón respecto del campo magnético aplicado')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    carpeta_salida = "/home/lucas/fenicsx/Pillbox/densidades2/" if host_actual == "ESSB145" else "./"   
    fig7.savefig(f"{carpeta_salida}resonance.png", dpi=300)


    plt.close(fig7)

B_ext = B0
y1, y1_0 = [], []

for k in range(npart):
    mass = m1 if k < npart // 4 else m2 if k >= npart // 4 and k < npart // 2 else m3 if k >= npart // 2 and k < 3 * npart // 4 else me
    x0, y0, z0, vx0, vy0, vz0 = pos_vel_random(mass, W, R, L1)
    suby1 = [x0,y0,z0,vx0,vy0,vz0,mass]
    y1.append(suby1)


y1[0][:3] = y1[-1][:3] 

ultimo_e = np.array(y1[-1])
vi = ultimo_e[3]**2 + ultimo_e[4]**2 + ultimo_e[5]**2 
gi = 1/np.sqrt(1-vi/(cte.c)**2)
Ei = (gi - 1) * me * (cte.c)**2 / (cte.elementary_charge)

print(f"Energía inicial del electrón: {Ei} eV. Con masa: {ultimo_e[-1] /cte.elementary_charge * cte.c**2 * 10**(-6)} MeV/c^2")
y1_0 = y1
x, y, z, vx, vy, vz = zip(*[sublista[:6] for sublista in y1])
x = list(x)
y = list(y)
z = list(z)
vx = list(vx)
vy = list(vy)
vz = list(vz)

xs = "CI" 
print(f"Guardando gráficas de la densidad de partículas y de energía al 0%")
get_histogram(x, y, z, vx, vy, vz, xs)


B_ext_values = [B0]


# Este bucle for principal recorre los valores de campo magnético B_ext_values.
# Para cada valor, calcula la evolución temporal de todas las partículas.
# En cada paso de tiempo, utiliza multiprocessing para aplicar el algoritmo de Boris en paralelo a cada partícula,
# lo que permite simular eficientemente la dinámica de muchas partículas bajo las mismas condiciones iniciales.
# Al finalizar, guarda los resultados y genera las gráficas correspondientes.
for run_idx, B_ext in enumerate(B_ext_values):
    print(f"Ejecución {run_idx + 1} con B_ext = {B_ext}")
    y1 = y1_0    
    e_path, p_path = [],[]
    Ex, Ey, Ez = [],[]
    for i in tqdm(range(len(tao)), desc=f"Calculando la evolución del sistema..."):
        with Pool(nproc) as p:  
            e_path.append(y1[-1])
            p_path.append(y1[0])
            E_valores = getfields(y1[-1][0],y1[-1][1],y1[-1][2])
            E_valores = E_valores * np.cos(om * tao[i] + phi)
            Ex.append(E_valores[0])
            Ey.append(E_valores[1]) 
            Ez.append(E_valores[2])
            # Multiprocessing: aplica el algoritmo Boris en paralelo a todas las partículas
            y1 = p.map(megaboris1, [(sublista, tao[i]) for sublista in y1])
    # ...existing code...
    if i == int(len(tao) - 1):
        x, y, z, vx, vy, vz = zip(*[sublista[:6] for sublista in y1])
        x = list(x)
        y = list(y)
        z = list(z)
        vx = list(vx)
        vy = list(vy)
        vz = list(vz)

        xs = str(int(run_idx + 1))
        print(f"Guardando gráficas de la densidad de partículas y de energía al 0% para B = {B_ext} T")

        get_histogram_electron(x, y, z, vx, vy, vz, xs)
        #get_histogram(x, y, z, vx, vy, vz, xs)

        E_t = np.array([Ex,Ey,Ez])

        get_paths(e_path,p_path,tao,E_t,xs,B_ext)

        getfieldsgraph(Ex,Ey,Ez,tao,xs)

        with open(f"./histogramas/posvel_B{xs}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z", "vx", "vy", "vz"])  
            for xi, yi, zi, vxi, vyi, vzi in zip(x, y, z, vx, vy, vz):
                writer.writerow([xi, yi, zi, vxi, vyi, vzi])


    print(f"Ejecución {run_idx + 1} finalizada.")


getresonance(B_ext_values, E_final_electron)


print("Simulaciones completadas.")
# El resto del código ejecuta la simulación principal, guarda los resultados y muestra mensajes informativos.
# Se generan posiciones y velocidades iniciales, se evoluciona el sistema y se visualizan los resultados.



