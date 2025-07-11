# TFG-camara-plasma-
Simulación de campos electromagnéticos y dinámica de partículas cargadas en una cavidad resonante tipo pillbox. Incluye cálculo de modos EM con FEM (FEniCSx) y seguimiento de partículas con el algoritmo de Boris.


# Simulación de Campos Electromagnéticos y Dinámica de Partículas

Este repositorio contiene dos programas desarrollados para simular el comportamiento de partículas cargadas dentro de una cavidad resonante tipo *pillbox*.

- **Campos EM:** Cálculo de modos electromagnéticos resonantes mediante el método de los elementos finitos (FEM), utilizando FEniCSx.
- **Dinámica de partículas:** Integración de las trayectorias de partículas cargadas bajo la acción de los campos mediante el algoritmo de Boris.

Ambas simulaciones están orientadas al estudio de fuentes de iones ECR y procesos de aceleración en cavidades resonantes.

## Requisitos

- Python 3.10+
- FEniCSx
- NumPy
- SciPy
- Matplotlib
- Multiprocessing
- tqdm
- Gmsh

Se recomienda utilizar un entorno virtual o entorno basado en Docker para ejecutar los scripts.

## Uso

1. Ejecutar `em_solver/solve_modes.py` para calcular y guardar los campos electromagnéticos.
2. Ejecutar `particle_dynamics/simulate_particles.py` para cargar los campos y simular el movimiento de partículas.

## Licencia

[MIT License](LICENSE) — uso libre con atribución.

## Autor

Lucas Morales del Arco — Trabajo de Fin de Grado en Física, UPV/EHU, 2025.
