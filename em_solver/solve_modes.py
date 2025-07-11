# -*- coding: utf-8 -*-
# Este script calcula los modos electromagnéticos de una cámara de plasma cilíndrica usando elementos finitos.
# Utiliza GMSH para generar la malla, FEniCSx para resolver el problema de autovalores y SLEPc para encontrar las frecuencias propias.
# Los campos eléctricos y magnéticos obtenidos se guardan para su posterior análisis.

import gmsh
import math 
import numpy as np
import dolfinx
import ufl
from dolfinx.fem import Function, assemble_scalar, form, functionspace
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from ufl import TestFunction, TrialFunction, as_vector, dot, dx, grad, inner, curl
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI
from dolfinx import default_scalar_type
from dolfinx import fem
from dolfinx.mesh import locate_entities_boundary, exterior_facet_indices
from dolfinx.fem import (dirichletbc, Expression, Function, FunctionSpace, functionspace, Constant,
                         locate_dofs_topological, assemble_matrix, petsc, form)
from petsc4py.PETSc import ScalarType
from slepc4py import SLEPc
import scipy.constants as cte
import matplotlib.pyplot as plt
from petsc4py import PETSc
from tqdm import tqdm
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde

 
 

# Función para crear la malla de la cámara cilíndrica usando GMSH
# L1: longitud, R: radio, lc: tamaño de celda, mesh_comm: comunicador MPI, model_rank: rango del modelo
# Devuelve la malla y las etiquetas de entidades

def chamber_mesh(L1,R,lc,mesh_comm,model_rank):
    gmsh.initialize()
    gmsh.model.geo.addPoint(R,0,0,lc,1)
    gmsh.model.geo.addPoint(0,R,0,lc,2)
    gmsh.model.geo.addPoint(-R,0,0,lc,3)
    gmsh.model.geo.addPoint(0,-R,0,lc,4)
    gmsh.model.geo.addPoint(0,0,0,lc,5)

    gmsh.model.geo.addCircleArc(1,5,2,10)
    gmsh.model.geo.addCircleArc(2,5,3,11)
    gmsh.model.geo.addCircleArc(3,5,4,12)
    gmsh.model.geo.addCircleArc(4,5,1,13)

    gmsh.model.geo.addCurveLoop([10,11,12,13],20)
    gmsh.model.geo.addPlaneSurface([20],21)
    
    cylinder = gmsh.model.geo.extrude([(2,21)],0,0,L1)
    surface = []
    volume = []
    for elemento in cylinder: 
        if elemento[0] == 2:
            surface.append(elemento[1])
        if elemento[0] == 3:
            volume.append(elemento[1])
    
    gmsh.model.addPhysicalGroup(2,surface,1,name= "superficie")
    gmsh.model.addPhysicalGroup(3,volume,2,name="volumen")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.generate(3)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2)
    gmsh.write ("cam_plasma_graf"+'.msh')

    mesh, ct, ft = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=3)

    # gmsh.fltk.run()

    gmsh.finalize()

    return mesh, ct, ft


# Inicialización de parámetros físicos y generación de la malla
L1, R = 0.0938, 0.040
lc = 0.25e-2
target_frequency= 2.7e9  # Frecuencia objetivo en Hz
eps = cte.epsilon_0
mu = cte.mu_0
c = cte.c
q = cte.e
mp = cte.proton_mass
qmp = q/mp
mesh_comm = MPI.COMM_WORLD
rank = mesh_comm.Get_rank()
model_rank = 0
mesh, ct, ft = chamber_mesh(L1,R,lc,mesh_comm=mesh_comm,model_rank=model_rank)

# Definición del espacio de funciones de Nedelec para campos electromagnéticos
nedelec_order = 1 
V = functionspace(mesh,("N1curl",nedelec_order))
u = TrialFunction(V)
v = TestFunction(V)

# Formulación débil del problema de autovalores
s = inner(curl(v),curl(u)) * dx 
t = inner(v,u) * dx 
# Condiciones de contorno: campo nulo en el borde
u_bc = Function(V)
u_bc.x.array.fill(0)
fdim = mesh.topology.dim - 1 
facets = exterior_facet_indices(mesh.topology)
dofs = locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)
bc=dirichletbc(u_bc, dofs=dofs)

# Ensamblaje de matrices y resolución del problema de autovalores
S = petsc.assemble_matrix(form(s), bcs=[bc])
S.assemble()
T = petsc.assemble_matrix(form(t), bcs=[bc])
T.assemble()

esolver = SLEPc.EPS().create(MPI.COMM_WORLD)
esolver.setOperators(S, T)
esolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
esolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
st = esolver.getST()
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(((2*cte.pi*target_frequency/cte.c)**2))
esolver.setDimensions(10)

# Obtención de las frecuencias propias y selección del modo de interés
esolver.solve()
nsolved = esolver.getConverged()
cuenta=0
freqs=[]
for i in range(nsolved):
    lr = (esolver.getEigenvalue(i)).real
    if lr.real > 1:
        autov = math.sqrt(lr.real)
        f_mode=autov*cte.c/(2*np.pi)
        cuenta=cuenta+1
        freqs.append (f_mode)
        if mesh_comm.rank == model_rank:
            print (f'index: {i}, frequency: {f_mode}')
        if cuenta>5:  # Solo toma los primeros 5 modos
            break
# Selecciona el modo de interés y calcula el campo electromagnético asociado
isol = 6
lr = (esolver.getEigenvalue(isol)).real
autov = np.sqrt(lr.real)
f_mode=autov*cte.c/(2*np.pi)
inv_omega = 1/(2*np.pi*f_mode) 


# Interpolación y escalado de los campos eléctricos y magnéticos
V_dg = functionspace (mesh, ("DG", 0, (3,)))
u_out = Function (V_dg)
B_out = Function(V_dg)
u_save = Function(V)
E = Function(V)

e_val = esolver.getEigenpair(isol, u_save.x.petsc_vec)
esolver.getEigenvector(isol, E.x.petsc_vec)

B = TrialFunction(V)
q = TestFunction(V)
f = fem.Constant(mesh, default_scalar_type(inv_omega))
a = inner(q,B) * dx
L = inner(q,curl(E)) * f * dx

problem = LinearProblem(a, L, bcs=[bc])
B_save = problem.solve()

ue = Function(V)
um = Function(V)
fe = fem.Constant(mesh, default_scalar_type(0.5*eps))
fm = fem.Constant(mesh, default_scalar_type(0.5/mu))
ue = fe * (u_save)**2
um = fm * (B_save)**2


Ue = fem.assemble_scalar(fem.form(ue * dx))
Um = fem.assemble_scalar(fem.form(um * dx))
k = np.sqrt(7.9*1e-6/Ue)

B_save.x.scatter_forward()
B_out.interpolate(B_save)
B_out.x.array[:] *= k 
B_out.name = f"B (f = {f_mode*1e-9} GHz)"


u_out.interpolate(u_save)
u_out.x.array[:] *= k 
u_out.name = f"E (f = {f_mode*1e-9} GHz)"

u_out.x.scatter_forward()
B_out.x.scatter_forward()

# Cálculo de la energía electromagnética total en cada punto de la malla
V_scalar = functionspace(mesh, ("DG", 0))
E_array = u_out.x.array.reshape((-1, 3))
B_array = B_out.x.array.reshape((-1, 3))
E_sq = np.sum(E_array**2, axis=1)
B_sq = np.sum(B_array**2, axis=1)
# Crear función de energía electromagnética
uem = fem.Function(V_scalar)
uem.name = f"Energía EM (f = {f_mode*1e-9} GHz)"
uem.x.array[:] = 0.5 * eps * E_sq + 0.5 / mu * B_sq
uem.x.scatter_forward()


# Guardado de la malla y los campos calculados en archivos XDMF y NPY para su posterior análisis
with XDMFFile(MPI.COMM_WORLD, "plasma_fields_graf.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function (u_out)
    xdmf.write_function (B_out)
    xdmf.write_function(uem)



E_field = u_out.x.array.reshape((-1, 3))
np.save("E_dg_graf.npy", E_field)


u_values = u_save.x.array  * k
np.save("u_save_graf.npy", u_values)
