# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:16:08 2026

@author: Josg7
"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math as mt
from scipy.integrate import solve_ivp

# --- CONSTANTES ---
delta1, delta2, delta3, delta4 = 0,0,0,0
alpha = 1
beta = 2 

# --- MATRICES DE PAULI Y FUNCIONES AUXILIARES ---
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[ 0, 1],
              [ 1, 0]], dtype=complex)
I = np.eye(2, dtype=complex)

def kron4(m1, m2, m3, m4):
    return np.kron(m1, np.kron(m2, np.kron(m3, m4)))

# --- HAMILTONIANO ---
H0 = (delta1 / 2) * kron4(Z, I, I, I) + \
     (delta2 / 2) * kron4(I, Z, I, I) + \
     (delta3 / 2) * kron4(I, I, Z, I) + \
     (delta4 / 2) * kron4(I, I, I, Z)

H_alpha = alpha * kron4(Z, Z, I, I)
H_beta = beta * kron4(I, I, Z, Z)

H = H0 + H_alpha + H_beta

def von_neumann(t, rho_flat):
    rho_matrix = rho_flat.reshape(16, 16)
    d_rho_dt = -1j * (H @ rho_matrix - rho_matrix @ H)
    return d_rho_dt.flatten()

# --- FUNCIÓN DE CONCURRENCIA UNIFICADA (Con filtro de ruido) ---
def calcular_concurrencia(rho_red):
    rho_red = (rho_red + np.conj(rho_red.T)) / 2 # Simetrizar
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sy_sy = np.kron(sy, sy)
    
    rho_tilde = sy_sy @ np.conj(rho_red) @ sy_sy
    R_matrix = rho_red @ rho_tilde
    
    evals = np.linalg.eigvals(R_matrix)
    evals = np.clip(np.real(evals), 0, None) # Limpiar ruido numérico
    
    R = np.sqrt(evals)
    R = np.sort(R)[::-1]
    
    C = max(0.0, R[0] - R[1] - R[2] - R[3])
    return C

# =====================================================================
# SIMULACIÓN 1: ESTADO INICIAL A (Para pares 12, 14, 24, 34)
# =====================================================================
rho_0_A = np.zeros((16, 16), dtype=complex)
estados_activos_A = list(range(16))
for i in estados_activos_A:
    for j in estados_activos_A:
        rho_0_A[i, j] = 1/16

t_span_A = (0, mt.pi/2)
t_eval_A = np.linspace(0, mt.pi/2, 500)

print("Simulando Estado A (Pares 12, 14, 24, 34)...")
sol_A = solve_ivp(von_neumann, t_span_A, rho_0_A.flatten(), t_eval=t_eval_A,
                  method='RK45', rtol=1e-8, atol=1e-10)

# =====================================================================
# SIMULACIÓN 2: ESTADO INICIAL B (Para par 13)
# =====================================================================
rho_0_B = np.zeros((16, 16), dtype=complex)
estados_activos_B = [2, 3, 6, 7, 8, 9, 12, 13]
for i in estados_activos_B:
    for j in estados_activos_B:
        rho_0_B[i, j] = 1/8

t_span_B = (0, mt.pi/2)
t_eval_B = np.linspace(0, mt.pi/2, 500)

print("Simulando Estado B (Par 13)...")
sol_B = solve_ivp(von_neumann, t_span_B, rho_0_B.flatten(), t_eval=t_eval_B, 
                  method='RK45', rtol=1e-6, atol=1e-8, max_step=0.1)


# =====================================================================
# EXTRACCIÓN DE TRAZAS PARCIALES Y CÁLCULO DE CONCURRENCIAS
# =====================================================================
def extraer_concurrencias(tiempos, sol_y, tipo_traza):
    concurrencias = []
    for k in range(len(tiempos)):
        rho_t = sol_y[:, k].reshape((16, 16))
        rho_red = np.zeros((4, 4), dtype=complex)
        
        if tipo_traza == '12':
            rho_red[0,0] = rho_t[0,0] + rho_t[1,1] + rho_t[2,2] + rho_t[3,3]
            rho_red[1,1] = rho_t[4,4] + rho_t[5,5] + rho_t[6,6] + rho_t[7,7]
            rho_red[2,2] = rho_t[8,8] + rho_t[9,9] + rho_t[10,10] + rho_t[11,11]
            rho_red[3,3] = rho_t[12,12] + rho_t[13,13] + rho_t[14,14] + rho_t[15,15]
            rho_red[0,1] = rho_t[0,4] + rho_t[1,5] + rho_t[2,6] + rho_t[3,7]
            rho_red[0,2] = rho_t[0,8] + rho_t[1,9] + rho_t[2,10] + rho_t[3,11]
            rho_red[0,3] = rho_t[0,12] + rho_t[1,13] + rho_t[2,14] + rho_t[3,15]
            rho_red[1,2] = rho_t[4,8] + rho_t[5,9] + rho_t[6,10] + rho_t[7,11]
            rho_red[1,3] = rho_t[4,12] + rho_t[5,13] + rho_t[6,14] + rho_t[7,15]
            rho_red[2,3] = rho_t[8,12] + rho_t[9,13] + rho_t[10,14] + rho_t[11,15]
            
        elif tipo_traza == '14':
            rho_red[0,0] = rho_t[0,0] + rho_t[2,2] + rho_t[4,4] + rho_t[6,6]
            rho_red[1,1] = rho_t[1,1] + rho_t[3,3] + rho_t[5,5] + rho_t[7,7]
            rho_red[2,2] = rho_t[8,8] + rho_t[10,10] + rho_t[12,12] + rho_t[14,14]
            rho_red[3,3] = rho_t[9,9] + rho_t[11,11] + rho_t[13,13] + rho_t[15,15]
            rho_red[0,1] = rho_t[0,1] + rho_t[2,3] + rho_t[4,5] + rho_t[6,7]
            rho_red[0,2] = rho_t[0,8] + rho_t[2,10] + rho_t[4,12] + rho_t[6,14]
            rho_red[0,3] = rho_t[0,9] + rho_t[2,11] + rho_t[4,13] + rho_t[6,15]
            rho_red[1,2] = rho_t[1,8] + rho_t[3,10] + rho_t[5,12] + rho_t[7,14]
            rho_red[1,3] = rho_t[1,9] + rho_t[3,11] + rho_t[5,13] + rho_t[7,15]
            rho_red[2,3] = rho_t[8,9] + rho_t[10,11] + rho_t[12,13] + rho_t[14,15]
        
        elif tipo_traza == '23':
            rho_red[0,0] = rho_t[0,0] + rho_t[1,1] + rho_t[8,8] + rho_t[9,9]
            rho_red[1,1] = rho_t[2,2] + rho_t[3,3] + rho_t[10,10] + rho_t[11,11]
            rho_red[2,2] = rho_t[4,4] + rho_t[5,5] + rho_t[12,12] + rho_t[13,13]
            rho_red[3,3] = rho_t[6,6] + rho_t[7,7] + rho_t[14,14] + rho_t[15,15]
            rho_red[0,1] = rho_t[0,2] + rho_t[1,3] + rho_t[8,10] + rho_t[9,11]
            rho_red[0,2] = rho_t[0,4] + rho_t[1,5] + rho_t[8,12] + rho_t[9,13]
            rho_red[0,3] = rho_t[0,6] + rho_t[1,7] + rho_t[8,14] + rho_t[9,15]
            rho_red[1,2] = rho_t[2,4] + rho_t[3,5] + rho_t[10,12] + rho_t[11,13]
            rho_red[1,3] = rho_t[2,6] + rho_t[3,7] + rho_t[10,14] + rho_t[11,15]
            rho_red[2,3] = rho_t[4,6] + rho_t[5,7] + rho_t[12,14] + rho_t[13,15]
            
        elif tipo_traza == '24':
            rho_red[0,0] = rho_t[0,0] + rho_t[2,2] + rho_t[8,8] + rho_t[10,10]
            rho_red[1,1] = rho_t[1,1] + rho_t[3,3] + rho_t[9,9] + rho_t[11,11]
            rho_red[2,2] = rho_t[4,4] + rho_t[6,6] + rho_t[12,12] + rho_t[14,14]
            rho_red[3,3] = rho_t[5,5] + rho_t[7,7] + rho_t[13,13] + rho_t[15,15]
            rho_red[0,1] = rho_t[0,1] + rho_t[2,3] + rho_t[8,9] + rho_t[10,11]
            rho_red[0,2] = rho_t[0,4] + rho_t[2,6] + rho_t[8,12] + rho_t[10,14]
            rho_red[0,3] = rho_t[0,5] + rho_t[2,7] + rho_t[8,13] + rho_t[10,15]
            rho_red[1,2] = rho_t[1,4] + rho_t[3,6] + rho_t[9,12] + rho_t[11,14]
            rho_red[1,3] = rho_t[1,5] + rho_t[3,7] + rho_t[9,13] + rho_t[11,15]
            rho_red[2,3] = rho_t[4,5] + rho_t[6,7] + rho_t[12,13] + rho_t[14,15]
            
        elif tipo_traza == '34':
            rho_red[0,0] = rho_t[0,0] + rho_t[4,4] + rho_t[8,8] + rho_t[12,12]
            rho_red[1,1] = rho_t[1,1] + rho_t[5,5] + rho_t[9,9] + rho_t[13,13]
            rho_red[2,2] = rho_t[2,2] + rho_t[6,6] + rho_t[10,10] + rho_t[14,14]
            rho_red[3,3] = rho_t[3,3] + rho_t[7,7] + rho_t[11,11] + rho_t[15,15]
            rho_red[0,1] = rho_t[0,1] + rho_t[4,5] + rho_t[8,9] + rho_t[12,13]
            rho_red[0,2] = rho_t[0,2] + rho_t[4,6] + rho_t[8,10] + rho_t[12,14]
            rho_red[0,3] = rho_t[0,3] + rho_t[4,7] + rho_t[8,11] + rho_t[12,15]
            rho_red[1,2] = rho_t[1,2] + rho_t[5,6] + rho_t[9,10] + rho_t[13,14]
            rho_red[1,3] = rho_t[1,3] + rho_t[5,7] + rho_t[9,11] + rho_t[13,15]
            rho_red[2,3] = rho_t[2,3] + rho_t[6,7] + rho_t[10,11] + rho_t[14,15]

        elif tipo_traza == '13':
            rho_red[0,0] = rho_t[0,0] + rho_t[1,1] + rho_t[4,4] + rho_t[5,5]
            rho_red[1,1] = rho_t[2,2] + rho_t[3,3] + rho_t[6,6] + rho_t[7,7]
            rho_red[2,2] = rho_t[8,8] + rho_t[9,9] + rho_t[12,12] + rho_t[13,13]
            rho_red[3,3] = rho_t[10,10] + rho_t[11,11] + rho_t[14,14] + rho_t[15,15]
            rho_red[0,1] = rho_t[0,2] + rho_t[1,3] + rho_t[4,6] + rho_t[5,7]
            rho_red[0,2] = rho_t[0,8] + rho_t[1,9] + rho_t[4,12] + rho_t[5,13]
            rho_red[0,3] = rho_t[0,10] + rho_t[1,11] + rho_t[4,14] + rho_t[5,15]
            rho_red[1,2] = rho_t[2,8] + rho_t[3,9] + rho_t[6,12] + rho_t[7,13]
            rho_red[1,3] = rho_t[2,10] + rho_t[3,11] + rho_t[6,14] + rho_t[7,15]
            rho_red[2,3] = rho_t[8,10] + rho_t[9,11] + rho_t[12,14] + rho_t[13,15]

        # Simetrizar la parte inferior
        rho_red[1,0] = np.conj(rho_red[0,1])
        rho_red[2,0] = np.conj(rho_red[0,2])
        rho_red[3,0] = np.conj(rho_red[0,3])
        rho_red[2,1] = np.conj(rho_red[1,2])
        rho_red[3,1] = np.conj(rho_red[1,3])
        rho_red[3,2] = np.conj(rho_red[2,3])

        concurrencias.append(calcular_concurrencia(rho_red))
        
    return concurrencias

C12 = extraer_concurrencias(sol_A.t, sol_A.y, '12')
C14 = extraer_concurrencias(sol_A.t, sol_A.y, '14')
C24 = extraer_concurrencias(sol_A.t, sol_A.y, '24')
C34 = extraer_concurrencias(sol_A.t, sol_A.y, '34')
C23 = extraer_concurrencias(sol_A.t, sol_B.y, '23')
C13 = extraer_concurrencias(sol_B.t, sol_B.y, '13')


# Ahora puedes realizar la operación sin errores

# =====================================================================
# GRÁFICA UNIFICADA (UNA SOLA FIGURA)
# =====================================================================
plt.figure(figsize=(12, 7))
plt.title(r'$a)$ Evolución de la Concurrencia (Interacción $H_1$) ($\alpha$=0.5, $\beta$=0.5, $\Delta_i$=1.0)', fontsize=20)

# Graficamos todas las curvas en el mismo plano
plt.plot(sol_A.t, C12, label='Qubits 1 y 2', color='red', linewidth=2, linestyle='-')
plt.plot(sol_A.t, C14, label='Qubits 1 y 4', color='green', linewidth=2, linestyle='--')
plt.plot(sol_A.t, C23, label='Qubits 2 y 3', color='pink', linewidth=2, linestyle='--')
plt.plot(sol_A.t, C24, label='Qubits 2 y 4', color='purple', linewidth=2, linestyle='--')
plt.plot(sol_A.t, C34, label='Qubits 3 y 4', color='blue', linewidth=2, linestyle='--')


# El par 13 se gráfica con un estilo diferente para notar que es otra condición inicial
plt.plot(sol_B.t, C13, label='Qubits 1 y 3 (Condición de Bell)', color='black', linewidth=2.5, linestyle='-')

plt.xlabel('Tiempo $t$', fontsize=14)
plt.ylabel('Concurrencia', fontsize=14)
plt.xlim(0, mt.pi/2)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)

# Leyenda fuera del gráfico
plt.legend(loc='upper right', fontsize=11, bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.savefig("h2_a.1_b.1_d1.pdf", format='pdf', bbox_inches='tight')
plt.show()
