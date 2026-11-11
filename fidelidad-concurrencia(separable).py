# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:55:23 2026

@author: Josg7
"""

import numpy as np
import matplotlib.pyplot as plt
import math as mt
from scipy.integrate import solve_ivp

# --- CONSTANTES ---
delta1, delta2, delta3, delta4 = 0,0,0,0
alpha, beta = 1,5

# --- MATRICES DE PAULI ---
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

def kron4(m1, m2, m3, m4):
    return np.kron(m1, np.kron(m2, np.kron(m3, m4)))

# --- HAMILTONIANO ---
H0 = (delta1/2)*kron4(Z,I,I,I) + (delta2/2)*kron4(I,Z,I,I) + \
     (delta3/2)*kron4(I,I,Z,I) + (delta4/2)*kron4(I,I,I,Z)
H = H0 + alpha*kron4(Z,Z,I,I) + beta*kron4(I,I,Z,Z)

def von_neumann(t, rho_flat):
    rho_matrix = rho_flat.reshape(16, 16)
    d_rho_dt = -1j * (H @ rho_matrix - rho_matrix @ H)
    return d_rho_dt.flatten()

# --- FUNCIONES DE CÁLCULO ---

def calcular_valor_solicitado(rho, num_qubit):
    """ Calcula |2 * (rho_red^2)_{01}| """
    rho_q = np.zeros((2,2), dtype=complex)
    if num_qubit == 1:
        for i in range(8):
            rho_q[0,0] += rho[i, i]; rho_q[1,1] += rho[i+8, i+8]
            rho_q[0,1] += rho[i, i+8]; rho_q[1,0] += rho[i+8, i]
    elif num_qubit == 3:
        idx0 = [0, 1, 4, 5, 8, 9, 12, 13]  # Estados donde el Qubit 3 es |0>
        idx1 = [2, 3, 6, 7, 10, 11, 14, 15] # Estados donde el Qubit 3 es |1>
        
        for i, j in zip(idx0, idx1):
            rho_q[0,0] += rho[i, i]; rho_q[1,1] += rho[j, j]
            rho_q[0,1] += rho[i, j]; rho_q[1,0] += rho[j, i]
    
    rho_sq = rho_q @ rho_q
    return np.abs(2 * rho_sq[0, 1])

def calcular_concurrencia(rho_red):
    rho_red = (rho_red + np.conj(rho_red.T)) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sy_sy = np.kron(sy, sy)
    rho_tilde = sy_sy @ np.conj(rho_red) @ sy_sy
    evals = np.linalg.eigvals(rho_red @ rho_tilde)
    evals = np.sort(np.sqrt(np.clip(np.real(evals), 0, None)))[::-1]
    return max(0.0, evals[0] - evals[1] - evals[2] - evals[3])

def get_rho13(rho_t):
    rho_red = np.zeros((4, 4), dtype=complex)
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
    return rho_red + np.conj(rho_red.T) - np.diag(np.diag(rho_red))

# --- SIMULACIONES ---
t_span = (0, 6*mt.pi)
t_eval = np.linspace(0, 6*mt.pi, 10000)

# Estado A
rho_0_A = np.full((16,16), 1/16, dtype=complex)
sol_A = solve_ivp(von_neumann, t_span, rho_0_A.flatten(), t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8, max_step=0.1)

# Estado B
rho_0_B = np.zeros((16, 16), dtype=complex)
idx_B = [2, 3, 6, 7, 8, 9, 12, 13]
for i in idx_B:
    for j in idx_B: rho_0_B[i, j] = 1/8
sol_B = solve_ivp(von_neumann, t_span, rho_0_B.flatten(), t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8, max_step=0.1)

# --- PROCESAMIENTO CRUZADO ---
vals_custom_A = []
concurrencia_13_B = []

for k in range(len(t_eval)):
    # Qubits 1 y 3 individuales del Estado A
    rho_A_t = sol_A.y[:, k].reshape(16, 16)
    v1 = calcular_valor_solicitado(rho_A_t, 1)
    v3 = calcular_valor_solicitado(rho_A_t, 3)
    vals_custom_A.append(v1 * v3)
    
    # Par 1-3 del Estado B
    rho_B_t = sol_B.y[:, k].reshape(16, 16)
    concurrencia_13_B.append(calcular_concurrencia(get_rho13(rho_B_t)))

# --- GRÁFICA ---
plt.figure(figsize=(10, 6))
plt.plot(t_eval, vals_custom_A, label=r'$F(t)$', color='crimson')
plt.plot(t_eval, concurrencia_13_B, label=r'$C_{13}$', color='darkblue', linestyle='--')


plt.title(' Amplitud de Fidelidad F(t) y Concurrencia $(C_{13})$ (Sistema Separable)', fontsize=20)
plt.xlabel('Tiempo $t$')
plt.ylabel('Valor')
plt.xlim(0, mt.pi/2)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=11, bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.show()