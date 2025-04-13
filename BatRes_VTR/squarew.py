# Criar uma onda quadrada com ruído, aplicando para janelas variadas
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square
from scipy.ndimage import median_filter, gaussian_filter

x = np.linspace(0, 1000, 3000)
onda_quadrada = square(2 * np.pi * x / 1000) * 50  # período de 60 unidades, amplitude 50
ruido = np.random.normal(0, 20, size=x.shape)
perfil_sq = onda_quadrada + ruido

# Aplicar filtros com janelas variadas
janela_medianas = [20, 50, 100]
sigmas_gaussianos = [20, 50, 100]

fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharex=True, sharey=True)

for i, (win, sigma) in enumerate(zip(janela_medianas, sigmas_gaussianos)):
    med = median_filter(perfil_sq, size=win)
    gau = gaussian_filter(perfil_sq, sigma=sigma)
    
    axs[i, 0].plot(x, perfil_sq, color='royalblue', alpha=0.5, label='Noisy wave')
    axs[i, 0].plot(x, med, color='darkblue', alpha=0.5, label=f'Median filter (win={win})')
    axs[i, 0].plot(x, onda_quadrada, color='black', lw=0.8, label='Pure wave')
    axs[i, 0].set_title(f"Median filter, window = {win}")
    axs[i, 0].legend()
    axs[i, 0].grid(True)
    
    axs[i, 1].plot(x, perfil_sq, color='green', alpha=0.5, label='Noisy wave')
    axs[i, 1].plot(x, gau, color='darkgreen', alpha=0.5, label=f'Gaussian filter (σ={sigma})')
    axs[i, 1].plot(x, onda_quadrada, color='black', lw=0.8, label='Pure wave')
    axs[i, 1].set_title(f"Gaussian filter, σ = {sigma}")
    axs[i, 1].legend()
    axs[i, 1].grid(True)

plt.suptitle("Median vs. Gaussian filter effect", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()