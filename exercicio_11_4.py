#Resolução de exercicio da materia Sistema de Controle I
# Grupo:
#       Alysson de Jesus Alves
#       Gabriel Shoji Sasaki Budoia
#       Gerson Vinicius Rodrigues de Macedo
#       Paulo Henrique Torres e Silva

# Problema 4 (Secao 11.2, Nise) - sistema com realimentacao unitaria
#   G(s) = K(s+20)(s+25) / [s(s+6)(s+9)(s+14)]
# item a: achar K para 15% de ultrapassagem usando so o diagrama de Bode
# item b: simular e conferir se a aproximacao de segunda ordem valeu a pena

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# numerador e denominador de G(s), como listas de coeficientes de polinomio
num = np.poly([-20, -25])          # (s+20)(s+25)
den = np.poly([0, -6, -9, -14])    # s(s+6)(s+9)(s+14)

# passo 1: da ultrapassagem desejada para o fator de amortecimento (zeta)
UP = 0.15
zeta = -np.log(UP) / np.sqrt(np.pi**2 + np.log(UP)**2)

# passo 2: do zeta para a margem de fase que o sistema precisa ter
PM = np.degrees(np.arctan(2 * zeta / np.sqrt(-2 * zeta**2 + np.sqrt(1 + 4 * zeta**4))))

# passo 3: em vez de desenhar o diagrama de Bode a mao, calculamos G(jw)
# para uma faixa de frequencias e procuramos o ponto onde a fase bate
# com o valor que precisamos (-180 + PM)
w = np.logspace(-2, 3, 200000)               # faixa de frequencias, de 0.01 a 1000 rad/s
Gjw = np.polyval(num, 1j * w) / np.polyval(den, 1j * w)   # G(jw) para cada frequencia
fase = np.degrees(np.unwrap(np.angle(Gjw)))
# unwrap evita que a fase "salte" 360 graus de um ponto para o outro

i = np.argmin(np.abs(fase - (-180 + PM)))   # indice da frequencia mais proxima do alvo
w_pm = w[i]                                  # frequencia de margem de fase
mag_K1 = np.abs(Gjw[i])                      # magnitude de G nessa frequencia, com K=1

# passo 4: o ganho que falta para essa magnitude chegar a 0 dB (equivalente
# a "subir a curva de magnitude" no metodo manual do livro)
K = 1 / mag_K1

print(f"zeta = {zeta:.4f}")
print(f"margem de fase necessaria = {PM:.2f} graus")
print(f"frequencia de margem de fase = {w_pm:.3f} rad/s")
print(f"|G(jw)| com K=1, nessa frequencia = {mag_K1:.4f} ({20*np.log10(mag_K1):.2f} dB)")
print(f"K = {K:.3f}")

# passo 5: fechar a malha (realimentacao unitaria) com o K encontrado
# e simular a resposta a um degrau
num_ol = K * num
den_cl = np.polyadd(den, np.pad(num_ol, (len(den) - len(num_ol), 0)))
sys_cl = signal.TransferFunction(num_ol, den_cl)

t = np.linspace(0, 3, 20000)
t, y = signal.step(sys_cl, T=t)

overshoot = (y.max() - 1) * 100
print(f"ultrapassagem na simulacao = {overshoot:.2f} %")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].semilogx(w, 20 * np.log10(np.abs(np.polyval(num, 1j*w) / np.polyval(den, 1j*w))))
axs[0].axhline(-20*np.log10(K), color="gray", linestyle="--", linewidth=0.8)
axs[0].axvline(w_pm, color="gray", linestyle="--", linewidth=0.8)
axs[0].set_title("Magnitude de G(jw), K = 1")
axs[0].set_xlabel("rad/s")
axs[0].set_ylabel("dB")
axs[0].grid(True, which="both", linewidth=0.3)

axs[1].plot(t, y)
axs[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
axs[1].set_title(f"Resposta ao degrau, K = {K:.2f}")
axs[1].set_xlabel("t (s)")
axs[1].grid(True, linewidth=0.3)

plt.tight_layout()
plt.savefig("exercicio_11_4.png", dpi=150)