# Requisitos de ambiente:
# For pip:
#     pip install -r requirements_batimetria.txt
# For conda:
#     conda env create -f environment_batimetria.yml
#     conda activate batimetria

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import median_filter, gaussian_filter
import rioxarray
from pyproj import Transformer
import cmcrameri
import os

# Garantir que o PROJ database está corretamente referenciado no Windows
# Substituição direta com o caminho confirmado
# pyproj já funcionando corretamente — linha abaixo não é mais necessária
# os.environ['PROJ_DATA'] = r"C:\Users\Luizemara\.conda\envs\batimetria\Library\share\proj"


# -------------------------------
# Modelos de Subsidência
# -------------------------------
def modelo_raiz(t, h0, k):
    return h0 + k * np.sqrt(t)

def modelo_linear(t, a, b):
    return a * t + b

def modelo_suave_np(t, t1, t2, h0, k, a, b):
    peso = np.clip((t - t1) / (t2 - t1), 0, 1)
    return (1 - peso) * modelo_raiz(t, h0, k) + peso * modelo_linear(t, a, b)

# -------------------------------
# Função para aplicar filtro por blocos manualmente
# -------------------------------
import psutil
from tqdm import tqdm

def filtro_mediana_simples(array, win=250, step=256):
    chunk = step
    mem = psutil.virtual_memory()
    if mem.available < 2e9:
        print("⚠️ Memória disponível insuficiente (<2GB). Reduzindo o tamanho do passo para economizar memória.")
        step = max(step // 2, 64)
        chunk = step

    ny, nx = array.shape
    resultado = np.full_like(array, np.nan, dtype=np.float32)

    total = (ny // chunk + 1) * (nx // chunk + 1)
    with tqdm(total=total, desc="Aplicando filtro de mediana") as pbar:
        for i in range(0, ny, chunk):
            for j in range(0, nx, chunk):
                mem = psutil.virtual_memory()
                print(f"🔍 Memória disponível: {mem.available / 1e9:.2f} GB | Usando passo: {step}")
                if mem.available < 2e9:
                    raise MemoryError("⛔ Memória insuficiente para continuar o filtro de mediana.")

                i0 = max(0, i - win // 2)
                i1 = min(ny, i + chunk + win // 2)
                j0 = max(0, j - win // 2)
                j1 = min(nx, j + chunk + win // 2)

                bloco = array[i0:i1, j0:j1]

                if np.isnan(bloco).all():
                    pbar.update(1)
                    continue

                bloco_filt = median_filter(np.nan_to_num(bloco, nan=0), size=win)

                di0 = i - i0
                dj0 = j - j0
                di1 = di0 + min(chunk, ny - i)
                dj1 = dj0 + min(chunk, nx - j)

                resultado[i:i+di1-di0, j:j+dj1-dj0] = bloco_filt[di0:di1, dj0:dj1]
                pbar.update(1)

    return resultado, win

# -------------------------------
# Processamento principal
# -------------------------------
def processar_batimetria(win_km=250, chunk_size=500):
    # Caminhos dos arquivos locais
    filename_bat = "BR-DTM-REV2024_VTR-2.grd"
    filename_age = "age.2020.1.GTS2012.2m.nc"

    # Abrir batimetria
    with rasterio.open(filename_bat) as src:
        x_min, y_min, x_max, y_max = src.bounds
        bat = src.read(1)
        transform = src.transform
        x = np.arange(bat.shape[1]) * transform.a + transform.c
        y = np.arange(bat.shape[0]) * transform.e + transform.f
        crs = src.crs
    if crs is None:
        print("⚠️  CRS não reconhecido do arquivo, definindo como 'EPSG:3395' (World Mercator)")
        crs = rasterio.crs.CRS.from_epsg(3395)

    # Evitar valores absurdos de batimetria, que podem vir do tipo de arquivo de entrada
    bat = np.where((bat > 10000) | (bat < -11000), np.nan, bat).astype(np.float32)

    # Transformar para lat/lon
    transformer = Transformer.from_crs("EPSG:3395", "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(x_min, y_min)
    lon_max, lat_max = transformer.transform(x_max, y_max)

    # Abrir e recortar idade
    idade = rioxarray.open_rasterio(filename_age)
    if idade.rio.crs is None:
        idade = idade.rio.write_crs("EPSG:4326")
    idade = idade.isel(band=0)
    idade_corte = idade.sel(x=slice(lon_min, lon_max), y=slice(lat_max, lat_min))
    idade_proj = idade_corte.rio.reproject("EPSG:3395").astype(np.float32)

    idade_interp = idade_proj.interp(y=y, x=x, method="nearest")

    # Parâmetros do modelo
    h0, k, a, b = -2650, -365, 32.6, -7593
    t1, t2 = 55, 65

    DOF = modelo_suave_np(idade_interp, t1, t2, h0, k, a, b)
    mask = np.isnan(idade_interp) | np.isnan(DOF)
    BAT_menos_DOF = np.where(mask, np.nan, bat - DOF)

    # Aplicar filtro de mediana simples (sem uso de dask)
    dx = abs(x[1] - x[0])
    win = int((win_km * 1000) / dx)
    step = 256

    BAT_median, win_median = filtro_mediana_simples(BAT_menos_DOF, win=win, step=chunk_size)

    # Aplicar filtro gaussiano
    from tqdm import tqdm
    print("⏳ Aplicando filtro gaussiano...")
    BAT_gauss = np.empty_like(BAT_median, dtype=np.float32)
    chunk = 500
    for i in tqdm(range(0, BAT_median.shape[0], chunk), desc="Filtro Gaussiano", leave=False):
        i_end = min(i + chunk, BAT_median.shape[0])
        BAT_gauss[i:i_end] = gaussian_filter(np.nan_to_num(BAT_median[i:i_end], nan=0), sigma=win * 0.1)
    print(" filtro finalizado.")

    # Calcular batimetria residual final
    BAT_gauss_masked = np.where(np.isnan(BAT_menos_DOF), np.nan, BAT_gauss)
    BAT_final = bat - BAT_gauss_masked

    # Salvar resultado final
    final_da = xr.DataArray(BAT_final, coords={"y": y, "x": x}, dims=("y", "x"))
    final_da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    from rasterio.crs import CRS
    crs_str = crs.to_wkt() if crs else None
    final_da.rio.write_crs(crs_str, inplace=True)
    final_da.rio.to_raster(f"BAT_final_MedWin{win}.tif")
    final_da.to_netcdf(f"BAT_final_MedWin{win}.nc")

    # Exportar CSV
    df = pd.DataFrame({
        "x": np.repeat(x, len(y)),
        "y": np.tile(y, len(x)),
        "BAT": bat.ravel(),
        "DOF": DOF.values.ravel(),
        "BAT_menos_DOF": BAT_menos_DOF.ravel(),
        "BAT_filtrado": BAT_gauss_masked.ravel(),
        "BAT_final": BAT_final.ravel()
    })
    df.to_csv(f"BAT_resultados_MedWin{win}.csv", index=False)

    # Plotar
    fig, axes = plt.subplots(5, 1, figsize=(10, 12))
    titulos = ["Bathymetry", "DOF", "Bathymetry - DOF", "Filtered Bathymetry - DOF (Win:{win}cells)", "Residual Bathymetry"]
    dados = [bat, DOF, BAT_menos_DOF, BAT_gauss_masked, BAT_final]

    for ax, img, title in zip(axes, dados, titulos):
        im = ax.imshow(img, extent=[x.min(), x.max(), y.min(), y.max()], origin='upper', cmap=cmcrameri.cm.batlowW)
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig("BAT_resultados.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Processamento de modelo de subsidência batimétrica")
    parser.add_argument("--win_km", type=int, default=250, help="Tamanho da janela do filtro em km")
    parser.add_argument("--chunk_size", type=int, default=500, help="Tamanho do bloco de processamento")
    args = parser.parse_args()

    processar_batimetria(win_km=args.win_km, chunk_size=args.chunk_size)
