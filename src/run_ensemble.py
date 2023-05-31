"""
Compares the four approaches,
computes the ensemble modelling (median of the results of NRP, DDRM, CLZs),
produces plots and maps.
"""

from scipy.stats import spearmanr
from matplotlib.colors import BoundaryNorm
from matplotlib.offsetbox import AnchoredText
from shapely.geometry import box, Polygon
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = 'raise'

# plt.rcParams.update({'font.size': 22})
plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1.0),  # white
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),  # white
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),  # transparent
})

output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)
figures_folder = os.path.join(output_folder, 'figures')
os.makedirs(figures_folder, exist_ok=True)
tables_folder = os.path.join(output_folder, 'tables')
os.makedirs(tables_folder, exist_ok=True)

europe = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
europe = europe[europe.continent == 'Europe']

# Load data

data_nrppc = pd.read_csv(
    os.path.join(
        'output',
        'natural_references_per_pedoclimate.csv'
    ),
    low_memory=False
)
data_nrppc.rename({'vref': 'vref_nrppc'}, axis=1, inplace=True)

data_ddrm = pd.read_csv(
    os.path.join(
        'output',
        'data_driven_reciprocal_modelling.csv'
    ),
    low_memory=False
)
data_ddrm.drop("toc", axis=1, inplace=True)
data_ddrm.rename({'vref': 'vref_ddrm'}, axis=1, inplace=True)

data_clz = pd.read_csv(
    os.path.join(
        'output',
        'carbon_landscape_zones.csv'
    ),
    low_memory=False
)
data_clz.drop("toc", axis=1, inplace=True)
data_clz.rename({'vref': 'vref_clz'}, axis=1, inplace=True)

data_maom = pd.read_csv(
    os.path.join(
        'output',
        'maom_capacity.csv'
    ),
    low_memory=False
)
data_maom.rename(
    {
        'vref_stock': 'vref_stock_maom',
        'deltastock': 'deltastock_maom'
    },
    axis=1,
    inplace=True
)

data = data_nrppc.merge(data_ddrm, on="POINT_ID", how="inner")
data = data.merge(data_clz, on="POINT_ID", how="inner")
data = data.merge(data_maom, on="POINT_ID", how="inner")

data.loc[:, 'geometry'] = gpd.GeoSeries.from_wkt(data['geometry'])
data = gpd.GeoDataFrame(data, geometry='geometry')
data = data.set_crs("EPSG:4326")

# Convert reference values to stock


def bulk_density(toc, sand, clay, depth=20):
    # Source: Hollis JM, Hannam J, Bellamy PH (2012) Empirically-derived pedotransfer functions
    # for predicting bulk density in European soils. European Journal of Soil Science, 63, 96–109.
    toc = toc / 10
    BD = 0.80806 + (0.823844 * np.exp(-0.27993*toc) +
                    (0.0014065*sand) - (0.0010299*clay))
    return BD


def coarse_volume_fraction(coarse_mass_fraction, BD, rho=2.6):
    return coarse_mass_fraction / ((rho / BD) + ((1 - rho/BD) * coarse_mass_fraction))


def toc2stock(toc, BD, coarse, depth=20):
    toc = toc / 10
    oc_stock = toc*BD*depth*(1-coarse)
    return oc_stock


data.loc[:, 'BD_approx'] = data.apply(
    lambda x: bulk_density(x.toc, x.sand, x.clay), axis=1)
data.loc[:, 'coarse_mass_fraction'] = data.coarse / 100
data.loc[:, 'coarse_volume_fraction_approx'] = data.apply(
    lambda x: coarse_volume_fraction(x.coarse_mass_fraction, x.BD_approx), axis=1)
data.loc[:, "toc_stock"] = data.apply(lambda x: toc2stock(
    x.toc, x.BD_approx, x.coarse_volume_fraction_approx), axis=1)
for suffix in ['nrppc', 'ddrm', 'clz']:

    data.loc[:, f'vref_stock_{suffix}'] = data.apply(lambda x: toc2stock(
        x[f'vref_{suffix}'], x.BD_approx, x.coarse_volume_fraction_approx), axis=1)

    data.loc[:, f"deltatoc_{suffix}"] = data[f'vref_{suffix}'] - data.toc
    data.loc[:, f"deltastock_{suffix}"] = data[f'vref_stock_{suffix}'] - \
        data.toc_stock

# Ensamble method: calculate median reference values
# (not considering MaOM capacity)

data.loc[:, 'vref_median'] = data.filter(
    regex='^vref(_nrppc|_ddrm|_clz)$', axis=1).median(axis=1)
data.loc[:, 'vref_stock_median'] = data.filter(
    regex='^vref_stock(_nrppc|_ddrm|_clz)$', axis=1).median(axis=1)
data.loc[:, 'deltatoc_median'] = data.filter(
    regex='^deltatoc(_nrppc|_ddrm|_clz)$', axis=1).median(axis=1)
data.loc[:, 'deltastock_median'] = data.filter(
    regex='^deltastock(_nrppc|_ddrm|_clz)$', axis=1).median(axis=1)


def get_intermediate_method(x):
    names = ['NRP', 'DDRM', 'CLZs']
    vrefs = x[['vref_nrppc', 'vref_ddrm', 'vref_clz']]
    intermediate = names[np.argsort(vrefs)[len(vrefs)//2]]
    return intermediate


data.loc[:, 'intermediate_method'] = data.apply(
    lambda x: get_intermediate_method(x), axis=1)

print("Intermediate method proportions:")
print(data['intermediate_method'].value_counts() / len(data))

# Figure 6E: map of intermediate methods

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

europe[europe.name != "Russia"].geometry.boundary.plot(
    ax=ax, linewidth=1, facecolor='gray', alpha=0.1, color='black')
ax.set_xlim(-15, 40)
ax.set_ylim(32, 75)

cmap = 'Dark2'
colors = plt.cm.get_cmap(cmap, 8)
cmap = ListedColormap([colors(i) for i in range(3)])

data.plot('intermediate_method', markersize=1, ax=ax, legend=True,
          cmap=cmap, categories=['NRP', 'DDRM', 'CLZs'])

left, bottom, width, height = [0.28, 0.7, 0.18, 0.15]
ax2 = fig.add_axes([left, bottom, width, height])
sns.histplot(data=data, x='intermediate_method', ax=ax2, shrink=.8)
ax2.set_xlabel('')
for i in range(3):
    ax2.patches[i].set_facecolor(colors(i))

plt.savefig(os.path.join(figures_folder, 'figure6E.png'),
            format='png', dpi=600)
plt.close()

# Analyze results

# Supplementary Figure 5: fraction of sites that have deltasoc < 0 per climate cluster

fig, axs = plt.subplots(2, 2, figsize=(15, 15))
groups = data.groupby(['Cluster_climate', 'Cluster_soil'])
tot = groups.POINT_ID.count()

methods = ['_nrppc', '_ddrm', '_clz', '_maom']
names = ['Natural references per pedoclimate',
         'Reciprocal modelling', 'Carbon landscape zones', 'MaOM capacity']
height = 0.7
for ax, method, name in zip(axs.flatten(), methods, names):
    negative = data[(data[f'deltastock{method}'] < 0)].groupby(
        ['Cluster_climate', 'Cluster_soil']).POINT_ID.count() / tot * tot  # hack to keep same labels as tot
    ax.barh(range(len(tot)), tot, color='white',
            edgecolor='gray', height=height, label='All')
    ax.barh(range(len(tot)), negative, color='slategray',
            edgecolor='k', height=height, label='ΔSOC < 0')
    for i in range(len(tot)):
        if negative.values[i] > 0:
            txt = f'{100*negative.values[i]/tot.values[i]:.0f}%'
        else:
            txt = '0%'
        ax.text(tot.values[i]+2, i+height/2, txt, color='slategray')
    ax.set_title(name)
    ax.set_yticks(range(len(tot)))
    ax.set_yticklabels([f'{cc}-{cs}' for (cc, cs) in tot.index])
    ax.set_xlabel('N. croplands')
    ax.set_xlim(0, 900)
    ax.invert_yaxis()
axs[0][0].set_ylabel('Pedoclimatic cluster')
axs[1][0].set_ylabel('Pedoclimatic cluster')
axs[0][0].legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'supplementaryfigure5.png'),
            format='png', dpi=600)
plt.close()

# Correlation between reference values


def correlate(df):
    corrcoeff, pvalue = spearmanr(df.values, nan_policy='omit')
    columns = df.columns
    corrcoeff = pd.DataFrame(corrcoeff, columns=columns, index=columns)
    pvalue = pd.DataFrame(pvalue, columns=columns, index=columns)
    return corrcoeff, pvalue


corrcoeff_stock, pvalue_stock = correlate(data.filter(
    regex='^vref_stock(_nrppc|_ddrm|_clz|_maom)$', axis=1))

corrcoeff_stock.to_csv(
    os.path.join(tables_folder, 'table2.csv')
)

# Maps


def plot_map(data, variable, file_path, variable_name=None, higher_better=True,
             annotate=True, annotate_sum=False, unit_measure=None,
             with_maom_capacity=False, vmin=None, vmax=None,
             color_resolution=None, ncolors=10, extend='both',
             xlim=(-15, 40), ylim=(32, 75), with_median=False,
             median_only=False):

    markersize = 2

    if vmin is None:
        if median_only:
            vmin = data.filter(
                regex=f'^{variable}(_median)$', axis=1).min().min()
        elif with_maom_capacity:
            vmin = data.filter(
                regex=f'^{variable}(_nrppc|_ddrm|_clz|_maom)$', axis=1).min().min()
        elif with_median:
            vmin = data.filter(
                regex=f'^{variable}(_nrppc|_ddrm|_clz|_median)$', axis=1).min().min()
        else:
            vmin = data.filter(
                regex=f'^{variable}(_nrppc|_ddrm|_clz)$', axis=1).min().min()

    if vmax is None:
        if median_only:
            vmax = data.filter(
                regex=f'^{variable}(_median)$', axis=1).max().max()
        elif with_maom_capacity:
            vmax = data.filter(
                regex=f'^{variable}(_nrppc|_ddrm|_clz|_maom)$', axis=1).max().max()
        elif with_median:
            vmin = data.filter(
                regex=f'^{variable}(_nrppc|_ddrm|_clz|_median)$', axis=1).max().max()
        else:
            vmax = data.filter(
                regex=f'^{variable}(_nrppc|_ddrm|_clz)$', axis=1).max().max()

    if color_resolution is None:
        bounds = np.linspace(vmin, vmax, ncolors)
    else:
        bounds = np.arange(vmin, vmax + color_resolution, color_resolution)

    norm = BoundaryNorm(boundaries=bounds, ncolors=256, extend=extend)

    if higher_better:
        cmap = 'rainbow_r'
    else:
        cmap = 'rainbow'

    def add_textbox(suffix, ax):
        # Annotate measn, std, sum

        median = data[f'{variable}{suffix}'].median()
        mean = data[f'{variable}{suffix}'].mean()
        std = data[f'{variable}{suffix}'].std()
        sum = data[f'{variable}{suffix}'].sum()
        box_text = f"Median: {median:.2f} {unit_measure}\nMean: {mean:.2f} {unit_measure}\nStd: {std:.2f} {unit_measure}"
        if annotate_sum:
            box_text += f"\nSum: {sum:.0f} {unit_measure}"
        text_box = AnchoredText(box_text, frameon=True,
                                loc=2, pad=0.5, prop=dict(fontsize=28))
        plt.setp(text_box.patch, facecolor='white', alpha=0.9)
        ax.add_artist(text_box)

    n_rows = 1
    if median_only:
        n_cols = 1
    elif with_maom_capacity or with_median:
        n_cols = 4
    else:
        n_cols = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 10 * n_rows))

    if median_only:
        data.plot(f'{variable}_median', markersize=markersize,
                  ax=axs, norm=norm, cmap=cmap)
        axs.set_title("Median", fontsize=28)

        if annotate:
            add_textbox('_median', axs)

        europe[europe.name != "Russia"].geometry.boundary.plot(
            ax=axs, linewidth=1, facecolor='gray', alpha=0.1, color='black')
        axs.set_xlim(xlim)
        axs.set_ylim(ylim)

    else:
        data.plot(f'{variable}_nrppc', markersize=markersize,
                  ax=axs[0], norm=norm, cmap=cmap)
        axs[0].set_title(
            "Natural references per pedoclimate", fontsize=28)

        data.plot(f'{variable}_ddrm', markersize=markersize,
                  ax=axs[1], norm=norm, cmap=cmap)
        axs[1].set_title("Data-driven reciprocal modeling", fontsize=28)

        data.plot(f'{variable}_clz', markersize=markersize,
                  ax=axs[2], norm=norm, cmap=cmap)
        axs[2].set_title(
            "Carbon Landscape Zones", fontsize=28)

        if annotate:
            add_textbox('_nrppc', axs[0])
            add_textbox('_ddrm', axs[1])
            add_textbox('_clz', axs[2])

        if with_maom_capacity:
            data.plot(f'{variable}_maom', markersize=markersize,
                      ax=axs[3], norm=norm, cmap=cmap)
            axs[3].set_title("MaOM capacity", fontsize=28)

            if annotate:
                add_textbox('_maom', axs[3])

        if with_median:
            data.plot(f'{variable}_median', markersize=markersize,
                      ax=axs[3], norm=norm, cmap=cmap)
            axs[3].set_title("Median", fontsize=28)

            if annotate:
                add_textbox('_median', axs[3])

        for i in range(n_cols):
            ax = axs[i]
            europe[europe.name != "Russia"].geometry.boundary.plot(
                ax=ax, linewidth=1, facecolor='gray', alpha=0.1, color='black')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    # Add colorbar
    if median_only:
        cax = fig.add_axes([0.93, 0.17, 0.02, 0.65])
    else:
        cax = fig.add_axes([0.95, 0.17, 0.02, 0.65])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)

    if variable_name is None:
        variable_name = variable

    if unit_measure is None:
        title = variable_name
    else:
        title = f'{variable_name} [{unit_measure}]'

    cbar.ax.yaxis.set_ticks(bounds)
    cbar.ax.yaxis.set_ticks_position('left')
    if median_only:
        cbar.ax.tick_params(labelsize=20)
        cbar.ax.yaxis.set_label_position('right')
        cbar.ax.set_ylabel(title, fontsize=22)
    else:
        cbar.ax.tick_params(labelsize=24)
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.set_ylabel(title, fontsize=26)

    plt.savefig(file_path, format='png', dpi=600)
    plt.close()


def plot_difference_map(data, variable, file_path, variable_name=None, higher_better=True,
                        annotate=True, unit_measure=None, vmin=None, vmax=None,
                        color_resolution=None, ncolors=10, extend='both',
                        xlim=(-15, 40), ylim=(32, 75)):

    markersize = 2

    data1 = data[f'{variable}_nrppc']
    data2 = data[f'{variable}_ddrm'].rename(
        {f'{variable}_ddrm': variable})
    data3 = data[f'{variable}_clz'].rename(
        {f'{variable}_ddrm': variable})

    datadiff12 = data1 - data2
    datadiff13 = data1 - data3
    datadiff23 = data2 - data3

    if vmin is None:
        vmin = pd.concat([datadiff12, datadiff13, datadiff23]).min().min()

    if vmax is None:
        vmax = pd.concat([datadiff12, datadiff13, datadiff23]).max().max()

    if color_resolution is None:
        bounds = np.linspace(vmin, vmax, ncolors)
    else:
        bounds = np.arange(vmin, vmax + color_resolution, color_resolution)

    norm = BoundaryNorm(boundaries=bounds, ncolors=256, extend=extend)

    if higher_better:
        cmap = 'bwr_r'
    else:
        cmap = 'bwr'

    def add_textbox(datadiff, ax):
        # Annotate mean and std

        median = datadiff[variable].median()
        mean = datadiff[variable].mean()
        std = datadiff[variable].std()
        box_text = f"Median: {median:.2f} {unit_measure}\nMean: {mean:.2f} {unit_measure}\nStd: {std:.2f} {unit_measure}"
        text_box = AnchoredText(box_text, frameon=True,
                                loc=2, pad=0.5, prop=dict(fontsize=28))
        plt.setp(text_box.patch, facecolor='white', alpha=0.9)
        ax.add_artist(text_box)

    datadiff12 = gpd.GeoDataFrame(
        datadiff12, columns=[variable], geometry=data.geometry)
    datadiff13 = gpd.GeoDataFrame(
        datadiff13, columns=[variable], geometry=data.geometry)
    datadiff23 = gpd.GeoDataFrame(
        datadiff23, columns=[variable], geometry=data.geometry)

    n_rows = 1
    n_cols = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 10 * n_rows))

    datadiff12.plot(variable, markersize=markersize,
                    ax=axs[0], norm=norm, cmap=cmap)
    axs[0].set_title(
        f"{variable_name}(NRP) - {variable_name}(DDRM)", fontsize=28)

    datadiff13.plot(variable, markersize=markersize,
                    ax=axs[1], norm=norm, cmap=cmap)
    axs[1].set_title(
        f"{variable_name}(NRP) - {variable_name}(CLZs)", fontsize=28)

    datadiff23.plot(variable, markersize=markersize,
                    ax=axs[2], norm=norm, cmap=cmap)
    axs[2].set_title(
        f"{variable_name}(DDRM) - {variable_name}(CLZs)", fontsize=28)

    if annotate:
        add_textbox(datadiff12, axs[0])
        add_textbox(datadiff13, axs[1])
        add_textbox(datadiff23, axs[2])

    for i in range(n_cols):
        ax = axs[i]
        europe[europe.name != "Russia"].geometry.boundary.plot(
            ax=ax, linewidth=1, facecolor='gray', alpha=0.1, color='black')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Add single colorbar
    cax = fig.add_axes([0.95, 0.17, 0.02, 0.65])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)

    if variable_name is None:
        variable_name = variable

    if unit_measure is None:
        title = f'{variable_name} difference'
    else:
        title = f'{variable_name} difference [{unit_measure}]'

    cbar.ax.yaxis.set_ticks(bounds)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.tick_params(labelsize=24)
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.set_ylabel(title, fontsize=26)

    plt.savefig(file_path, format='png', dpi=600)
    plt.close()


plot_map(
    data,
    'vref',
    os.path.join(figures_folder,
                 'figure3top.png'),
    variable_name='SOCref',
    higher_better=False,
    unit_measure='gC/kg',
    vmin=10,
    vmax=50,
    color_resolution=5
)

plot_map(
    data,
    'vref',
    os.path.join(figures_folder,
                 'figure6A.png'),
    variable_name='SOCref',
    higher_better=False,
    unit_measure='gC/kg',
    vmin=10,
    vmax=50,
    color_resolution=5,
    median_only=True
)

plot_difference_map(
    data,
    'vref',
    os.path.join(figures_folder,
                 'figure5.png'),
    variable_name='SOCref',
    higher_better=False,
    unit_measure='gC/kg',
    vmin=-20,
    vmax=20,
    color_resolution=5
)

plot_map(
    data,
    'deltatoc',
    os.path.join(figures_folder,
                 'figure4top.png'),
    variable_name='ΔSOC',
    higher_better=False,
    unit_measure='gC/kg',
    vmin=0,
    vmax=30,
    color_resolution=5
)

plot_map(
    data,
    'deltatoc',
    os.path.join(figures_folder,
                 'figure6C.png'),
    variable_name='ΔSOC',
    higher_better=False,
    unit_measure='gC/kg',
    vmin=0,
    vmax=30,
    color_resolution=5,
    median_only=True
)

plot_map(
    data,
    'vref_stock',
    os.path.join(figures_folder,
                 'figure3bottom.png'),
    variable_name='SOCref',
    higher_better=False,
    unit_measure='Mg/ha',
    with_maom_capacity=True,
    vmin=10,
    vmax=90,
    color_resolution=5
)

plot_map(
    data,
    'vref_stock',
    os.path.join(figures_folder,
                 'figure6B.png'),
    variable_name='SOCref',
    higher_better=False,
    unit_measure='Mg/ha',
    median_only=True,
    vmin=10,
    vmax=90,
    color_resolution=5
)

plot_map(
    data,
    'deltastock',
    os.path.join(figures_folder,
                 'figure4bottom.png'),
    variable_name='ΔSOC',
    higher_better=False,
    unit_measure='Mg/ha',
    with_maom_capacity=True,
    vmin=0,
    vmax=50,
    color_resolution=5
)

plot_map(
    data,
    'deltastock',
    os.path.join(figures_folder,
                 'figure6D.png'),
    variable_name='ΔSOC',
    higher_better=False,
    unit_measure='Mg/ha',
    median_only=True,
    vmin=0,
    vmax=50,
    color_resolution=5
)

# France potential in Mt (MaOM capacity)
print('France potential in Mt (MaOM capacity):', data.groupby(
    'Country').median()['deltastock_maom']['France'] * (239395 * 100) * 1e-6)

# Estimate European SOC storage

land_cover_data = pd.read_csv(
    os.path.join(
        'data', 'Main farm land use by NUTS 2 regions',
        'extract.csv'
    ),
    low_memory=False
)
land_cover_data = land_cover_data.set_index('Country')
land_cover_data = land_cover_data.apply(
    lambda x: x.str.replace('\xa0', '').astype(int))
cropland_ha = land_cover_data['Arable land'] + \
    land_cover_data['Permanent crops']

deltastock_country = data.groupby('Country').median().filter(
    regex='^deltastock(_nrppc|_ddrm|_clz|_maom|_median)$').multiply(cropland_ha, axis=0)
# Convert to Mt
deltastock_country = deltastock_country.multiply(
    1e-6).rename({c: c+'_Mt' for c in deltastock_country.columns}, axis=1).dropna()

print('Europe C storage potential (Gt of C):')
print(deltastock_country.sum().divide(1e3))

# Supplementary figure 7: map of black sands

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

europe[europe.name != "Russia"].geometry.boundary.plot(
    ax=ax, linewidth=1, facecolor='gray', alpha=0.1, color='black')
ax.set_xlim(-15, 40)
ax.set_ylim(32, 75)

cmap = 'Dark2'
colors = plt.cm.get_cmap(cmap, 8)
cmap = ListedColormap([colors(i) for i in range(3)])

data.plot(markersize=1, ax=ax, color='gray', alpha=0.1)
data[(data.sand > 80) & (data["C/N"] > 13)].plot(markersize=1,
                                                 ax=ax, color='black', label='Black sands')

ax.legend()
plt.savefig(os.path.join(figures_folder, 'supplementaryfigure7.png'),
            format='png', dpi=600)
plt.close()

print('N. black sands = ', data[(data.sand > 80)
      & (data["C/N"] > 13)].POINT_ID.count())

# Save results
data.to_csv(
    os.path.join(
        output_folder,
        'methods_comparison.csv'
    ),
    index=False
)
