import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import palettable as pt


##### Make panel plots for various progress reports using a generic function
def make_panels(df, plotkey, filename, days=True, query='', grid=(3,3)):

    dfp = df.query(query)
    if len(dfp['Week Ending'].dropna()) != 0:
        dfp.index = pd.to_datetime(dfp['Week Ending'])
    else:
        dfp.index = pd.to_datetime(dfp['Year'], format='%Y')

    fig, axarr = plt.subplots(grid[0], grid[1], figsize=(5,5), dpi=200)
    axarr = axarr.flatten()
    
    for i, (state, df_state) in enumerate(dfp.groupby('State')):
        ax = axarr[i]
        if i==0:
            linelist = []
            labellist = []
            
        ax.set_title(' '.join([y.capitalize() for y in state.split()]))
        for year, df_year in df_state.groupby(df_state.index.year):
            if days:
                p = ax.plot(df_year.index.dayofyear, df_year[plotkey])
            else:
                p = ax.plot(df_year.index.year, df_year[plotkey])
                
            if i==0:
                linelist.append(p[0])
                labellist.append(year)

    fig.tight_layout()
    lgd = fig.legend(linelist, labellist, 'center right', fontsize=8,
                     bbox_to_anchor=(1.2, 0.5))
    plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

# Planted area
df_area_corn = pd.read_csv('area_planted_all_corn.csv', sep=',', thousands=',')
df_area_soy = pd.read_csv('area_planted_all_soy.csv', sep=',', thousands=',')
df_area_wheat = pd.read_csv('area_planted_all_wheat.csv', sep=',', thousands=',')

# Progress
df_pp_corn = pd.read_csv('progress_corn.csv', sep=',', thousands=',')
df_pp_soy = pd.read_csv('progress_soy.csv', sep=',', thousands=',')
df_pp_wheat =  pd.read_csv('progress_wheat.csv', sep=',', thousands=',')

# Production and Yield
df_production = pd.read_csv('corn_soy_production.csv', sep=',', thousands=',')
df_yield = pd.read_csv('corn_soy_yield.csv', sep=',', thousands=',')

df_planted_soy = df_pp_soy.iloc[np.where(df_pp_soy["Data Item"] == "SOYBEANS - PROGRESS, MEASURED IN PCT PLANTED")[0],:]
df_planted_corn = df_pp_corn.iloc[np.where(df_pp_corn["Data Item"] == "CORN - PROGRESS, MEASURED IN PCT PLANTED")[0],:]


for week in range(16, 21):
    cov = []
    for year, dfp in df_planted_soy.query(f"Period == 'WEEK #{week}'").groupby("Year"):
        #print(year, dfp.Value.mean(), dfp.Value.std())
        cov.append(dfp.Value.std() / dfp.Value.mean())
    print(week, np.mean(cov))


# Quality
df_status_corn = pd.read_csv('quality_all_corn.csv', sep=',', thousands=',')

period = "YEAR"
start_year = 2010
end_year = 2016
query = f'(Period == "{period}") & (Year >= {start_year}) & (Year <= {end_year})'

cp = pt.colorbrewer.qualitative.Paired_10.mpl_colors

fig, axarr = plt.subplots(2, 2, figsize=(5, 6), dpi=300, squeeze=True)

axarr = axarr.flatten()

axarr[0].set_prop_cycle('color', cp)
llist = []
lablist = []
for this_state, dfg in df_area_corn.query(query).groupby('State'):
    llist.append(axarr[0].plot(dfg['Year'], dfg['Value'] / 1e6, '-', label=this_state)[0])
    lablist.append(' '.join([y.capitalize() for y in this_state.split()]))
axarr[0].set_title('Corn')
    
axarr[1].set_prop_cycle('color', cp)
for this_state, dfg in df_area_soy.query(query).groupby('State'):
    axarr[1].plot(dfg['Year'], dfg['Value'] / 1e6, '-', label=this_state)
axarr[1].set_title('Soybean')
axarr[1].yaxis.tick_right()

axarr[2].set_prop_cycle('color', cp)
for this_state, dfg in df_area_wheat.query(query).groupby('State'):
    axarr[2].plot(dfg['Year'], dfg['Value'] / 1e6, '-', label=this_state)
axarr[2].set_title('Wheat')

axarr[0].set_ylabel('Area planted [$10^6$ acres]')
axarr[2].set_ylabel('Area planted [$10^6$ acres]')

#axarr[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#                fontsize=8, ncol=3)

corn_sum = df_area_corn.query(query).groupby('Year').sum()['Value']
soy_sum = df_area_soy.query(query).groupby('Year').sum()['Value']
wheat_sum = df_area_wheat.query(query).groupby('Year').sum()['Value']
total_sum = corn_sum + soy_sum + wheat_sum

ax = axarr[3]

ax.set_prop_cycle('color', pt.colorbrewer.qualitative.Set2_4.mpl_colors)
ax.plot(corn_sum.index, 100 * corn_sum / corn_sum[2010], ':', label='Corn')
ax.plot(soy_sum.index, 100 * soy_sum / soy_sum[2010], '--', label='Soybean')
ax.plot(wheat_sum.index, 100 * wheat_sum / wheat_sum[2010], '-.', label='Wheat')
ax.plot(wheat_sum.index, 100 * total_sum / total_sum[2010], '-', label='All')

#ax.plot(wheat_sum.index, 100* np.ones(len(wheat_sum.index)), '--', color='grey')
ax.yaxis.tick_right()
ax.set_ylabel("in [%] of 2009")

ylims = ax.get_ylim()
ax.set_ylim(ylims[0], 1.05*ylims[1])
ax.legend(fontsize='8', ncol=1, labelspacing=0.25)
ax.set_title('All 9 states')

fig.legend(llist, lablist, 'lower center', ncol=3,
           bbox_to_anchor=(0.5, 0.05),
           fontsize=8,
           labelspacing=0.25)

#fig.subplots_adjust(bottom=0.15)
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)

plt.savefig('area_planted.pdf')
plt.close()


fig, axarr = plt.subplots(1, 2, dpi=200, figsize=(5,4))

for i, comm in enumerate(['CORN', 'SOYBEANS']):

    llist = []
    lablist = []
    ax = axarr[i]
    ax.set_prop_cycle('color', cp)
    query = f'(Period == "YEAR") & (Commodity == "{comm}") & (Year > 2009) & (Year < 2017)'

    for state, df_state in df_yield.query(query).groupby('State'):
        llist.append(ax.plot(df_state['Year'], df_state['Value'])[0])
        lablist.append(state)
        
fig.legend(llist, lablist, 'lower center', ncol=3,
           bbox_to_anchor=(0.5, 0.05),
           fontsize=8,
           labelspacing=0.25)

#fig.subplots_adjust(bottom=0.15)
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
plt.savefig('crop_yield.pdf', bbox_inches='tight')

make_panels(df_pp_corn[df_pp_corn['Data Item'] == "CORN - PROGRESS, MEASURED IN PCT PLANTED"],
            'Value', 'corn_progress.pdf',
            query='(Commodity == "CORN") & (Year > 2009) & (Year < 2017)')
make_panels(df_pp_soy[df_pp_soy['Data Item'] == "SOYBEANS - PROGRESS, MEASURED IN PCT PLANTED"],
            'Value', 'soy_progress.pdf',
            query='(Commodity == "SOYBEANS") & (Year > 2009) & (Year < 2017)')

make_panels(df_pp_wheat[df_pp_wheat['Data Item'] == "WHEAT, WINTER - PROGRESS, MEASURED IN PCT PLANTED"],
            'Value', 'winter_wheat_progress.pdf',
            query='(Commodity == "WHEAT") & (Year > 2009) & (Year < 2017)')
make_panels(df_pp_wheat[df_pp_wheat['Data Item'] == "WHEAT, SPRING, DURUM - PROGRESS, MEASURED IN PCT PLANTED"],
            'Value', 'durum_wheat_progress.pdf',
            query='(Commodity == "WHEAT") & (Year > 2009) & (Year < 2017)')
make_panels(df_pp_wheat[df_pp_wheat['Data Item'] == "WHEAT, SPRING, (EXCL DURUM) - PROGRESS, MEASURED IN PCT PLANTED"],
            'Value', 'spring_wheat_progress.pdf',
            query='(Commodity == "WHEAT") & (Year > 2009) & (Year < 2017)')

for status in ['VERY POOR', 'POOR', 'FAIR', 'GOOD', 'EXCELLENT']:
    make_panels(df_status_corn[df_status_corn['Data Item'] == f"CORN - CONDITION, MEASURED IN PCT {status}"],
                'Value', f'corn_status_{status}.pdf',
                query='(Commodity == "CORN") & (Year > 2009) & (Year < 2017)')

