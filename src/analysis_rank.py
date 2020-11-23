import os
import numpy as np
import pandas as pd
import argparse

import matplotlib
import matplotlib.pyplot as plt

# Get rid of Type 3 font which is not allowed for ICPR 2020
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams.update({'hatch.color': '#FFFFFF'})
PATH_LOGS = '../results/logs'
PATH_FIGURE = '../results/figure'

my_colors = {
    'colors':         '#000000',  # black
    'bw':             '#777777',  # grey
    'hog':            '#984EA3',  # purple
    'vgg19':          '#FF7F00',  # orange
    'resnet18':       '#E41A1C',  # red
    'densenet121':    '#4DAF4A',  # green
    'efficientnetB0': '#377EB8',  # blue
    'osnetAINMarket': '#F781BF',  # pink
    'vehReid':        '#F781BF'   # pink
}

dict_feature = {
    'bw':             'GR',
    'colors':         'RGB',
    'hog':            'HOG',
    'vgg19':          'VGG',
    'resnet18':       'RSN',
    'densenet121':    'DNS',
    'efficientnetB0': 'EFF',
    'osnetAINMarket': 'OSN',
    'vehReid':        'VID'
}

dict_affinity = {
    'manhattan':     'L1',
    'euclidean':     'L2',
    'rank1':         'R1',
    'bhattacharyya': 'B',
    'cosine':        'C'
}

if __name__ == '__main__':

    # Parser
    parser = argparse.ArgumentParser(description="Ranking of visual features x affinity measures")
    parser.add_argument('--dataset', type=str, default='WildTrack', choices=['WildTrack', 'DETRAC', 'MOT17', 'UAVDT'],
                        help="Multiple object tracking dataset")
    args = parser.parse_args()

    df_logs = pd.read_csv(os.path.join(PATH_LOGS, 'macro_logs.csv'), dtype={'scene':str})

    # Parameters
    N = 5  # minimal number of barplots
    list_sigma = [0, 0.05, 0.10, 0.20]
    list_step = [1, 2, 4, 8, 16, 32]

    n_sigma = len(list_sigma)
    n_step = len(list_step)

    # Filter on the dataset, the classes and the affinities
    df_logs = df_logs[(df_logs['dataset'] == args.dataset)]
    df_logs = df_logs.loc[(df_logs['sigma'].isin(list_sigma)) & (df_logs['step'].isin(list_step))]
    df_logs = df_logs.loc[df_logs['affinity'].isin(list(dict_affinity.keys()))]
    df_logs = df_logs.reset_index(drop=True)

    # Number of scenes
    n_scenes = len(np.unique(df_logs['scene']))
    print(f"Number of scenes : {n_scenes}")
    assert n_scenes != 0, f"Run main.py on the dataset {args.dataset} to get some data"

    # mAP over scenes and ranking
    df_ranks = df_logs.groupby(['sigma', 'step', 'feature', 'affinity'])['precision'].agg(['mean']).rename(columns={"mean": "precision"})
    df_ranks = df_ranks.reset_index()
    df_ranks['name'] = df_ranks['feature'].map(dict_feature) + '-' + df_ranks['affinity'].map(dict_affinity)
    df_ranks['rank'] = df_ranks.groupby(["sigma", 'step'])['precision'].rank(ascending=False)

    fig, axs = plt.subplots(n_sigma, n_step, figsize=(14, 8))
    fig.suptitle(f"Average precision on {args.dataset}", y=1.02)
    fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"

    for idx in range(n_sigma):
        for idy in range(n_step):

            # Filter on this specific case sigma-step
            df_case = df_ranks.loc[(df_ranks['sigma'] == list_sigma[idx]) & (df_ranks['step'] == list_step[idy])]

            # Take the N best pairs feature-affinity
            df_best = df_case.loc[df_ranks['rank'] <= N]

            # If none of colors or bw is in the N best, we add the best one among them (same for HOG, CNN-based and ReID models)

            if ('bw' not in df_best['feature'].values) & ('colors' not in df_best['feature'].values):

                # Find the best color or bw
                df_best_histo = df_case.loc[df_case['feature'].isin(['colors', 'bw'])].nlargest(1, 'precision')
                df_best = pd.concat([df_best, df_best_histo])

            if 'hog' not in df_best['feature'].values:

                # Find the best hog
                df_best_hog = df_case.loc[df_case['feature'].isin(['hog'])].nlargest(1, 'precision')
                df_best = pd.concat([df_best, df_best_hog])

            if ('vgg19' not in df_best['feature'].values) & ('resnet18' not in df_best['feature'].values) & ('densenet121' not in df_best['feature'].values) & ('efficientnetB0' not in df_best['feature'].values):

                # Find the best cnn-based model
                df_best_cnn = df_case.loc[df_case['feature'].isin(['vgg19', 'resnet18', 'densenet121', 'efficientnetB0'])].nlargest(1, 'precision')
                df_best = pd.concat([df_best, df_best_cnn])

            if (args.dataset in ['MOT17', 'WildTrack']) and ('osnetAINMarket' not in df_best['feature'].values):

                # Find the results from ReID osnetAINMarket
                df_best_osnet = df_case.loc[df_case['feature'].isin(['osnetAINMarket'])].nlargest(1, 'precision')
                df_best = pd.concat([df_best, df_best_osnet])

            if (args.dataset in ['DETRAC', 'UAVDT']) and ('vehReid' not in df_best['feature'].values):

                # Find the results from ReID vehReid
                df_best_vehreid = df_case.loc[df_case['feature'].isin(['vehReid'])].nlargest(1, 'precision')
                df_best = pd.concat([df_best, df_best_vehreid])

            df_best = df_best.sort_values('precision', ascending=True).reset_index(drop=True)

            n_bars = df_best.shape[0]  # number of bars : 5 or more
            df_best.plot.barh(ax=axs[idx, idy], y='precision', x='name', alpha=0.99,
                              legend=None, color=[my_colors.get(x) for x in df_best['feature']])

            # Bug of matplotlib under Linux : hatches are not displayed after pdf rendering
            # Solution : use alpha=0.99 (https://stackoverflow.com/questions/5195466)

            print(df_best)

            for i, bar in enumerate(axs[idx, idy].patches):
                if df_best.loc[i, 'affinity'] == 'manhattan':
                    bar.set_hatch("\\\\\\")
                elif df_best.loc[i, 'affinity'] == 'euclidean':
                    bar.set_hatch('///')
                elif df_best.loc[i, 'affinity'] == 'bhattacharyya':
                    bar.set_hatch('XXX')
                elif df_best.loc[i, 'affinity'] == 'rank1':
                    bar.set_hatch('OO')
                elif df_best.loc[i, 'affinity'] == 'dotProduct':
                    bar.set_hatch('|||')

            # Add position of the +6th bar if necessary
            y_pos = -0.25
            for i, bar in enumerate(axs[idx, idy].patches):
                rank = df_best.loc[i, 'rank']
                if rank > N:  # it is an additional one
                    precision = df_best.loc[i, 'precision']
                    if (rank > 20) and (rank % 10 == 1):
                        axs[idx, idy].text(precision+5, y_pos, f"{rank:.0f}"r'$^{st}$', fontweight='bold')
                    elif (rank > 20) and (rank % 10 == 2):
                        axs[idx, idy].text(precision+5, y_pos, f"{rank:.0f}"r'$^{nd}$', fontweight='bold')
                    elif (rank > 20) and (rank % 10 == 3):
                        axs[idx, idy].text(precision+5, y_pos, f"{rank:.0f}"r'$^{rd}$', fontweight='bold')
                    else:
                        axs[idx, idy].text(precision+5, y_pos, f"{rank:.0f}"r'$^{th}$', fontweight='bold')
                    y_pos += 1

            axs[idx, idy].yaxis.set_label_text("")  # remove axis name
            axs[idx, idy].tick_params(axis="y", labelsize=9)
            axs[idx, idy].set(xlim=(0, 125))

            if idy == 0:  # first column
                axs[idx, idy].yaxis.set_label_text(list_sigma[idx], fontweight='bold')
            if idx == (n_sigma-1):  # last row
                axs[idx, idy].xaxis.set_label_text(list_step[idy], fontweight='bold')

    # Set common labels
    fig.text(0.5, -0.02, 'sampling step', ha='center', va='center')
    fig.text(-0.02, 0.5, 'sigma', ha='center', va='center', rotation='vertical')

    plt.savefig(os.path.join(PATH_FIGURE, args.dataset + '_ranks.pdf'), bbox_inches='tight', pad_inches=0.07)
