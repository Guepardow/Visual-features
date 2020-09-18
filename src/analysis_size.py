import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

PATH_LOGS = '../results/logs'
PATH_FIGURE = '../results/figure'

# Parameters
classes = 'all'
affinity = 'euclidean'
sigma, step = 0.2, 32

common_list_features = ['bw', 'colors', 'hog', 'vgg19', 'resnet18', 'densenet121', 'efficientnetB0']

dict_complete_name = {'bw': 'grayscale',
                      'colors': 'colors',
                      'hog': 'HOG',
                      'vgg19': 'VGG-19',
                      'resnet18': 'ResNet-18',
                      'densenet121': 'DenseNet-121',
                      'efficientnetB0': 'EfficientNet-B0',
                      'osnetAINMarket': 'OSNet-AIN',
                      'vehReid': 'Vehicle ReID'}

my_colors = {
    'colors':          '#000000',  # black
    'grayscale':       '#777777',  # grey
    'HOG':             '#984EA3',  # purple'
    'VGG-19':          '#FF7F00',  # orange
    'ResNet-18':       '#E41A1C',  # red
    'DenseNet-121':    '#4DAF4A',  # green
    'EfficientNet-B0': '#377EB8',  # blue
    'OSNet-AIN':       '#F781BF',  # pink
    'Vehicle ReID':    '#F781BF'   # pink
}

if __name__ == '__main__':

    # Parser
    parser = argparse.ArgumentParser(description="Effect of size on visual features x affinity measures")
    parser.add_argument('--dataset', type=str, default='WildTrack', choices=['WildTrack', 'DETRAC', 'MOT17', 'UAVDT'],
                        help="Multiple object tracking dataset")
    args = parser.parse_args()

    list_scene = os.listdir(os.path.join(PATH_LOGS, args.dataset))

    if args.dataset in ['DETRAC', 'UAVDT']:
        list_features = common_list_features + ['vehReid']
    elif args.dataset in ['WildTrack', 'MOT17']:
        list_features = common_list_features + ['osnetAINMarket']
    else:
        list_features = [None]

    df_logs_scene = pd.DataFrame()

    for scene in tqdm(list_scene):
        for feature in list_features:

            df_logs = pd.read_csv(os.path.join(PATH_LOGS, args.dataset, scene, 'all', affinity, f'logs_{feature}_{step}_{sigma:.1f}.csv'))
            df_logs['isCorrect'] = 100 * (df_logs['objectID1'] == df_logs['objectID2']).astype(int)
            df_logs['scene'] = scene
            df_logs['feature'] = feature
            df_logs['name'] = df_logs['feature'].map(dict_complete_name)

            df_logs_scene = pd.concat([df_logs_scene, df_logs])

    # Bins at dataset level
    bins_ = np.quantile(df_logs_scene['area'], q=np.arange(0, 1.1, 0.1))
    bins_[0] -= 1
    df_logs_scene['bins_area'] = df_logs_scene['area'].transform(lambda x: pd.cut(x, bins=bins_, labels=bins_[:-1]))
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    df_size = df_logs_scene.groupby(['bins_area', 'name'])['isCorrect'].agg(['mean'])
    df_size = df_size.reset_index()

    sns.lineplot(x='bins_area', y='mean', data=df_size, hue='name', marker='o', palette=my_colors)
    plt.ylabel("average precision")
    plt.xlabel("size of the query object (px²)")
    plt.title(f'Average precision with regard to BBs size on {args.dataset} at sigma-step = {sigma}-{step}')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])  # remove the legend 'name'

    plt.savefig(os.path.join(PATH_FIGURE, f'{args.dataset}_size.pdf'), bbox_inches='tight', pad_inches=0.07)
