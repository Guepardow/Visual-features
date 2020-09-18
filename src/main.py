import os
import time
import argparse

from dataset import *

PATH_LOGS = "../results/logs"


if __name__ == '__main__':

    # Parser
    parser = argparse.ArgumentParser(description="Visual features analysis for multiple object tracking")
    parser.add_argument('--dataset', type=str, default='WildTrack', choices=['WildTrack', 'DETRAC', 'MOT17', 'UAVDT'],
                        help="Multiple object tracking dataset")
    parser.add_argument('--scene', type=str, help="Scene's name")
    parser.add_argument('--sigma', type=int, default=0, help="Noise on bounding boxes coordinates")
    parser.add_argument('--feature', type=str, default='colors', choices=['bw', 'colors', 'hog', 'vgg19', 'resnet18', 'densenet121', 'efficientnetB0', 'osnetAINMarket', 'vehReid'],
                        help="Method used to compute features")

    args = parser.parse_args()

    # Load the boundind boxes associated to the dataset
    print(f"\nLoading dataset {args.dataset} on the scene {args.scene} ... ", end='', flush=True)
    boundingboxes = load_dataset(args.dataset, args.scene)
    print('Done')

    args.sigma = args.sigma / 100
    if args.sigma != 0:
        print(f"Adding a white Gaussian noise at sigma = {args.sigma} ... ", end='', flush=True)
        boundingboxes = boundingboxes.get_noisy(args.sigma)  # get noisy alter the true values
        print('Done')
    else:
        print(f"Not adding a white Gaussian noise ... OK")

    print(boundingboxes)

    # Computes the features vector for each bounding box for each frame of the dataset filtered on the scene
    print(f"Computes the features vectors of {args.feature} ... ", flush=True)
    boundingboxes = boundingboxes.compute_features(args.scene, args.feature)
    print('Done')

    # Computes the matching score
    list_affinity = ['manhattan', 'euclidean', 'rank1', 'cosine']
    if args.feature in ['colors', 'bw', 'hog']:
        list_affinity += ['bhattacharyya']

    for affinity in list_affinity:

        print()

        for step in [1, 2, 4, 8, 16, 32]:

            start_time = time.time()
            print(f"Computes matching score with {affinity} at a step at {step:02d} ... ", end='', flush=True)
            df_logs, counter, total = boundingboxes.get_matching_score(affinity, step)
            end_time = time.time()
            
            # Save the results of each bounding boxes
            path_logs = os.path.join(PATH_LOGS, args.dataset, args.scene, affinity)
            os.makedirs(path_logs, exist_ok=True)
            df_logs.to_csv(os.path.join(path_logs, f"logs_{args.feature}_{step}_{args.sigma}.csv"), index=False)
            
            # Compute global accuracy on this scene
            precision = 100 * counter/total
            print(f"Precision : {precision:.2f}% ({counter}/{total}) in {end_time - start_time:.3f}s")

            # Store the results in a file
            path_logs_file = os.path.join(PATH_LOGS, 'macro_logs.csv')
            df_macro_logs = pd.read_csv(path_logs_file)
            
            df_macro_logs = df_macro_logs.append({'dataset': args.dataset, 'scene': args.scene, 'sigma': args.sigma, 'feature': args.feature, 'affinity': affinity,
                                                  'step': step, 'precision': precision, 'counter': counter, 'total': total},
                                                 ignore_index=True)
                    
            df_macro_logs.to_csv(path_or_buf=path_logs_file, index=False, float_format='%.4f')