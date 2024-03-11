"""
    Filename: 
    Description:
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: domhnallboyle@gmail.com
    Python Version: 3.6
"""
import datetime
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap

from processor import get_embedding

colormap = np.array([
    [76, 255, 0],
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
], dtype=np.float) / 255


def main():
    # load saved unlabelled embeddings
    with open('embeddings.pkl', 'rb') as f:
        embedding_data = pickle.load(f)
    start_times, end_times, embeddings = zip(*embedding_data)
    embeddings = np.asarray(embeddings)

    # get reference (labelled) embeddings
    references = []
    for audio_path in Path('reference').glob('*'):
        label = audio_path.name.split('.')[0]
        embedding = get_embedding(audio_path=audio_path)
        references.append([label, embedding])

    # fit transform umap projection using the unlabelled embeddings
    reducer = umap.UMAP()
    projected = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots()
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round', fc='w'),
                        arrowprops=dict(arrowstyle='->'))
    annot.set_visible(False)

    # plot unlabelled and labelled embeddings (projected)
    sc = plt.scatter(projected[:, 0], projected[:, 1])
    for i, (label, embedding) in enumerate(references):
        embedding = reducer.transform(embedding.reshape(1, -1))[0]
        plt.scatter(embedding[0], embedding[1], c=colormap[i], label=label)

    def update_annot(ind):
        index = ind['ind'][0]
        pos = sc.get_offsets()[index]
        annot.xy = pos

        start_time = int(start_times[index])
        end_time = int(end_times[index])

        start_time = str(datetime.timedelta(seconds=start_time))
        end_time = str(datetime.timedelta(seconds=end_time))

        print(index, start_time, end_time)
        annot.set_text(f'{start_time} - {end_time}')
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', hover)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # USAGE: python -W ignore analyse.py
    main()
