
import torch
import umap
import matplotlib.pyplot as plt
import seaborn as sns

ACTION_NAMES = ["Directions", "Discussion", "Eating", "Greeting", "Phoning", "Photo", "Posing", "Purchases",
                "Sitting", "SittingDown", "Smoking", "Waiting", "WalkDog", "Walking", "WalkTogether"]


def decode_embedding(config, model):
    decoder = model[1]
    decoder.eval()
    with torch.no_grad():
        samples = torch.randn(10, 30).to(config.device)
        samples = decoder(samples)
        if '3D' in decoder.name:
            samples = samples.reshape([-1, 16, 3])
        elif 'RGB' in decoder.name:
            samples = samples.reshape([-1, 256, 256])
        # TODO save as images to tensorboard


def plot_umap(zs, metrics):
    """plot UMAP

    Arguments:
        zs {list} -- list of latent embeddings of 2D poses (/images)
        metrics {list} -- metrics to color the embeddings say, actions
    """
    print("[INFO]: UMAP reducing ", [*zs.shape])
    n_neighbors = 50
    min_dist = 0.1
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        metric='euclidean')
    embedding = reducer.fit_transform(zs)
    print('[INFO]: Embedding shape ', embedding.shape)
    # sns.color_palette("hls", 30)
    colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#ffd8b1', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#a9a9a9', '#469990', '#dcbeff', '#aaffc3', '#f58231', '#fabed4'
    ]
    sns.set_palette(sns.color_palette(colors))

    sns.scatterplot(x=embedding[:, 0],
                    y=embedding[:, 1],
                    hue=[ACTION_NAMES[int(x-2)] for x in metrics.tolist()],
                    # palette="Set2",
                    alpha=1)

    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., fontsize=20)
    plt.axis('tight')
    plt.xticks([]),plt.yticks([])

    # plt.gca().set_aspect( 'datalim') #'equal',
    # plt.title(f'2D Pose Embeddings in 2D Space  {[*zs.shape]}', fontsize=45)
    # plt.show()
    import os
    plt.savefig(f"{os.getenv('HOME')}/lab/HPE3D/src/results/latent_space_{n_neighbors}_{min_dist}.pdf", bbox_inches='tight', dpi=300, format='pdf')
