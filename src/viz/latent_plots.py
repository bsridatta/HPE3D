
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
    reducer = umap.UMAP(n_neighbors=3,
                        min_dist=0.1,
                        metric='cosine')
    embedding = reducer.fit_transform(zs)
    print('[INFO]: Embedding shape ', embedding.shape)

    sns.scatterplot(x=embedding[:, 0],
                    y=embedding[:, 1],
                    hue=[ACTION_NAMES[int(x-2)] for x in metrics.tolist()],
                    palette="Set2",
                    alpha=0.6)

    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.axis('tight')
    # plt.gca().set_aspect( 'datalim') #'equal',
    plt.title(f'UMAP projection of Z  {[*zs.shape]}', fontsize=15)
    plt.show()
