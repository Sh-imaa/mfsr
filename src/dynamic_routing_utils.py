import numpy as np

def get_routing_clusters(c, limit=1e1):
    # get clusters
    c = np.array([x.item() for x in c])
    c_sorted = np.sort(-c)
    sorted_i = np.argsort(-np.array(c))
    ep = 1e-100
    c_sorted -= ep
    gaps = np.array([(c_sorted[i] - c_sorted[i + 1])/ c_sorted[i + 1] for i in range(len(c_sorted) - 1)])
    if gaps.max() > limit:
        # contains bad images
        for i, g in enumerate(gaps):
            if g > limit:
                good_images = sorted_i[:i + 1]
                clusters = np.zeros_like(sorted_i)
                clusters[good_images] = 1
                return sorted_i[i + 1:], good_images, clusters

    return [], sorted_i, np.ones_like(sorted_i)


def smooth_weights(c, limit=1e1, alpha=0.5):
    bad_images, good_images, _ = get_routing_clusters(c, limit=limit)
    c = np.array([x.item() for x in c])
    good_weights = c[good_images] 
    slope =
