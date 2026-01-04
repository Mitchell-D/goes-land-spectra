"""
Methods for generating permutations that preseve locality per an arbitrary
euclidean distance metric.

adapted from emulate_era5_land.helpers
"""
import numpy as np
import numba as nb
from pathlib import Path
import pickle as pkl
from multiprocessing import Pool,current_process
from scipy.spatial import KDTree

config_labels = ["target_avg_dist", "roll_threshold", "threshold_diminish",
        "recycle_count", "seed", "dynamic_roll_threshold"]
config_labels_conv = ["dist_threshold", "jump_cap", "shuffle_frac", "seed"]
config_labels_fast = [
        "dist_threshold", "jump_cap", "shuffle_frac", "seed", "leaf_size"]

def permutations_from_configs(
        dataset_name, coords, configs, pkl_dir, mode="conv", seed=None,
        enum_start=0, max_iterations=64, pool_size=4096, return_stats=True,
        nworkers=1, batch_size=4096, batches_per_stat=1, kdt_workers=2,
        debug=True, debug_freq=500):
    """
    Generate a bunch of permutations given a list of parameter configurations

    if mode==conv, expects cofnigurations to be a list of 4-tuples:
        ("dist_threshold", "jump_cap", "shuffle_frac", "seed"),

    if mode==global, expects configurations to be a list of
        ("target_avg_dist", "roll_threshold", "threshold_diminish",
            "recycle_count", "seed", "dynamic_roll_threshold")

    :@param dataset_name: unique string for this coordinate aset
    :@param coords: 2d array (N,C) for N points with C coordinate axes
    :@param configs: list of tuple configs as specified above
    :@param pkl_dir: directory where generated pkls will be stored
    :@param mode: must be 'fast', 'conv', or 'global'

    :@param return_stats:

        -- ( only relevant in fast mode ) --
    :@param batch_size: Number of distance lookups to perform in parallel
        when evaluating in 'fast' mode
    :@param batches_per_stat: When return_stats is True and using 'fast' mode,
        this is the number of batches between recordings of the global
        permutation distance statistics. Defaults to every batch (1)
    :@param kdt_workers: number of threads to query the KDTree with

    :@param return: paths of permutation pkl files that were generated
    """
    out_paths = []
    ## Generate a bunch of permutations given different parameters
    if mode == "global":
        init_perm = np.arange(coords.shape[0])
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(init_perm)
        default_args = {
                "coord_array":coords,
                "initial_perm":init_perm,
                "max_iterations":max_iterations,
                #"max_iterations":3,
                "return_stats":return_stats,
                "debug":debug,
                }

        args = [{**dict(zip(config_labels,c)),**default_args} for c in configs]
        with Pool(nworkers) as pool:
            for i,(a,r) in enumerate(
                    pool.imap_unordered(mp_get_permutation,args),
                    enum_start):
                perm,stats = r
                r_perm = np.asarray(tuple(zip(*sorted(zip(
                    list(perm), range(len(perm))
                    ), key=lambda v:v[0])))[1])
                pkl_path = pkl_dir.joinpath(
                        f"permutation_{dataset_name}_cycle_{i:03}.pkl")
                pkl.dump((a, np.stack([perm,r_perm], axis=-1), stats),
                        pkl_path.open("wb"))
                out_paths.append(pkl_path)
                print(f"Generated {pkl_path.name}")
            enum_start += len(out_paths)

    ## Use convolutional method to generate some permutations
    elif mode == "conv":
        default_args = {
                "coord_array":coords,
                "return_stats":return_stats,
                "debug":debug,
                "debug_freq":debug_freq,
                "pool_size":pool_size,
                }
        args = [{
            **dict(zip(config_labels_conv,c)), **default_args
            } for c in configs]
        with Pool(nworkers) as pool:
            for i,(a,r) in enumerate(
                    pool.imap_unordered(mp_get_permutation_conv,args),
                    enum_start):
                perm,stats = r
                r_perm = np.asarray(tuple(zip(*sorted(zip(
                    list(perm), range(len(perm))
                    ), key=lambda v:v[0])))[1])
                pkl_path = pkl_dir.joinpath(
                    f"permutation_{dataset_name}_conv_{i:03}.pkl")
                pkl.dump((a, np.stack([perm,r_perm], axis=-1), stats),
                        pkl_path.open("wb"))
                print(f"Generated {pkl_path.name}")
                out_paths.append(pkl_path)

    ## Use fast method
    elif mode == "fast":
        default_args = {
                "coords":coords,
                "batch_size":batch_size,
                "return_stats":return_stats,
                "batch_size":batch_size,
                "batches_per_stat":batches_per_stat,
                "kdt_workers":kdt_workers,
                }
        ## config:
        ## "dist_threshold" "jump_cap", "shuffle_frac", "seed", "leaf_size"

        ## args:
        ## coords, dist_threshold, jump_cap=8, shuffle_frac=0.5,
        ## batch_size=4096, leaf_size=32, seed=None, nworkers=-1):

        args = [{
            **dict(zip(config_labels_fast,c)), **default_args
            } for c in configs]
        with Pool(nworkers) as pool:
            for i,(a,r) in enumerate(
                    pool.imap_unordered(mp_get_permutation_fast,args),
                    enum_start):
                perm,stats = r
                r_perm = np.asarray(tuple(zip(*sorted(zip(
                    list(perm), range(len(perm))
                    ), key=lambda v:v[0])))[1])
                pkl_path = pkl_dir.joinpath(
                    f"permutation_{dataset_name}_fast_{i:03}.pkl")
                pkl.dump((a, np.stack([perm,r_perm], axis=-1), stats),
                        pkl_path.open("wb"))
                print(f"Generated {pkl_path.name}")
                out_paths.append(pkl_path)

    else:
        raise ValueError(f"mode must be oneo f ['global', 'conv'], not {mode}")
    return out_paths

def get_permutation_inverse(perm:np.array):
    """
    Get a random permutation of the provided number of elements and its inverse

    :@param perm: (N,) integer array mapping original positions to new indeces
    """
    return np.asarray(tuple(zip(*sorted(zip(
        list(perm), range(len(perm))
        ), key=lambda v:v[0])))[1])

def get_permutation(coord_array, initial_perm=None, target_avg_dist=3,
        roll_threshold=.66, threshold_diminish=.01, recycle_count=2,
        dynamic_roll_threshold=False, max_iterations=None, seed=None,
        return_stats=False, debug=False):
    """
    Method for iteratively discovering a semi-random permutation that
    balances preserving the approximate spatial locality of coordinates while
    randomly shuffling indeces.

    The goal is for the permuted data to be partitioned into contiguous chunks
    in a way that simultaneously (1) minimizes the number of chunks needed
    to extract a global random subset of points and (2) minimizes the number
    of chunks needed to extract a subset of points within local bounds.

    This algorithm accomplishes that goal by ranking the distance of each
    permutation, and re-shuffling the most and least distant. Both are mutually
    shuffled together because otherwise the closest would converge on their
    original positions.

    :@param coord_array: (N,C) array of N data points located by C cartesian
        coordinate dimensions. These are the original positions of each point.
    :@param initial_perm: (N,) integer array capturing the first-guess
        permutation of the N points from their original position.
    :@param target_avg_distance: Mean cartesian distance in coordinate space
        below which the search stops. Set this to a level that maintains
        reasonable locality without restoring the original positions.
    :@param roll_threshold: Initial ratio of furthest points to reshuffle.
    :@param threshold_diminish: Decrease in ratio of reshuffled points / iter.
    :@param recycle_count: Number of closest points to include in reshuffling.
    :@param dynamic_roll_threshold: If True, the threshold of far-distance
        points to reshuffle is calculated as the ratio of the current mean
        distance to that given the initial_perm, with roll_threshold as an
        upper bound. Theoretically this should solve more throroughly, but
        is typically much slower to converge.
    :@param max_iterations: Max iterations allowed to discover a permutation
    :@param seed: Random seed for initial permutation and reshuffling.
    :@param return_stats: If True, a list of 2-tuples (mean_dist, stdev_dist)
        corresponding to the permutation from each iteration is returned
    :@param debug: If True, prints mean and stdev of distance each iteration.

    :@return: Array of the permutation, or 2-tuple (permutation, stats) if
        return_stats is True
    """
    ## establish the random number generator and initial index permutation
    rng = np.random.default_rng(seed=seed)
    if initial_perm is None:
        initial_perm = np.arange(coord_array.shape[0])
        rng.shuffle(initial_perm)
    tmp_perm = initial_perm

    init_avg_dist = None
    iter_count=0
    stats = []
    while True:
        ## determine the euclidean distance in coordinate space of each
        ## point's destination from its origin given the current permutation
        tmp_ond = np.asarray([
            (ixo,ixn,np.sum((coord_array[ixn]-coord_array[ixo])**2)**(1/2))
            for ixo,ixn in enumerate(tmp_perm)
            ])
        ## sort permutations by distance
        dsort_old_ix,dsort_new_ix,dsort_dist = map(
            np.asarray,zip(*sorted(tmp_ond, key=lambda ond:ond[-1])))
        avg_dist = np.average(dsort_dist)
        if debug:
            print(f"Distance Avg: {avg_dist:<6.3f} " + \
                f"Stdev: {np.std(dsort_dist):<6.3f}")
        if return_stats:
            stats.append((avg_dist, np.std(dsort_dist)))
        if init_avg_dist is None:
            init_avg_dist = avg_dist

        ## If dynamic roll threshold is requested, calculate it, using the
        ## user-provided roll_threshold as an upper bound
        if dynamic_roll_threshold:
            roll_threshold = min([avg_dist / init_avg_dist, roll_threshold])
        else:
            roll_threshold -= threshold_diminish

        ## return the current permutation if the roll threshold is fully
        ## diminished, the target average has been met, or out of iterations.
        if int(tmp_perm.size * roll_threshold) <= 0:
            break
        if avg_dist <= target_avg_dist:
            break
        if max_iterations != None:
            if iter_count > max_iterations:
                break
            else:
                iter_count += 1

        ## determine the threshold of most distant points to re=roll
        roll_range =  int(dsort_dist.size * roll_threshold)
        ## get and shuffle the indeces of the most and least distant points
        reperm_idxs = np.concatenate([
            np.arange(recycle_count), ## least-distant points to recycle
            np.arange(dsort_dist.size)[-roll_range:], ## most-distant points
            ], axis=0)
        rng.shuffle(reperm_idxs)

        ## develop the new permutation by remapping nearest and furthest points
        reperm = np.arange(dsort_dist.size)
        reperm[:recycle_count] = reperm_idxs[:recycle_count]
        reperm[-roll_range:] = reperm_idxs[-roll_range:]
        dsort_new_ix = dsort_new_ix[reperm]

        ## restore the modified permutation by sorting by initial indeces
        dsort_old_ix,tmp_perm = zip(*sorted(
            zip(dsort_old_ix,dsort_new_ix),
            key=lambda d:d[0]
            ))
        tmp_perm = np.asarray(tmp_perm, dtype=int)
    if return_stats:
        return tmp_perm, stats
    return tmp_perm

def conv_perm_step(ix_cur, m_valid, cur_perm, jump_count, coords, rng,
        jump_cap=8, subpool_size=None, dist_threshold=1.0, shuffle_frac=0.5):

    """
    Modifies cur_perm by shuffling the indices of pixels ix that are nearby
    ix_cur given Euclidean distance between coords[ix_cur] and coords[ix].

    The maximum number of times the destination index of a pixel can change
    is defined by jump_cap, and the number of times the destination of
    each element has already changed is tracked by jump_count.

    Returns 3-tuple (m_valid, cur_perm, jump_count) given the results
    of this iteration.

    :@param ix_cur: current index to permute around (int in [0,N)))
    :@param m_valid: boolean mask of valid indices in [0,N) that haven't
        yet reached their jump cap.
    :@param cur_perm: (N,) ints for the element destinations given the
        current permutation.
    :@param jump_count: (N,) array of integers counting the number of times
        that each pixel's destination index has been changed.
    :@param coords: (N,C) array of coords for each pixel.
    :@param jump_cap: int max number of jumps that are allowed.
    :@param subpool_size: maximum number of pixels to check for distance
        inclusion while choosing pixels to permute.
    :@param dist_threshold: maximum Euclidean distance to consider for mixing.
    :@param shuffle_frac: fraction of the mixable indices to shuffle.
    """
    ## Choose a subset of valid indices based on m_valid
    ix_valid = np.where(m_valid)[0]
    if subpool_size is None:
        subpool_size = ix_valid.size
    ix_subpool = rng.choice(
            ix_valid,
            size=min(subpool_size, ix_valid.size),
            replace=False
            )

    ## Unpermuted distances wrt chosen pixel and mix candidate mask
    dists = np.sum((coords[ix_subpool] - coords[ix_cur])**2, axis=1)

    ## Pixels in the subpool that are within the distance threshold
    m_mix = (dists < dist_threshold**2)
    ix_tomix = ix_subpool[m_mix]

    ## Determine number to mix given any shuffle fraction constraint
    num_mix = int(ix_tomix.size * shuffle_frac)

    ## If no mixable pixels were returned, ignore this index
    if num_mix <= 1:
        return m_valid,cur_perm,jump_count

    ## Initial indices' destinations to shuffle
    ix_unmixed = rng.choice(ix_tomix, size=num_mix, replace=False)

    ## Shuffled permutation destinations
    ix_mixed = np.copy(ix_unmixed)
    rng.shuffle(ix_mixed)
    cur_perm[ix_unmixed] = cur_perm[ix_mixed]

    ## Carry over the number of jumps for this pixel from previous position
    #new_jump_count = np.copy(jump_count)
    jump_count[ix_unmixed] = jump_count[ix_mixed] + 1

    ## Update m_valid, setting idxs at the jump cap to False
    m_valid[ix_unmixed[jump_count[ix_unmixed] > jump_cap]] = False

    return m_valid,cur_perm,jump_count


def get_permutation_conv(coord_array, dist_threshold, jump_cap, shuffle_frac,
        seed=None, return_stats=False, pool_size=None, debug=False,
        debug_freq=500):
    """
    Alt method for generating a locality preserving semi-random permutation

    :@param coord_array: (N,C) array of C coord configurations per N points
    :@param dist_threshold: Maximum distance to permute
    """
    rng = np.random.default_rng(seed)
    jump_count = np.zeros(coord_array.shape[0])
    ## index order wrt original array for mixing steps
    conv_order = np.arange(coord_array.shape[0])
    rng.shuffle(conv_order)

    stats = []
    ## start with identity permutation
    cur_perm = np.arange(coord_array.shape[0])
    m_valid = np.full(coord_array.shape[0], True)
    jump_count = np.full(coord_array.shape[0], 0)

    if debug:
        print(current_process().name,
            f"getting perm for {coord_array.shape}; " + \
            f"{dist_threshold=}, {jump_cap=}, {shuffle_frac=}")
    for c,ix in enumerate(conv_order):
        m_valid,cur_perm,jump_count = conv_perm_step(
                ix_cur=ix,
                m_valid=m_valid,
                cur_perm=cur_perm,
                jump_count=jump_count,
                coords=coord_array,
                rng=rng,
                jump_cap=jump_cap,
                subpool_size=pool_size,
                dist_threshold=dist_threshold,
                shuffle_frac=shuffle_frac,
                )

        if (debug and c%debug_freq==0) or return_stats:
            tmpd = np.sum((coord_array-coord_array[cur_perm])**2, axis=1)**(.5)

        if (debug and c%debuf_freq==0):
            print(f"proc {current_process().name} "
                f"{c} / {coord_array.shape[0]} " + \
                f"({c/coord_array.shape[0]:.3f}) " + \
                f"dist avg: {np.average(tmpd):<6.3f} " + \
                f"stddev: {np.std(tmpd):<6.3f} " + \
                f"{pool_size = }")
        if (return_stats and c%debuf_freq==0):
            stats.append((np.average(tmpd), np.std(tmpd)))

    if return_stats:
        return cur_perm,stats
    return cur_perm

def mp_get_permutation_global(args):
    return args,get_permutation_global(**args)

def mp_get_permutation_conv(args):
    return args,get_permutation_conv(**args)

def mp_get_permutation_fast(args):
    return args,get_permutation_fast(**args)


def get_permutation_fast(
        coords, dist_threshold, jump_cap=8, shuffle_frac=0.5,
        batch_size=4096, leaf_size=32, seed=None, kdt_workers=-1,
        return_stats=True, batches_per_stat=1):
    """
    Optimized for >2M points with batching and multi-core support.
    """
    rng = np.random.default_rng(seed)
    N = coords.shape[0]
    cur_perm = np.arange(N)
    jump_count = np.zeros(N, dtype=np.int32)
    m_valid = np.ones(N, dtype=bool)

    stats = []
    tree = KDTree(coords, leafsize=leaf_size)

    ## randomize index access order
    visit_order = rng.permutation(N)
    for bnum,ix_cur in enumerate(range(0, N, batch_size)):
        ## select a batch of indeces to use as the origin points
        ixs_batch = visit_order[ix_cur:ix_cur+batch_size]

        ## remove points that have already reached their jump cap.
        ixs_batch = ixs_batch[m_valid[ixs_batch]]
        if ixs_batch.size == 0:
            continue

        ## query the batch in parallel, return list of arrays of nearby indeces
        neighbors = tree.query_ball_point(
                coords[ixs_batch],
                r=dist_threshold,
                workers=kdt_workers
                )

        for i,ix_cur in enumerate(ixs_batch):
            ## Check if pixel crossed jump_cap from earlier swap in this batch
            if not m_valid[ix_cur]:
                continue

            ## swappable neighbors must be below the jump cap
            ix_nbr = np.array(neighbors[i])
            ix_tomix = ix_nbr[m_valid[ix_nbr]]

            ## no swappable pairs within distance threshold
            if ix_tomix.size < 2:
                continue

            ## if shuffle_frac, remove the requested fraction of mixable pixels
            num_mix = int(ix_tomix.size * shuffle_frac)
            if num_mix < 2:
                continue

            ## update the permutation and increment the jump counts
            ix_unmixed = rng.choice(ix_tomix, size=num_mix, replace=False)
            ix_mixed = rng.permutation(ix_unmixed)
            cur_perm[ix_unmixed] = cur_perm[ix_mixed]
            jump_count[ix_unmixed] += 1

            ## invalidate points that have exceeded the jump cap
            newly_invalid = ix_unmixed[jump_count[ix_unmixed] >= jump_cap]
            if newly_invalid.size > 0:
                m_valid[newly_invalid] = False
        if return_stats and (bnum%batches_per_stat==0):
            tmpd = np.sum((coords-coords[cur_perm])**2, axis=1)**(.5)
            stats.append((np.average(tmpd), np.std(tmpd)))
    if return_stats:
        return cur_perm,stats
    return cur_perm


## index pool masking rather than bool mask method
'''
@nb.njit ## no python jit compile
def conv_perm_step(
        ix_cur, ix_pool, cur_perm, jump_count, coords, rng,
        jump_cap=8, subpool_size=None):
    """
    Modifies cur_perm by shuffling the indices of pixels ix that are nearby
    ix_cur given euclidean distance between coords[ix_cur] and coords[ix].

    The maximum number of times the destination index of a pixel can change
    is defined by jump_cap, and the number of times the destination of
    each element has already changed is tracked by jump_count.

    Returns 3-tuple (ix_pool, cur_perm, jump_count) given the results
    of this iteration.

    :@param ix_cur: current index to permute around (int in [0,N)))
    :@param ix_pool: Integers in [0,N) that haven't yet reached their
        jump cap. As evaluation proceeds, this should become much more
        efficient than checking all elements against jump_cap
    :@param cur_perm: (N,) ints for the element destinations given the
        current permutation.
    :@param jump_count: (N,) array of integers counting the number of
        times that each pixel's destination index has been changed
    :@param coords: (N,C) array of coords for each pixel
    :@param jump_cap: int max number of jumps that are allowed.
    :@param pool_size: maximum number of pixels to check for distance
        inclusion while choosing pixels to permute.
    """
    ## choose a subset of them to check distances with
    if subpool_size is None:
        subpool_size = ix_pool.size
    ix_subpool = rng.choice(
            ix_pool,
            size=min(pool_size,ix_pool.size),
            replace=False
            )

    ## unpermuted distances wrt chosen pixel and mix candidate mask
    dists = np.sum((coords[ix_subpool]-coords[ix_cur])**2,axis=1)

    ## pixels in the subpool that are within the distance threshold
    m_mix = (dists < dist_threshold**2)
    ix_tomix = ix_subpool[m_mix]

    ## determine number to mix given any shuffle fraction constraint
    num_mix = int(ix_tomix.size*shuffle_frac)

    ## if no mixable pixels were returned, ignore this index.
    if not num_mix > 1:
        return  ix_pool,cur_perm,jump_count

    ## initial indeces destinations to shuffle
    ix_unmixed = rng.choice(ix_tomix, size=num_mix, replace=False)

    ## shuffled permutation destinations
    ix_mixed = np.copy(ix_unmixed)
    rng.shuffle(ix_mixed)
    cur_perm[ix_unmixed] = cur_perm[ix_mixed]

    ## carry over the number of jumps for this pixel from prev position
    new_jump_count = np.copy(jump_count)
    new_jump_count[ix_unmixed] = jump_count[ix_mixed] + 1

    ## exclude elements that met their jump cap from the index pool
    ix_pool_new = np.setdiff1d(
            ix_pool,
            np.where(new_jump_count[ix_unmixed]>jump_cap)[0],
            )
    return ix_pool_new,cur_perm,jump_count
'''
