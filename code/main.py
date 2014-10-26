#!/usr/bin/env python
"""
Main script to identify patterns using Spectracl Clustering.
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import glob
import logging
import numpy as np
import os
import pickle
import time

import segmenter

bpm_dict = {"wtc2f20-poly": 84,
            "sonata01-3-poly": 118,
            "mazurka24-4-poly": 138,
            "silverswan-poly": 54,
            "sonata04-2-poly": 120
            }


def load_features(wav_file):
    """Loads the audio features (or reads them if already precomputed)."""
    print '- ', os.path.basename(wav_file)
    features_file = wav_file.replace(".wav", "-feats.pk")
    if not os.path.isfile(features_file):
        X, beats = segmenter.features(wav_file)
        with open(features_file, "w") as f:
            pickle.dump((X, beats), f)
    else:
        with open(features_file, "r") as f:
            X, beats = pickle.load(f)
    return X, beats


def bounds_to_patterns(bounds, labels):
    """Converts boudnaries to patterns and their occurrences."""
    maxK = np.max(labels)
    bounds_inters = zip(bounds[:-1], bounds[1:])

    patterns = []
    for k in np.arange(maxK + 1):
        occ_idxs = np.where(labels == k)[0]

        # We want at least one repetition
        if len(occ_idxs) >= 2:
            occs = []
            for occ_idx in occ_idxs:
                occs.append(bounds_inters[occ_idx])
            patterns.append(occs)
    return patterns


def output_patterns(all_patterns):
    """TODO."""
    for patterns in all_patterns:
        for i, pattern in enumerate(patterns):
            print "pattern %d" % i
            for j, occ in enumerate(pattern):
                print "occurrence %d" % j
                print occ


def process(in_dir, out_dir, start_layer=8, n_layers=3):
    """Main process."""
    # Get wav files
    wav_files = glob.glob(os.path.join(in_dir, "*.wav"))

    # Setup parameters
    parameters = {"verbose": False, "num_types": None}

    # Extract patters for all the wav files
    for wav_file in wav_files:
        logging.info("Extracting patterns for %s..." % wav_file)

        # Load the features
        X, beats = load_features(wav_file)

        all_patterns = []
        for layer in range(start_layer, start_layer + n_layers):
            parameters["num_types"] = layer
            bounds_idxs, labels = segmenter.do_segmentation(X, beats, parameters)
            patterns = bounds_to_patterns(beats[bounds_idxs], labels)
            all_patterns.append(patterns)

        # Final patterns
        output_patterns(all_patterns)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
        "Identifies musical patters using the Spectral Clustering method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset dir")
    parser.add_argument("out_path",
                        action="store",
                        help="Output dir")
    parser.add_argument("-j",
                        action="store",
                        dest="n_jobs",
                        type=int,
                        help="Number of jobs (threads)",
                        default=4)
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, args.out_path)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))
