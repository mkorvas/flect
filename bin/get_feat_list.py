#!/usr/bin/env python
# coding=utf-8
#


"""
Listing trained model features with non-zero coeficitents.

Usage: ./get_feat_list.py model-file.gz output.txt
"""


import sys
import getopt
from flect.model import Model
from flect.logf import log_info
import codecs
from operator import itemgetter

__author__ = "Ondřej Dušek"
__date__ = "2013"


def get_features(model_file, output_file):
    """\
    """
    m = Model.load_from_file(model_file)
    labels = m.data_headers.get_attrib(m.class_attr).labels
    feats = m.vectorizer.get_feature_names()
    fh = codecs.open(output_file, 'w', 'UTF-8')
    for i, label in enumerate(labels):
        log_info('Enumerating features for label %d (\'%s\')' % (i, label))
        coefs = m.classifier.coef_[i]
        nonzero = [(f, c) for (f, c) in zip(feats, coefs) if c != 0]
        print('LABEL == %s' % label, file=fh)
        for f, c in sorted(nonzero, key=itemgetter(1), reverse=True):
            print('%s - %f' % (f, c), file=fh)
        print("\n\n", file=fh)
    fh.close()


def display_usage():
    """\
    Display program usage information.
    """
    print(__doc__, file=sys.stderr)


def main():
    """\
    Main application entry.
    """
    _, filenames = getopt.getopt(sys.argv[1:], '')
    if len(filenames) != 2:
        display_usage()
        sys.exit(1)
    get_features(*filenames)


if __name__ == '__main__':
    main()
