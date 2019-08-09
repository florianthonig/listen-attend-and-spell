import os
import sys
import string
from argparse import ArgumentParser
import glob
from python_speech_features import mfcc
import numpy as np
import scipy.io.wavfile as wav

try:
    import speechpy
except:
    raise ImprtError('Run `pip install speechpy` first')

try:
    import soundfile as sf
except:
    raise ImportError('Run `pip install soundfile` first')

FRAME_LENGTH = 0.025
FRAME_SHIFT = 0.01
FEATURE_VECTOR_SIZE = 39

def compute_mfcc(audio_data, sample_rate):
    ''' Computes the mel-frequency cepstral coefficients.
    The audio time series is normalised and its mfcc features are computed.
    Args:
        audio_data: time series of the speech utterance.
        sample_rate: sampling rate.
    Returns:
        mfcc_feat:[num_frames x F] matrix representing the mfcc.
    '''

    audio_data = audio_data - np.mean(audio_data)
    audio_data = audio_data / np.max(audio_data)
    mfcc_feat = mfcc(audio_data, sample_rate, winlen=FRAME_LENGTH, winstep=FRAME_SHIFT,
                     numcep=FEATURE_VECTOR_SIZE, nfilt=2*FEATURE_VECTOR_SIZE, nfft=512, lowfreq=0, highfreq=sample_rate/2,
                     preemph=0.97, ceplifter=22, appendEnergy=True)
    return mfcc_feat

def make_example(spec_feat, labels):
    ''' Creates a SequenceExample for a single utterance.
    This function makes a SequenceExample given the sequence length,
    mfcc features and corresponding transcript.
    These sequence examples are read using tf.parse_single_sequence_example
    during training.
    Note: Some of the tf modules used in this function(such as
    tf.train.Feature) do not have comprehensive documentation in v0.12.
    This function was put together using the test routines in the
    tensorflow repo.
    See: https://github.com/tensorflow/tensorflow/
    blob/246a3724f5406b357aefcad561407720f5ccb5dc/
    tensorflow/python/kernel_tests/parsing_ops_test.py
    Args:
        spec_feat: [TxF] matrix of mfcc features.
        labels: list of words representing the encoded transcript.
    Returns:
        Serialized sequence example.
    '''
    
    # Feature lists for the sequential features of the example
    feature_lists = tf.train.FeatureLists(feature_list={
        'labels': tf.train.FeatureList(feature=[
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[p.encode()]))
            for p in labels
        ]),
        'inputs': tf.train.FeatureList(feature=[
            tf.train.Feature(float_list=tf.train.FloatList(value=frame))
            for frame in spec_feat
        ])
    })

    ex = tf.train.SequenceExample(feature_lists=feature_lists)

    return ex.SerializeToString()

def parse_args():
    parser = ArgumentParser('Process Single input file')

    parser.add_argument('--input', help='input file to be processed')
    parser.add_argument('--output_file', help='output file path or name')
    args = parser.parse_args()

    return args

def main(args):
    # open sound file
    audio, sample_rate = sf.read(args.input)
    # compute mfcc
    features = compute_mfcc(audio, sample_rate)
    # put the features and 'no label' into a tfrecords file
    tfrecord_data = make_example(features, ' UNKNOWN')
    # output file
    record_writer = tf.python_io.TFRecordWriter(args.output_file)
    record_writer.write(tfrecord_data)
    record_writer.close()

if __name__ == '__main__':
    main(parse_args())
