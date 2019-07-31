import os
import sys
import string
from argparse import ArgumentParser
import glob
from python_speech_features import mfcc
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf

try:
    import speechpy
except:
    raise ImportError('Run`pip install speechpy` first')

try:
    import soundfile as sf
except:
    raise ImportError('Run`pip install soundfile` first')

FRAME_LENGTH = 0.025
FRAME_SHIFT = 0.01
FEATURE_VECTOR_SIZE = 39 # this implementation used 39 features

s = set()


def parse_args():
    parser = ArgumentParser('Process LibriSpeech dataset')

    parser.add_argument('--data_dir', help='root directory of LibriSpeech dataset')
    parser.add_argument('--output_dir', help='output directory of processed dataset')

    args = parser.parse_args()

    return args

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
    
    s.update(labels.split(' '))
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

def process_data(partition):
    """ Reads audio waveform and transcripts from a dataset partition
    and generates mfcc featues.
    Args:
        parition - represents the dataset partition name.
    Returns:
        feats: dict containing mfcc feature per utterance
        transcripts: dict of lists representing transcript.
        utt_len: dict of ints holding sequence length of each
                 utterance in time frames.
    """

    feats = []
    transcripts = []

    for filename in glob.iglob(partition+'/**/*.txt', recursive=True):
        with open(filename, 'r') as file:
            for line in file:
                parts = line.split()
                audio_file = parts[0]
                file_path = os.path.join(os.path.dirname(filename),
                                         audio_file+'.flac')
                audio, sample_rate = sf.read(file_path)
                feats.append(compute_mfcc(audio, sample_rate))
                s = ' '.join(parts[1:])
                translator = str.maketrans('', '', string.punctuation)
                transcripts.append(s.translate(translator).lower())
    
    return feats, transcripts

def create_records(audio_path, output_path):
    """ Pre-processes the raw audio and generates TFRecords.
    This function computes the mfcc features, encodes string transcripts
    into integers, and generates sequence examples for each utterance.
    Multiple sequence records are then written into TFRecord files.
    """
    for partition in sorted(glob.glob(audio_path+'/*')):        
        print('Processing' + partition)
        feats, transcripts = process_data(partition)
        # Create destination directory
        write_dir = output_path
        # Create single TFRecord for dev and test partition
        filename = os.path.join(write_dir, os.path.basename(partition) +
                                '.tfrecords')
        print('Creating', filename)
        record_writer = tf.python_io.TFRecordWriter(filename)
        for utt in range(len(transcripts)):
            example = make_example(feats[utt],
                                    transcripts[utt])
            record_writer.write(example)
        record_writer.close()
        print('Processed '+str(len(transcripts))+' audio files')
            
def main(args):
    create_records(args.data_dir, args.output_dir)
    d = sorted(list(s))
    print('Creating vocabulary table')
    with open(args.output_dir+'/vocab.table', 'w') as f:
        print('\n'.join(d), file=f)

if __name__ == '__main__':
    main(parse_args())
