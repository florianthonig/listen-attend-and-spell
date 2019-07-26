import os
import sys
import string
from argparse import ArgumentParser

import numpy as np
import scipy.io.wavfile as wav

try:
    import speechpy
except:
    raise ImportError('Run`pip install speechpy` first')

try:
    import soundfile as sf
except:
    raise ImportError('Run`pip install soundfile` first')


s = set()


def parse_args():
    parser = ArgumentParser('Process LibriSpeech dataset')

    parser.add_argument('--data_dir', help='root directory of LibriSpeech dataset')
    parser.add_argument('--output_dir', help='output directory of processed dataset')

    args = parser.parse_args()

    return args

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
    
    s.update(labels)
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

    feats = {}
    transcripts = {}

    for filename in tqdm(glob.iglob(partition+'/**/*.txt', recursive=True)):
        with open(filename, 'r') as file:
            for line in file:
                parts = line.split()
                audio_file = parts[0]
                file_path = os.path.join(os.path.dirname(filename),
                                         audio_file+'.flac')
                audio, sample_rate = sf.read(file_path)
                feats[audio_file] = compute_mfcc(audio, sample_rate)
                target = ' '.join(parts[1:])
                transcripts[audio_file] = target
    
    return feats, transcripts

def create_records(audio_path, output_path):
    """ Pre-processes the raw audio and generates TFRecords.
    This function computes the mfcc features, encodes string transcripts
    into integers, and generates sequence examples for each utterance.
    Multiple sequence records are then written into TFRecord files.
    """
    for partition in sorted(glob.glob(audio_path+'/*')):
        if os.path.basename(partition) == 'processed':
            continue
        
        print('Processing' + partition)
        feats, transcripts, utt_len = process_data(partition)
        sorted_utts = sorted(utt_len, key=utt_len.get)
        # bin into groups of 100 frames.
        max_t = int(utt_len[sorted_utts[-1]]/100)
        min_t = int(utt_len[sorted_utts[0]]/100)

        # Create destination directory
        write_dir = output_path
        if tf.io.gfile.exists(write_dir):
            tf.gfile.DeleteRecursively(write_dir)
        tf.io.gfile.makedirs(write_dir)

        # Create single TFRecord for dev and test partition
        filename = os.path.join(write_dir, os.path.basename(write_dir) +
                                '.tfrecords')
        print('Creating', filename)
        record_writer = tf.python_io.TFRecordWriter(filename)
        for utt in tqdm(sorted_utts):
            example = make_example(feats[utt].tolist(),
                                    transcripts[utt])
            record_writer.write(example)
        record_writer.close()
        print('Processed '+str(len(sorted_utts))+' audio files')
            
def main(args):
    create_records(args.data_dir, args.output_dir)
    d = sorted(list(s))
    with open(args.output_dir+'/vocab.table', 'w') as f:
        print('\n'.join(d), file=f)

if __name__ == '__main__':
    main(parse_args())
