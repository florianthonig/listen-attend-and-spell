ROOT_DIR=.

REPO_LS_DIR=$ROOT_DIR/librispeech
DATA_DIR=$ROOT_DIR/librispeech-corpus
DATA_PROCESSED_DIR=$ROOT_DIR/librispeech-processed
DATA_TF_DIR=$ROOT_DIR/tfdata
DATA_MODEL_DIR=$ROOT_DIR/model


mkdir -p $DATA_TF_DIR

echo Processing dataset ...
python3.6 $REPO_LS_DIR/preprocess.py --data_dir $DATA_DIR --output_dir $DATA_TF_DIR

echo Training ...
#python3.6 train.py --train $VCTK_TF_DIR/train.tfrecord \
                 --valid $VCTK_TF_DIR/test.tfrecord \
                 --vocab $VCTK_TF_DIR/vocab.table \
                 --model_dir $VCTK_MODEL_DIR \
                 --encoder_layers 3 \
                 --encoder_units 128 \
                 --decoder_layers 1 \
                 --decoder_units 128 \
                 --dropout 0.2 \
                 --batch_size 32 \
                 --use_pyramidal \
                 --embedding_size 0 \
                 --sampling_probability 0.2 \
                 --eval_secs 1200 \
                 --attention_type luong
