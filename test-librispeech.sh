MODEL_DIR=./tensorboard/libri-vctk-hparams-full-corpus/
VOCAB_PATH=./vocab/vocab_libri.table

INPUT_FILE="$1"

if [ ! -f "$INPUT_FILE" ]; then
	echo "Unable to open $INPUT_FILE"
	echo "You need to provide an input file"
	echo "Usage example:\n./test-librispeech sample.flac"
	exit 1
fi

TFILE=$(mktemp ./XXXXXX.tfrecord)
SFILE=$(mktemp ./result-XXXXX.txt)

echo "Preprocess $INPUT_FILE"
python3 testing/preprocess.py --input $INPUT_FILE  --output_file $TFILE

echo "Do inference"
python3 infer.py --data $TFILE --vocab $VOCAB_PATH --model_dir $MODEL_DIR --save $SFILE --beam_width 50 --batch_size 1

rm -f "$TFILE"

echo "Inference result written to $SFILE"
