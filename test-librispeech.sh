MODEL_DIR=./model
VOCAB_PATH=./tfdata/vocab.table

INPUT_FILE="$1"

if [ ! -f "$INPUT_FILE" ]; then
	echo "Unable to open $INPUT_FILE"
	echo "You need to provide an input file"
	echo "Usage example:\n./test-librispeech sample.mp3"
	exit 1
fi

TFILE=$(mktemp ./XXXXXX.tfrecord)
SFILE=$(mktemp ./resultXXXXX.txt)

echo "Preprocess $INPUT_FILE"
python3.6 testing/preprocess.py --input $INPUT_FILE  --output_file $TFILE

echo "Do inference"
python3.6 infer.py --data $TFILE --vocab $VOCAB_PATH --model_dir $MODEL_DIR --save $SFILE --beam_width 10 --batch_size 1

rm -f "$TFILE"

echo "Inference result written to $SFILE"
