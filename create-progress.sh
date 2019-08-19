#!/bin/sh

prefix=$(date +%s)
name=model-$prefix.tar

# create the archive
tar -cf $name ./model/eval
# add the events from the "normal" line
tar -rf $name ./model/events.out.tfevents.*

echo "Archive '$name' created"
