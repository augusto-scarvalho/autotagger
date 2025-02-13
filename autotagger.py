import cv2
import numpy as np
import os
import csv
import copy
import argparse
from glob import glob
from tqdm import tqdm

tags = None
include_characters = False
tag2index = {}
import sqlite3

try:
    with open("tf_paths.txt", "r") as f:
        TF_MODELS = f.readlines()
except:
    TF_MODELS = []

try:
    with open("trt_paths.txt", "r") as f:
        TRT_MODELS = f.readlines()
except:
    TRT_MODELS = []


def tags_no_characters():
    conn = sqlite3.connect('tags.db')
    command = f"SELECT name FROM tags WHERE category='0' ORDER BY order_id ASC"
    cur = conn.cursor()
    cur.execute(command)
    rows = cur.fetchall()
    tagout = np.array([row[0] for row in rows])
    return tagout


def tags_with_characters():
    conn = sqlite3.connect('tags.db')
    command = f"SELECT name FROM tags WHERE category='0' OR category='4' ORDER BY order_id ASC"
    cur = conn.cursor()
    cur.execute(command)
    rows = cur.fetchall()
    tagout = np.array([row[0] for row in rows])
    return tagout


def process_probabilities(probabilities, files, threshold, filter,append_to_front, leave_underscores):
    global include_characters, tags
    for i, file in enumerate(files):

        probs = probabilities[i]
        spl = file.split('.')
        txt_file = '.'.join(spl[0:-1]) + '.txt'
        file_name = os.path.basename(file)
        if include_characters:
            probs = probs[4:9083] * filter
        else:
            probs = probs[4:6951] * filter
        passedprobs = probs >= threshold
        passedtags = tags[passedprobs]
        passedconfidences = probs[passedprobs]
        sorted_confidecnes = np.argsort(-1 * passedconfidences)
        passedtags = passedtags[sorted_confidecnes]
        appender=copy.copy(append_to_front)
        appender.extend(passedtags)
        appender = [x.replace("_", " ") for x in appender] if not leave_underscores else appender
        outtext = ', '.join(appender)
        wrt = open(txt_file, 'w')
        wrt.write(outtext)
        wrt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process folder')
    parser.add_argument('--image_dir', help='Input directory containing imagesfiles', required=True)
    parser.add_argument("--include_characters", action="store_true", help="enable the ability to ")
    parser.add_argument('--tag_threshold', default='.3')
    parser.add_argument('--model_path',required=True)
    parser.add_argument("--use_tensorrt", action="store_true", help="if you wnt to use ")
    parser.add_argument("--exclude_tags", nargs="*", help="the tags to exclude from being tagged in ")
    parser.add_argument("--append_tags", nargs="*", help="tags to append the the front of the caption, automatic , at end")
    parser.add_argument("--leave_underscores", default=False, action="store_true", help="leaves the underscores in the tags")
    parser.add_argument("--greedy", default=False, action="store_true", help="gets the max probabilities for each tag instead of taking the mean")

    args = parser.parse_args()
    if args.append_tags is None:
        append_to_front=[]
    else:
        append_to_front=args.append_tags

    if args.exclude_tags is None:
        exclude_tags=[]
    else:
        exclude_tags=args.exclude_tags

    include_characters = args.include_characters
    threshold = float(args.tag_threshold)
    if not include_characters:
        tags = tags_no_characters()
    else:
        tags = tags_with_characters()

    for i, tag in enumerate(tags):
        tag2index[tag] = i

    filter = np.ones(len(tags), dtype=np.float32)
    for tag in exclude_tags:
        if tag in tag2index:
            index = tag2index[tag]
            filter[index] = 0

    if args.use_tensorrt:
        from trtmodel import TrtTagger
        # tagger = TrtTagger(args.model_path) # original code
        taggers = [TrtTagger(modelpath) for modelpath in TRT_MODELS]

    else:
        from tfmodel import TFTagger
        # tagger = TFTagger(args.model_path) # original code
        taggers = [TFTagger(modelpath) for modelpath in TF_MODELS]

    image_dir = args.image_dir
    filelist = []
    filelist.extend(glob(image_dir + '/*.jpg'))
    filelist.extend(glob(image_dir + '/*.jpeg'))
    filelist.extend(glob(image_dir + '/*.png'))
    filelist.extend(glob(image_dir + '/*.webp'))
    accumulated_files = []
    img_list = []
    batch_size = taggers[0].batch_size

    for file in tqdm(filelist):
        try:
            img = cv2.imread(file)
        except:
            img = None
        if not img is None:
            img_list += [img]
            accumulated_files += [file]
        if len(accumulated_files) >= batch_size:
            #output_probabilities = tagger(img_list) # original code

            output_probabilities_lst = [tagger(img_list) for tagger in taggers]

            if args.greedy:
                output_probabilities = np.max(
                    np.array(output_probabilities_lst), axis=0
                )
            else:
                output_probabilities = np.mean(
                    np.array(output_probabilities_lst), axis=0
                )
            
            process_probabilities(
                output_probabilities,
                accumulated_files,
                threshold,
                filter,
                append_to_front,
                args.leave_underscores
            )
            accumulated_files = []
            img_list = []

    if len(accumulated_files) > 0:
        Length = len(img_list)
        diff = batch_size - Length
        blank = np.zeros(
            (
                taggers[0].height,
                taggers[0].width, 3
            ),
            dtype=np.uint8
        )
        img_list.extend([blank] * diff)

        #output_probabilities = tagger(img_list)[0:Length] # original code

        output_probabilities_lst = [tagger(img_list)[0:Length] for tagger in taggers]
        if args.greedy:
            output_probabilities = np.max(
                np.array(output_probabilities_lst), axis=0
            )
        else:
            output_probabilities = np.mean(
                np.array(output_probabilities_lst), axis=0
            )

        process_probabilities(
            output_probabilities,
            accumulated_files,
            threshold,
            filter,
            append_to_front,
            args.leave_underscores
        )
        accumulated_files = []
        img_list=[]
