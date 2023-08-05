import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from glob import glob
import cv2
import os
import numpy as np
import copy
from tqdm import tqdm
import sqlite3

import tfmodel
import trtmodel

include_characters=False
tags=None
tag2index={}

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

try:
    with open("tag_block_list.txt", "r") as f:
        TAG_BLOCK_LIST = f.readlines()
except:
    TAG_BLOCK_LIST = ""


# preprocess the images to the same dimensions used in the tagger training
# https://github.com/SmilingWolf/SW-CV-ModelZoo
# source: https://github.com/kohya-ss/sd-scripts/blob/main/finetune/tag_images_by_wd14_tagger.py
def preprocess_image(image):
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    interp = cv2.INTER_AREA if size > 448 else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (448, 448), interpolation=interp)

    image = image.astype(np.float32)
    return image


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


class ProgressBarHandler:
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar

    def update_progress(self, progress):
        self.progress_bar['value'] = progress
        self.progress_bar.update()


def load_tf_model():
    file_path = filedialog.askdirectory()
    if file_path=='':
        return
    try:
        current_models = tf_models_lst.get("0.0", "end").replace('\n','').strip()
        if current_models != "":
            tf_models_lst.insert("0.0", file_path+",")
        else:
            tf_models_lst.insert("0.0", file_path)
    except Exception as ex:
        error_popup(ex)


def load_trt_model():
    filetypes = [("TRT or Engine files", "*.trt;*.engine")]
    file_path = filedialog.askopenfilename(filetypes=filetypes)
    if file_path=='':
        return
    try:
        current_models = trt_models_lst.get("0.0", "end").replace('\n','').strip()
        if current_models != "":
            trt_models_lst.insert("0.0", file_path+",")
        else:
            trt_models_lst.insert("0.0", file_path)
    except Exception as ex:
        error_popup(ex)


def process_probabilities(probabilities, files, threshold, filter,append_to_front, \
                          leave_underscores, include_characters):
    global tags
    for i, file in enumerate(files):
        probs = probabilities[i]
        spl = file.split('.')
        txt_file = '.'.join(spl[0:-1]) + '.txt'
        probs = probs[4:9083] * filter if include_characters else probs[4:6951] * filter
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


def select_directory():
    global progress_handler, chk_var, leave_underscores_var, greedy_var
    global tags
    global TF_MODELS, TRT_MODELS

    taggers = []

    TF_MODELS = tf_models_lst.get("0.0", "end").replace('\n','').strip().split(',')
    TRT_MODELS = trt_models_lst.get("0.0", "end").replace('\n','').strip().split(',')

    if TF_MODELS == [''] and TRT_MODELS == ['']:
        error_popup("please fill in the model paths")
        return
    if TRT_MODELS != ['']:
        try:
            from trtmodel import TrtTagger
            [taggers.append(TrtTagger(modelpath)) for modelpath in TRT_MODELS]
        except:
            error_popup(f"Error while loading models from:\n {TRT_MODELS}")
    if TF_MODELS != ['']:
        try:
            from tfmodel import TFTagger
            [taggers.append(TFTagger(modelpath)) for modelpath in TF_MODELS]
        except:
            error_popup(f"Error while loading models from:\n {TF_MODELS}")
    
    folder_path = filedialog.askdirectory()
    if folder_path=='':
        return

    include_characters = chk_var.get()
    leave_underscores = leave_underscores_var.get()
    greedy_mode = greedy_var.get()
    
    exclude_tags = entry_tags.get("0.0", "end").replace('\n','').strip().split(',') if leave_underscores \
        else entry_tags.get("0.0", "end").replace('\n','').strip().replace('_',' ').split(',')

    append_to_front = entry_tags2.get("0.0", "end").replace('\n','').split(',')

    if len(append_to_front)==1:
        if append_to_front[0]=='':
            append_to_front=[]

    if not include_characters:
        tags = tags_no_characters()
    else:
        tags = tags_with_characters()
    L=(len(tags)-1)
    filter = np.ones(len(tags), dtype=np.float32)
    for tag in exclude_tags:
        if tag in tag2index:
            index = tag2index[tag]
            if index<L:
                filter[index] = 0

    try:
        threshold = float(entry_threshold.get())
    except Exception as ex:
        error_popup(ex)
        return

    
    filelist=[]
    filelist.extend(glob(folder_path+'/*.jpg'))
    filelist.extend(glob(folder_path + '/*.png'))
    filelist.extend(glob(folder_path + '/*.jpeg'))
    filelist.extend(glob(folder_path + '/*.webp'))
    total=len(filelist)
    div=100/total

    accumulated_files = []
    img_list = []
    batch_size = taggers[0].batch_size
    pbar=tqdm(total=len(filelist))

    for i,file in enumerate(filelist):
        pbar.update(1)
        num = int(div * i)
        progress_handler.update_progress(num)
        try:
            img = cv2.imread(file)
        except:
            img = None
        if not img is None:
            img_list += [img]
            accumulated_files += [file]
        img = preprocess_image(img)
        if len(accumulated_files) >= batch_size:

            output_probabilities_lst = [tagger(img_list) for tagger in taggers]
            if greedy_mode:
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
                leave_underscores,
                include_characters
            )
            accumulated_files = []
            img_list = []

    if len(accumulated_files) > 0:
        Length = len(img_list)
        diff = batch_size - Length
        blank = np.zeros((taggers[0].height, taggers[0].width, 3), dtype=np.uint8)
        img_list.extend([blank] * diff)

        output_probabilities_lst = [tagger(img_list)[0:Length] for tagger in taggers]
        if greedy_mode:
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
            leave_underscores,
            include_characters
        )
        accumulated_files = []
        img_list = []
        
    progress_handler.update_progress(0)


def save_current_config():
    with open("trt_paths.txt", "w") as f:
        f.write(trt_models_lst.get("0.0", "end").replace('\n','').strip())
    with open("tf_paths.txt", "w") as f:
        f.write(tf_models_lst.get("0.0", "end").replace('\n','').strip())
    with open("tag_block_list.txt", "w") as f:
        f.write(entry_tags.get("0.0", "end").replace('\n','').strip())


def tag_images():
    try:
        select_directory()
        print("tagging is done!")
        save_current_config()
    except Exception as e:
        error_popup(e)



def error_popup(message):
    messagebox.showerror("Error", message)


ftags=tags_with_characters()
for i, tag in enumerate(ftags):
    tag2index[tag] = i


root = tk.Tk()

btn_pick_tf_model = tk.Button(root, text="Select TF model", command=load_tf_model)
btn_pick_tf_model.grid(row=0, column=0, columnspan=1, ipadx=5, ipady=5, padx=5, pady=5)
tf_models_lst = tk.Text(root, height=5, width=50)
if TF_MODELS != [] and TF_MODELS != None:
    tf_models_lst.insert("0.0", *[x for x in TF_MODELS])
tf_models_lst.grid(row=0, column=1,ipadx=5, ipady=5, padx=5, pady=5)


btn_pick_trt_model = tk.Button(root, text="Select TensorRT model", command=load_trt_model)
btn_pick_trt_model.grid(row=1, column=0, columnspan=1, ipadx=5, ipady=5, padx=5, pady=5)
trt_models_lst = tk.Text(root, height=5, width=50)
if TRT_MODELS != [] and TRT_MODELS != None:
    trt_models_lst.insert("0.0", *[x for x in TRT_MODELS])
trt_models_lst.grid(row=1, column=1,ipadx=5, ipady=5, padx=5, pady=5)

btn_caption_dir = tk.Button(root, text="Caption Directory", command=tag_images)
btn_caption_dir.grid(row=2, column=0, columnspan=3, ipadx=5, ipady=5, padx=5, pady=5)

lbl_tags = tk.Label(root, text="Tags to exclude (comma separated)")
lbl_tags.grid(row=4, column=0,ipadx=5, ipady=5, padx=5, pady=5)
entry_tags = tk.Text(root, height=3, width=50)
if TAG_BLOCK_LIST != "":
    entry_tags.insert("0.0", TAG_BLOCK_LIST)
entry_tags.grid(row=4, column=1,ipadx=5, ipady=5, padx=5, pady=5)

lbl_tags2 = tk.Label(root, text="Tags to append (comma separated)")
lbl_tags2.grid(row=5, column=0,ipadx=5, ipady=5, padx=5, pady=5)
entry_tags2 = tk.Text(root, height=3, width=50)
entry_tags2.grid(row=5, column=1,ipadx=5, ipady=5, padx=5, pady=5)

lbl_threshold = tk.Label(root, text="Tag probability threshold")
lbl_threshold.grid(row=6, column=1,ipadx=5, ipady=5, padx=5, pady=5)
entry_threshold = tk.Entry(root, width=10)
entry_threshold.grid(row=7, column=1, ipadx=5, ipady=5, padx=5, pady=5)
entry_threshold.insert(0, '0.35')

chk_var = tk.BooleanVar()
chk_include_tags = tk.Checkbutton(root, text="Include character tags",variable=chk_var)
chk_include_tags.grid(row=8, column=0)

greedy_var = tk.BooleanVar()
chk_greedy_var = tk.Checkbutton(root, text="Greedy mode",variable=greedy_var)
chk_greedy_var.grid(row=8, column=1)

leave_underscores_var = tk.BooleanVar()
chk_leave_underscores_var = tk.Checkbutton(root, text="Keep tag underscores",variable=leave_underscores_var)
chk_leave_underscores_var.grid(row=8, column=2)

progress_bar = ttk.Progressbar(root, length=650, mode='determinate')
progress_bar.grid(row=9, column=0, columnspan=3)
progress_handler = ProgressBarHandler(progress_bar)


root.geometry("1280x768")
root.title("Auto Tagger")
root.mainloop()
