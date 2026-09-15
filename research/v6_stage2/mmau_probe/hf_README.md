---
license: cc-by-nc-4.0
---
Here is the dataset repo of paper "Abductive Ego-View Accident Video Understanding for Safe Driving Perception"

The Github repo: [Link](https://github.com/jeffreychou777/LOTVS-MM-AU)

Due to the large amount of data, chunked compression is used before uploading.

After downloading the data, You need to merge the files before extracting them.

```
#Take DADA2000 as an example

cd DADA-2000_chunks
cat DADA2000.part_* > DADA2000.tar.gz
tar -xzvf DADA2000.tar.gz

```

After decompression, please check the completeness of your download data and make the file structured as following:
```
MM-AU # root of your MM-AU
├── CAP-DATA
│   ├── 1-10    #total 1556 video sequences, takes 44GB
│       ├── 1
│           ├── 001537/images
│               ├── 000001.jpg
│               ├── ......
│       ├── 2
│       ├── ......
│       ├── 10
│   ├── 11       #total 3083 video sequences, takes 96GB
│   ├── 12-42    #total 1629 video sequences, takes 45GB
│   ├── 43       #total 2150 video sequences, takes 44GB
│   ├── 44-62    #total 1350 video sequences, takes 30GB
│   ├── cap_text_annotations.xls
├── DADA-DATA    #total 1962 video sequences, takes 131GB
│   ├── 1
│       ├── 001/images
│           ├── 0001.png
│           ├── ......
│   ├── 2
│   ├── ......
│   ├── 61
│   ├── dada_text_annotations.xlsx
```

NEW!: The coco style datasets for object detection task has been uploaded!

Note: The object detection data used in the paper and the improved version MMAU-Detectv1 differ in both file names and number of videos due to different data cleaning methods and organization, but both maintain the same cocodataset style and the same dataset division strategy. The dataset used in the paper is provided to ensure the reproducibility of our paper, while the organization of MMAU-Detectv1 allows for better access to the video and image metadata when needed.

Note: There are 295013 items(48GB) in MMAU_det_paper/train, 64745 items(11GB) in MMAU_det_paper/test, 62731 items(9.9GB) in MMAU_det_paper/val, 299015(42GB) items in MMAU_detv1/train, 63473 items(9.0GB) in MMAU_detv1/test and 65386 items(9.1GB) in MMAU_detv1/val. After decompression, please check the completeness of your download data.


NEW!:The checkpoints in the LOTVS-CAP Github repo has been uploaded!

Include:

1.bert model to encode the text.Download the bert-base-uncased-pytorch_model

2.the inference model on the MINI-Test evaluation

3.inference model on the FULL-Test evaluation