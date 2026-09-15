## Document Liveness Challenge (DLC-2021)
--------------------------------------------------------------------------------

Contents:

1. Intro
2. Data structure
3. Contact information
4. Journal reference

--------------------------------------------------------------------------------

1. Intro

Various government and commercial services, including but not limited to 
e-government, fintech, banking and sharing economy widely use smartphones to 
simplify service access and user authorization. Many organizations involved in 
these areas use identity document analysis systems in order to improve user 
personal data input processes. The tasks of such systems are not only ID 
document data recognition and extraction but also identity fraud prevention by 
detecting document forgery or by checking whether the document is genuine and 
real. Modern systems of such kind are often expected to operate in 
unconstrained environments. 

Significant amount of research has been published on topic of mobile ID 
document analysis, but the main difficulty for such research is the lack of 
public datasets due to the fact that the subject is protected by security 
requirements. In this paper, we present a dataset DLC-2021 which consists of 
1424 video clips captured in wide range of real-word conditions and focused on 
ID document forensics tasks. The dataset is available for download at 
ftp://smartengines.com/dlc-2021

--------------------------------------------------------------------------------

2. Structure

DLC-2021 data structure extends MIDV-2020 structure 
(https://arxiv.org/abs/2107.00396). 
Images and video from MIDV-2020 can be used as "real document" samples. 
The identity document types of MIDV-2020 are listed in Table DOCTYPES.


| N    | CODE                 | Description                 | MIDV-500 code |
| ---- | -------------------- | --------------------------- | ------------- |
| 01   | alb_id               | ID Card of Albania          | 01            |
| 02   | aze_passport         | Passport of Azerbaijan      | 05            |
| 03   | esp_id               | ID Card of Spain            | 21            |
| 04   | est_id               | ID Card of Estonia          | 22            |
| 05   | fin_id               | ID Card of Finland          | 24            |
| 06   | grc_passport         | Passport of Greece          | 25            |
| 07   | lva_passport         | Passport of Latvia          | 32            |
| 08   | rus_internalpassport | Internal passport of Russia | 39            |
| 09   | srb_passport         | Passport of Serbia          | 41            |
| 10   | svk_id               | ID Card of Slovakia         | 42            |

The original captured clips were separated into frames using ffmpeg 
version n4.4 with default parameters. To provide a rich annotation each video 
clip was annotated with 10 frames per second for the first 5 second.
Sampled frames from video clips with annotations are placed in clips.zip.

In general DLC-2021 follows MIDV-2020 folder and files structure except clip 
names. In DLC-2021 two-digit document template number clip name (NXX) is 
extended with two-letter video type code (TX) and four-digit serial number.

|Video type code| Description                                              |
|---------------|----------------------------------------------------------|
|cc             | unlaminated color copy                                   |
|cg             | unlaminated gray copy                                    |
|or             | "original" laminated documents from MIDV-2020 collection |
|re             | video recapture for document on device screen            |


Frames are placed in clips folder.
```
/clips/
  /images/
    /<CODE>/
      /<N01>.<T1>0001/
        000001.jpg
        000007.jpg
        ...
      /<N01>.<T1>0002/
        000001.jpg
        000007.jpg
        ...
      /<N01>.<T2>0001/
        000001.jpg
        000007.jpg
        ...
      ...
      /<N02>.<T1>0001/
        000001.jpg
        000007.jpg
        ...
      /<N02>.<T2>0001/
        000001.jpg
        000007.jpg
        ...
      ...
  /annotations/
    /<CODE>/
      <N01>.<T1>0001.json
      <N01>.<T1>0002.json
      <N01>.<T2>0001.json
      ...
      <N02>.<T1>0001.json
      ...
```

All annotations are made using VGG Image Annotator (VIA) v2.0.11, which can be
obtained via this link: 
https://www.robots.ox.ac.uk/~vgg/software/via/downloads/via-2.0.11.zip
The developer's website: https://www.robots.ox.ac.uk/~vgg/software/via/

The original video clips are placed in clips_video folder.

```
/clips_video/
  /video/
    /<CODE>/
      <N01>.<T1>0001.mp4[MOV]
      <N01>.<T1>0002.mp4[MOV]
      ...
      <N01>.<T2>0001.mp4[MOV]
      ...
      <N02>.<T1>0001.mp4[MOV]
      ...
    ...
```

Video clips were captured with different conditions, description can be found 
in table dlc-2021.csv.

--------------------------------------------------------------------------------

3. Contact information

Any questions, complaints, etc. can be directed to: 
polevoy@smartengines.com (Dmitry  Polevoy)


--------------------------------------------------------------------------------

4. Journal reference

The article about this dataset is currently under review. When the article is
published, the full reference will be added.
