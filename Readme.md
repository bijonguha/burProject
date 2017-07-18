# Object detection and finding difference


## Getting Started  
  
Clone this repository and download all files

### Prerequisites

>Anaconda3-4.4.0-Windows-x86_64,
>bzip2=1.0.6=vc14_3,
>cycler=0.10.0=py35_0,
>freetype=2.5.5=vc14_2,
>icu=57.1=vc14_0,
>jpeg=9b=vc14_0,
>libpng=1.6.27=vc14_0,
>libtiff=4.0.6=vc14_3,
>matplotlib=2.0.2=np113py35_0,
>mkl=2017.0.3=0,
>numpy=1.13.1=py35_0,
>olefile=0.44=py35_0,
>openssl=1.0.2l=vc14_0,
>pillow=4.2.1=py35_0,
>pip=9.0.1=py35_1,
>pyparsing=2.1.4=py35_0,
>pyqt=5.6.0=py35_2,
>python=3.5.3=3,
>python-dateutil=2.6.0=py35_0,
>pytz=2017.2=py35_0,
>qt=5.6.2=vc14_5,
>scikit-learn=0.18.2=np113py35_0,
>scipy=0.19.1=np113py35_0,
>setuptools=27.2.0=py35_1,
>sip=4.18=py35_0,
>six=1.10.0=py35_0,
>tk=8.5.18=vc14_0,
>vs2015_runtime=14.0.25420=0,
>wheel=0.29.0=py35_0,
>zlib=1.2.8=vc14_3,
>opencv3=3.1.0=py35_0,

## One step solution
```
###Windows

Download Anaconda3-4.4.0-Windows-x86_64 and install it:

Create environment:
>conda env create -f py35_opencv_with_contrib_WIN.yml

to use environment:
>activate py35
>deactivate py35

###Linux 

Installing Anaconda
Link: https://www.digitalocean.com/community/tutorials/
how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04

Create environment:
>conda env create -f py35_opencv_with_contrib_LINUX.yml

to use environment:
>source activate py35
>source deactivate py35

```

## Running the tests

```
Examples:  

>python main.py -h  

ROI Detection Evaluation

positional arguments:
  QRY                   Query Image
  REF                   Reference Image
  Q_REF                 Reference of Query Image
  N_QRY                 Name of Query Image

optional arguments:
  -h, --help            show this help message and exit
  --algo ALGO, -a ALGO  algorithms : SIFT_Rect | find_defect (default:
                        find_defect)
  --thresh T, -t T      Threshold between 0.45 and 0.8 (default : 0.70)
  --view                View result
```

```
python main.py <REF_path> <QRY_path> <Q_REF_path> <name_of_query_image>

Ex 1: python main.py ../data/1010a.jpg ../data/6004.jpg ../data/6000.jpg 6004

Ex 2: python main.py ../data/bura_dig.jpg ../data/bura_disturb.jpg ../data/
bura_good.jpg bura_disturb  -a diff_approach

```

<h3>Work Flow Diagram [Find_defects]</h3>
<img src="https://github.com/bijonguha/burProject/blob/master/documentation/flow_of_algo.jpg?raw=true" alt="work_flow">



