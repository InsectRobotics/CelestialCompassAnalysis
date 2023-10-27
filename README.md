# Celestial Compass Analysis ![GitHub top language](http://img.shields.io/github/languages/top/InsectRobotics/CelestialCompassAnalysis) [![GitHub](http://img.shields.io/github/license/insectrobotics/celestialcompassanalysis)](./LICENSE) ![GitHub last commit](http://img.shields.io/github/last-commit/InsectRobotics/CelestialCompassAnalysis) [![DOI](https://zenodo.org/badge/641965643.svg)](https://zenodo.org/doi/10.5281/zenodo.8393055)

Code that creates the analyses the data and figures of the article:

**Gkanias, E., Mitchell, R., Stankiewicz, J., Khan, S.R., Mitra, S., and Webb, B. (2023).
*Celestial compass sensor mimics the insect eye for navigation under cloudy and occluded skies***.
*Communications Engineering*.

## Clone the repository
Open a terminal, clone the project, and navigate to its working directory:
```commandline
git clone https://github.com/InsectRobotics/CelestialCompassAnalysis.git
cd CelestialCompassAnalysis
```
The processed data can be downloaded from [here](https://doi.org/10.7488/ds/6106) and
they should be placed in the [csv](csv) directory. Alternatively, they can be generated by following the instructions below.

## Analyse the data and create the processed datasets

If you have access to the ROS-bag files produced by the robot (upon request from the authors),
proceed with step 1a, otherwise proceed with step 1b.

### 1a. Generate the raw_dataset.csv
Copy the ```sardinia_data``` and ```south_africa_data``` directories
to the working directory, navigate to the [templates](templates) directory (e.g., ```cd templates```), and run the
script that creates the raw dataset:
```commandline
python create_csv.py -t raw
```
The ROS-bag files contain a lot of information, including low resolution videos captured by the fish-eye camera
during the experiments that increased the size of the datasets to almost 100 GB. The above script creates a CSV
file (named ```raw_dataset.csv```) in the [csv](csv) directory, which includes only the data used in the article
without any further processing and reduces the size of the datasets to 677.7 MB. It also creates a directory in
named ```sessions``` in the [csv](csv) directory, which contains the high-resolution fish-eye images captured at
beginning of each session (400.1 MB).

### 1b. Download the raw_dataset.csv
Navigate to the [csv](csv) directory and download the raw dataset from the above link. Alternatively, run
the below lines from the working directory: 
```commandline
cd csv
wget https://datashare.ed.ac.uk/bitstream/handle/10283/7116/raw_dataset.csv?sequence=9&isAllowed=n
wget https://datashare.ed.ac.uk/bitstream/handle/10283/7116/sessions.zip?sequence=5&isAllowed=n
cs ..
```
This will download the CSV file (```raw_dataset.csv```) and fish-eye images (```sessions.zip```)
in the ```csv``` directory. These contain all the important data from the ```sardinia_data``` and
```south_africa_data``` directories (which are excluded from the dataset because of their large size).
Finally, extract the ```sessions.zip``` in the [csv](csv) directory.

### 2. Generate the pooled_dataset.csv
Providing the ```raw_dataset.csv``` as input, should allow the following commands to work without problems.
The option ```-o [output_path]``` allows for a different output path.
To create the pooled data from the above CSV file, run:
```commandline
python create_csv.py -t pooled -i DATASET_DIR/raw_dataset.csv
```
which will create another CSV file in the ```csv``` directory (named ```pooled_dataset.csv```).
The ```-i DATASET_DIR/raw_dataset.csv``` part is optional, and if you followed step 1a or 1b, it
shouldn't be necessary.

### 3. Generate the error_dataset.csv
The errors for all the sessions and models can be calculated by running:
```commandline
python create_csv.py -t error
```
This CSV file is saved by default in ```csv/error_dataset.csv``` and needs as input the
```csv/pooled_dataset.csv``` file. If an alternative name was chosen for this file, it needs
to be specified through the ```-i [input_path]``` as before.

## Generate the plots of the articles

To generate the plots for the article, the above datasets need to be generated first. If
alternatives names or paths were used for the generated dataset files they can be set as
inputs using the ```-i [input_file]``` as an option. The plots can be generated in order
of appearance in the article by running:
```commandline
python plot_csv.py -f 2d -o png
python plot_csv.py -f 3a -o png
python plot_csv.py -f 3b -o png
python plot_csv.py -f 4a -o png
python plot_csv.py -f 4b -o png
python plot_csv.py -f 4c -o png
python plot_csv.py -f 5a -o png
python plot_csv.py -f 5b -o png
python plot_csv.py -f 5c -o png
python plot_csv.py -f 6a -o png
python plot_csv.py -f 6b -o png
python plot_csv.py -f 6c -o png
python plot_csv.py -f 7 -o png
python plot_csv.py -f 9h -o png
python plot_csv.py -f S2a -o png
python plot_csv.py -f S2b -o png
python plot_csv.py -f S2c -o png
python plot_csv.py -f S2d -o png
python plot_csv.py -f S3a -o png
python plot_csv.py -f S3b -o png
python plot_csv.py -f S4b -o png
python plot_csv.py -f S4c -o png
```
Note that the option ```-o [output_file]``` can be used either to specify the output file path
or the file extension (supported extensions are ```png```, ```jpeg```, ```jpg```, ```svg```,
```pdf```).

## Report an issue

If you have any issues installing or using the package, you can report it
[here](https://github.com/InsectRobotics/CelestialCompassAnalysis/issues).

## Author

The code was written by [Evripidis Gkanias](https://evgkanias.github.io/) and Robert Mitchell.

## Copyright

Copyright &copy; 2023, Insect Robotics Group, Institute of Perception,
Action and Behaviour, School of Informatics, the University of Edinburgh.
