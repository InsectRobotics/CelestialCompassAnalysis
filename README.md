# Celestial Compass Analysis ![GitHub top language](http://img.shields.io/github/languages/top/InsectRobotics/CelestialCompassAnalysis) ![GitHub license](http://img.shields.io/github/license/InsectRobotics/CelestialCompassAnalysis) ![GitHub last commit](http://img.shields.io/github/last-commit/InsectRobotics/CelestialCompassAnalysis)

Code that creates the analyses the data and figures of the article:

**Gkanias, E., Mitchell, R., Stankiewicz, J., Khan, S.R., Mitra, S., and Webb, B. (2023).
*Celestial compass design mimics the fan-like polarisation filter array of insect eyes***.
*Nature Communications*.


## Analyse the data and create the processed datasets

The processed data can be downloaded from [here](). Alternatively, the data can be created
by copying the ROS-bag files in the repositories workspace and following the instructions
below.

Open a terminal, navigate to the [templates](templates) directory, and run:
```commandline
python create_csv.py -t raw
```
This will create a CSV file in the ```csv``` directory (named ```raw_dataset.csv```) that
contains all the important data from the ```sardinia_data``` and ```south_africa_data```
directories (that contain the raw ROS-bag files). The option ```-o [output_path]``` allows
for a different output path. To create the pooled data from the above CSV file, run:
```commandline
python create_csv.py -t pooled
```
which will create another CSV file in the ```csv``` directory (named ```pooled_dataset.csv```).
Again the output path can be changed using the ```-o [output_path]``` option. This process
needs the ```csv/raw_dataset.csv```, so if an alternative output path was chosen before,
this has to be specified through the input option ```-i [input_path]```. Finally, the errors
for all the sessions and models can be calculated by running:
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

Copyright &copy; 2023, Insect robotics Group, Institute of Perception,
Action and Behaviour, School of Informatics, the University of Edinburgh.
