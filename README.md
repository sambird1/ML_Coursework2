# ML_Coursework2
Hi guys.

I've set up a workspace for our project here.

Project spec says needs to be compatible to run in Linux.

Let's try to comment on our commits so we know what we did at which point and why. Any useful or relevant information we could add in the README.

Cheers,

Gerard



Files:

main.py
-----------
Python script to load in dataset.

config_example.json
-------------------
Contains example format for config.json file (not tracked) so individual team members can set their own local
filepath locations without needing to amend the main python file.

alternate_read.py
-----------------
5 line read in for pandas DataFrames. Input are 5 csv files for:
IMAGES AND CLASS LABELS & BOUNDING BOXES in bird_labs.csv
PART LOCATIONS in parts.csv
ATTRIBUTE LABELS in attributes.csv, certainties.csv & class_labs_continuous.csv
(NEEDS TO BE VALIDATED -- IN PROGRESS)

create_csv.py
---------------
Contains code used to concatenate the new csv files.
