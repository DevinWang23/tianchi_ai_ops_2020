use tspdb;

create table if not exists device_info(
building_id INT NOT NULL,
device_id INT NOT NULL,
device_category CHAR(20),
device_name CHAR(20) NOT NULL,
PRIMARY KEY(building_id, device_id)
);

create table if not exists meter_reading_data(
date date NOT NULL,
meter_reading FLOAT(20,2),
device_id CHAR(20) NOT NULL,
building_id CHAR(20) NOT NULL,
PRIMARY KEY(building_id, device_id)
);

create table if not exists weather_data(
date date NOT NULL,
city CHAR(10) NOT NULL,
temp_morning INT NOT NULL,
temp_evening INT NOT NULL,
weather_morning CHAR(10) NOT NULL,
weather_evening CHAR(10) NOT NULL,
PRIMARY KEY(date, city)
);

create table if not exists building_info(
building_id INT NOT NULL,
building_name CHAR(10) NOT NULL,
building_area FLOAT(20,2) NOT NULL,
num_resident INT NOT NULL,
building_floor INT NOT NULL,
num_room INT NOT NULL,
year_built YEAR NOT NULL,
PRIMARY KEY(building_id)
);

LOAD DATA LOCAL INFILE '/home/data/device_info.csv'
INTO TABLE device_info
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA LOCAL INFILE '/home/data/building_info.csv'
INTO TABLE building_info
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA LOCAL INFILE '/home/data/meter_reading_data.csv'
INTO TABLE meter_reading_data
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA LOCAL INFILE '/home/data/weather_data.csv'
INTO TABLE weather_data
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;