# Prediction-Of-Solar-Power-System-Coverage
Prediction of Solar Power System Coverage with from DeepSolar database, a solar installation database for the United States.


The data is a subset of the DeepSolar database, a solar installation database for the US, built by extracting information from satellite images. Photovoltaic panel installations are identified from over one billion image tiles covering all urban areas as well as locations in the US by means of an advanced machine learning framework. Each image tile records the amount of solar panel systems (in terms of panel surface and number of solar panels) and is complemented with features describing social, economic, environmental, geographical, and meteorological aspects. As such, the database can be employed to relate key environmental, weather and socioeconomic factors with the adoption of solar photovoltaics energy production.
More information about this database is at the link:
http://web.stanford.edu/group/deepsolar/home
The dataset data_project_deepsolar.csv contains a subset of the DeepSolar database. Each row of the dataset is a “tile” of interest, that is an area corresponding to a detected solar power system, constituted by a set of solar panels on top of a building or at a single location such as a solar farm. For each system, a collection of features record social, economic, environmental, geographical, and meteorological aspects of the tile (area) in which the system has been detected. Information about the features are in the file data_project_deepsolar_info.csv.



Task: Predict solar power system coverage
The target variable is solar_system_count. This variable is a binary variable indicating the coverage of solar power systems in a given tile. The variable takes outcome low if the tile has a low number of solar power systems (less than or equal to 10), while it takes outcome high if the tile has a large number of solar power systems (more than 10).
