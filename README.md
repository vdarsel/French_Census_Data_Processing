# Data Processing from French Census Data

This Git repository describes the processing from the French Census data, described in [ADD REF WHEN AVAILABLE]. The datasets are derived from the French Institute for Statistics (INSEE), using the 2021 Census data: [link](https://www.insee.fr/fr/statistiques/8268848). 

Processed datasets are available to this link: [Data](https://data.mendeley.com/datasets/p2gcy7x7sd/1)

Details on the preprocessing are available in [ADD REF]. 

The datasets are suitable for a direct usage in [PopulationSynthesis](https://github.com/vdarsel/PopulationSynthesis).

## Variables 
In  [ADD REF], we present different data scenarios with different complexities, to adapt to be in line with the complexity of the real use case. Here is the list of the variables and the different use cases.

Different data scenarios are provided to test different model complexities. 

| Category               | Variable name       | Type        | # of modalities | Basic | Socio | Extended |
|------------------------|---------------------|-------------|-----------------|-------|-------|------|
| **Person attributes**  | Age                 | Integer     | 100             | ✓     | ✓     | ✓    |
|                        | Sex                 | Binary      | 2               | ✓     | ✓     | ✓    |
|                        | Diploma             | Categorical | 9               | ✓     | ✓     | ✓    |
|                        | Marital             | Categorical | 6               |       | ✓     | ✓    |
|                        | Cohabitation        | Binary      | 2               |       |       | ✓    |
|                        | Employment          | Categorical | 14              |       |       | ✓    |
|                        | Socioprofessional   | Categorical | 8               |       | ✓     | ✓    |
|                        | Activity            | Categorical | 18              |       | ✓     | ✓    |
|                        | Hours               | Categorical | 3               |       |       | ✓    |
|                        | Transport           | Categorical | 7               |       |       | ✓    |
|                        | ReferenceLink       | Categorical | 10              |       |       | ✓    |
|                        | FamilyLink          | Categorical | 5               |       | ✓     | ✓    |
| **Household attributes** | HouseholdSize      | Integer     | 18              | ✓     | ✓     | ✓    |
|                        | nChildren           | Integer     | 13              |       |       | ✓    |
|                        | nRooms              | Integer     | 20              |       |       | ✓    |
|                        | Surface             | Integer     | 7               |       |       | ✓    |
|                        | Parking             | Binary      | 2               |       |       | ✓    |
|                        | nCars               | Integer     | 4               |       | ✓     | ✓    |
|                        | Accommodation       | Categorical | 7               |       | ✓     | ✓    |
|                        | Household           | Categorical | 5               |       | ✓     | ✓    |
|                        | Occupancy           | Categorical | 6               |       |       | ✓    |
|                        | HouseholdID         | Categorical | 5,576,102       |       |       |      |
| **Geographical**       | Department          | Categorical | 8               | ✓     |       |      |
|                        | County              | Categorical | 181             |       | ✓     |      |
|                        | City                | Categorical | 416             |       |       |      |
|                        | TRIRIS              | Categorical | 1350            |       |       | ✓    |
|                        | IRIS                | Categorical | 4315            |       |       | ✓    |
