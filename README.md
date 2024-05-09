# Movie Recommendation System Project
![image](https://github.com/sangwon224/Movie_Recommendation_System_Project/blob/main/data/image.webp)

## Project Overview
Having a good recommendation system is important for the success of any streaming companies. People on average spend ~25 minutes everyday trying to find something to watch. This translate to close to 120 days of one’s life trying to find something to watch. 

In addition, one of the most successful social network/streaming platforms, TikTok and Youtube’s success is dedicated to its top-notch recommendation system, which keeps their users sticky to their platforms. 

This project aims to build a movie recommendation system based on collaborative filtering mechanism. 

## Business Problem
SAW is a movie streaming service startup that needs a recommendation system for its users.

## Data Understanding and Preparation
For this project, following dataset was utilized for the analysis:

- Movie rating dataset from GroupLens research lab at University of Minnesota

The dataset comprises of ~100k movie ratings by ~600 unique users across ~10,000 movies. The ratings are based on traditional 5 star method with the increment of 0.5 stars.

## Recommendation System Overview
Team has utilized surprise recommendation package as a foundation of the collaborative filtering-based recommendation system.

```
# Package
import surprise
from surprise.prediction_algorithms import *
from surprise import accuracy
from surprise import Reader, Dataset
```

```
# Set Up
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(df_merged[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()
```

```
# Final Model & Performance
SVD_model = SVD().fit(trainset)

SVD_RMSE = surprise.accuracy.rmse(SVD_model.test(testset))
SVD_MAE = surprise.accuracy.mae(SVD_model.test(testset))

RMSE: 0.4845
MAE:  0.3764
```
## Insights & Limitations
- Due to the nature of collaborative filtering recommendation system, recommendation will be less accurate with less data
- Hidden gems (i.e., a movie that only handful amount of people watched, yet has high rating and quality) is less likely to be recommended
- Newer releases are less likely to be recommended due to limited rating counts

## Future Improvements
Consider following aspects for the next version of the recommendation system:
- Higher weight for user's latest rating
- Diversity (e.g., ethnicity, sexual orientation)

## For More Information
Please review our full analysis in jupyter notebook ([Movie Recommendation System Project](https://github.com/sangwon224/Movie_Recommendation_System_Project/blob/main/Movie_Recommendation_System_Project.ipynb))\
And also refer to our ([Presentation](https://github.com/sangwon224/Movie_Recommendation_System_Project/blob/main/presentation.pdf)) 

## Contributors
[Sang-won Shim](https://github.com/sangwon224)

[Anthony Mansion](https://github.com/MansionAnthony)

[William Howard](https://github.com/WilliamHowardGit)

## Repository Structure
```
|— .gitignore                                                <- gitignore exclude selected file execute
|— README.md                                                 <- The top-level README for reviewers of this project
|— Movie_Recommendation_System_Project.ipynb                 <- Interactive computing environment including analysis in Jupyter notebook
|— Data                                                      <- Both sourced externally and generated from code
    |— README.txt                                            <- Readme for raw data
    |— images.webp                                           <- Image file for the readme
    |— links.csv                                             <- Raw data
    |— movies.csv                                            <- Raw data
    |— ratings.csv                                           <- Raw data
    |— tags.csv                                              <- Raw data
|_ presentation.pdf                                          <- PDF version of project presentation
```
