# Predicting Kickstarter Campaign Success
# By: Sharon Agai

### Introduction
Crowdfunding has exploded in popularity over the past decade. Although the concept of crowdfunding has been around since the 1700s, online crowdfunding emerged in 2001 (ArtistShare) and really began to take off when IndieGogo, Kickstarter, and GoFundMe entered the crowdfunding industry in 2007, 2009, and 2010 respectively. [source](https://www.startups.co/articles/history-of-crowdfunding)
These companies provided better platforms for crowdfunding and did so at the right time. Under the pressure of the 2008 recession, crowdfunding was a way for people to raise money without using an official financial institution as an intermediary.
In the past few years, crowdfunding has become political. Healthcare-related crowdfunding campaigns are commonly used to highlight issues in the American healthcare system [NPR article about this](https://www.npr.org/sections/health-shots/2018/12/27/633979867/patients-are-turning-to-gofundme-to-fill-health-insurance-gaps). Most recently, the GoFundMe campaigns to build the proposed border wall and protests against it (digging tunnels under the proposed wall, or using ladders to climb over it) have [caught national attention](https://www.bbc.com/news/world-us-canada-46657393).

### What's in this repository?
[ks-projects-201801.csv:](ks-projects-201801.csv) This dataset is from Kaggle, originally [here](https://www.kaggle.com/kemical/kickstarter-projects#ks-projects-201801.csv)

[RFclassifier_executable.py:](RFclassifier_executable.py) An command-line executable Python file that takes in a dataset ("ks-projects-201801.csv", in this case) and outputs a cross-validation score, a confusion matrix, and a dataframe of feature importances for the random forest classifier.

[RFclassifier_codewithexplanation.ipynb:](RFclassifier_codewithexplanation.ipynb) A Jupyter notebook with all my code, reasoning, and visualizations. I fully explained my thought process and each output here.

[RFclassifier_codewithexplanation_v2.ipynb:](RFclassifier_codewithexplanation_v2.ipynb) Some improvements made to RFclassifier_codewithexplanation.ipynb.

[visualizing_a_decision_tree.pdf:](visualizing_a_decision_tree.pdf) A second decision tree visualization; the other is embedded in the Jupyter notebook.
